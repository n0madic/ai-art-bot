import config
import dataclasses
import diffusion
import enhancement
import logging
import os
import prompt
import re
import queue
import PIL
import random
import sys
import telebot
import textwrap
import threading
import time
import torch
import tweepy


@dataclasses.dataclass
class Job:
    prompt: str
    target_chat: str
    seed: int = 0
    scale: float = 0
    steps: int = 0
    image: PIL.Image.Image = None
    message_id: int = 0
    delete_message: int = 0

    def __post_init__(self):
        params = {}
        self.prompt = re.sub(r'(scale)[:=]\s?(\d+\.\d+)', lambda m: params.update({m.group(1).lower(): float(m.group(2))}) or '', self.prompt)
        self.prompt = re.sub(r'(seed|steps)[:=]\s?(\d+)', lambda m: params.update({m.group(1).lower(): int(m.group(2))}) or '', self.prompt)
        self.prompt = re.sub(r'[(|]\s*[)|]','', self.prompt)
        self.prompt = self.prompt.strip()
        self.seed = params.get('seed', self.seed or random.randint(0, 2**32 - 1))
        self.scale = params.get('scale', self.scale or round(random.uniform(7,10), 1))
        self.steps = params.get('steps', self.steps or random.randint(20,100))


class Bot:
    def __init__(self) -> None:
        logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        self.logger = logging.getLogger('bot')
        self.logger.setLevel(logging.INFO)

        self.cfg = config.load()

        self.prompt = prompt.Prompt(self.cfg['prompt_model_id'],
                                    self.cfg['prompt_model_tokenizer'],
                                    self.cfg['sd_model_id'],
                                    self.cfg['prompt_prefix'],
                                    )
        self.enhancement = enhancement.Enhancement(self.cfg['face_enhancer_model_path'],
                                                   self.cfg['face_enhancer_arch'],
                                                   self.cfg['realesrgan_model_path'],
                                                   )
        self.__init_pipeline()

        self.bot = telebot.TeleBot(self.cfg['telegram_token'], parse_mode='HTML')
        self.bot.add_custom_filter(telebot.custom_filters.ChatFilter())

        if self.cfg['twitter_consumer_key'] and self.cfg['twitter_consumer_secret'] and self.cfg['twitter_access_token'] and self.cfg['twitter_access_token_secret']:
            try:
                auth = tweepy.OAuthHandler(self.cfg['twitter_consumer_key'], self.cfg['twitter_consumer_secret'])
                auth.set_access_token(self.cfg['twitter_access_token'], self.cfg['twitter_access_token_secret'])
                self.twitter_api_v1 = tweepy.API(auth)
                self.tw_creds = self.twitter_api_v1.verify_credentials(skip_status=True, include_entities=False)
                self.twitter_client = tweepy.Client(
                    consumer_key=self.cfg['twitter_consumer_key'], consumer_secret=self.cfg['twitter_consumer_secret'],
                    access_token=self.cfg['twitter_access_token'], access_token_secret=self.cfg['twitter_access_token_secret']
                )
            except Exception as e:
                self.logger.error("Twitter authentication: {}".format(e))
                self.twitter_api_v1 = None
            else:
                self.logger.info('Logged in Twitter as {}'.format(self.tw_creds.screen_name))
        else:
            self.twitter_api_v1 = None

        self.worker_queue = queue.Queue()

        if not os.path.exists(self.cfg['image_cache_dir']):
            os.mkdir(self.cfg['image_cache_dir'])

        self._init_commands()

    def __init_pipeline(self):
        """Initialize diffusion pipeline"""
        self.pipe = diffusion.Pipeline(self.cfg['sd_model_id'],
                                       self.cfg['sd_model_vae_id'],
                                       self.cfg['sd_refiner_id'],
                                       self.cfg['fp16'],
                                       self.cfg['low_vram'],
                                       )

    def clean_cache(self, age=86400, interval=3600):
        """Clean image cache"""
        for f in os.listdir(self.cfg['image_cache_dir']):
            if os.stat(os.path.join(self.cfg['image_cache_dir'],f)).st_mtime < time.time() - age:
                os.remove(os.path.join(self.cfg['image_cache_dir'],f))
        threading.Timer(interval, self.clean_cache).start()

    def twitter_send(self, image_path, message):
        """Send image to Twitter"""
        message = message.splitlines()[0] + '\n#AIart #stablediffusion'
        status = textwrap.shorten(message, width=280, placeholder='...')
        self.logger.info('Send image to Twitter...')
        try:
            media = self.twitter_api_v1.media_upload(image_path)
            resp = self.twitter_client.create_tweet(text=status, media_ids=[media.media_id])
        except Exception as e:
            self.logger.error(e)
        else:
            self.logger.info("https://twitter.com/{}/status/{}".format(self.tw_creds.screen_name, resp.data['id']))
            return True
        return False

    def _init_commands(self):
        """Initialize methods to represent bot commands"""
        self.start = self.bot.message_handler(chat_id=self.cfg['telegram_admin_ids'], commands=['start', 'help'])(self._start_command)
        self.die = self.bot.message_handler(chat_id=self.cfg['telegram_admin_ids'], commands=['die'])(self._die_command)
        self.change_config = self.bot.message_handler(chat_id=self.cfg['telegram_admin_ids'], commands=['config'])(self._change_config_command)
        self.generate = self.bot.message_handler(chat_id=self.cfg['telegram_admin_ids'])(self._generate_command)
        self.update_file = self.bot.message_handler(chat_id=self.cfg['telegram_admin_ids'], content_types=['document'], func=lambda m: m.document.file_name.endswith('.txt'))(self._file_update_command)
        self.callback_query = self.bot.callback_query_handler(func=lambda call: call.from_user.id in self.cfg['telegram_admin_ids'])(self._callback_query_command)

    def _start_command(self, message):
        """Send start message"""
        self.bot.send_message(message.chat.id, 'Just type the text prompt for image generation\n\nUse the <code>+</code> symbol at the end of the query to expand it with random data, for example:\n<code>cat+</code>')

    def _die_command(self, message):
        """Stop bot"""
        self.bot.send_message(message.chat.id, 'Bye!')
        os._exit(0)

    def _change_config_command(self, message):
        """Change config parameter"""
        parameter = message.text.split()[1].lower()
        value = message.text.split()[2]
        if parameter in self.cfg:
            old_value = self.cfg[parameter]
        else:
            old_value = None
        if old_value:
            if type(old_value) == bool:
                value = True if value.lower() in ['true', 'on', 'yes', '1'] else False
            elif type(old_value) == int:
                value = int(value)
            elif type(old_value) == float:
                value = float(value)
            self.cfg[parameter] = value
            value = self.cfg[parameter]
            self.bot.send_message(message.chat.id, 'Parameter {} changed to {}'.format(parameter, value))
            if parameter in ['sd_model_id', 'sd_model_vae_id', 'sd_refiner_id', 'fp16', 'low_vram']:
                self.logger.info('Reloading pipeline...')
                self.__init_pipeline()
                self.pipe.load_pipe()
        else:
            self.bot.send_message(message.chat.id, 'Parameter {} not found'.format(parameter))

    def _generate_command(self, message):
        """Generate image"""
        prompt = message.text.strip()
        if prompt == "":
            self.bot.send_message(message.chat.id, 'Please provide a prompt')
        else:
            batch = re.findall(r'batch=(\d+)', prompt)
            if batch:
                batch = int(batch[0])
                prompt = re.sub(r'batch=(\d+)', '', prompt).strip()
            else:
                batch = 1
            for i in range(batch):
                if not prompt or prompt.endswith('+'):
                    job = Job(self.prompt.generate(prompt.removesuffix('+'), random_prompt_probability=self.cfg['random_prompt_probability']), message.chat.id)
                else:
                    job = Job(prompt, message.chat.id)
                job.seed += i
                self.worker_queue.put(job)
                if prompt != job.prompt or i == batch - 1:
                    time.sleep(0.1)
                    msg = self.bot.send_message(message.chat.id, 'Put prompt <code>{}</code> in queue: {}'.format(job.prompt, self.worker_queue.qsize()), disable_notification=True)
                    job.delete_message = msg.message_id

    def _file_update_command(self, message):
        """Update txt file"""
        if not os.path.exists(message.document.file_name):
            self.bot.send_message(message.chat.id, 'File {} not found!'.format(message.document.file_name))
        else:
            file_info = self.bot.get_file(message.document.file_id)
            downloaded_file = self.bot.download_file(file_info.file_path)
            with open(message.document.file_name, 'wb') as f:
                f.write(downloaded_file)
            self.bot.send_message(message.chat.id, 'File {} updated'.format(message.document.file_name))
        self.bot.delete_message(message.chat.id, message.message_id)

    def _callback_query_command(self, call):
        """Handle callback query"""
        image_path = os.path.join(self.cfg['image_cache_dir'], '{}.jpg'.format(call.message.message_id))
        if not os.path.exists(image_path) and not call.data == 'post_to_channel':
            try:
                file_info = self.bot.get_file(call.message.photo[-1].file_id)
                downloaded_file = self.bot.download_file(file_info.file_path)
            except Exception as e:
                self.logger.error(e)
                self.bot.answer_callback_query(call.id, 'Error downloading image')
                return
            else:
                with open(image_path, 'wb') as f:
                    f.write(downloaded_file)
        if call.data == 'fix_face' or call.data == 'undo_face':
            if not os.path.exists(image_path):
                self.bot.answer_callback_query(call.id, 'Image not found')
                return
            reply_markup = call.message.reply_markup
            if call.data == 'fix_face':
                image = self.enhancement.fixface(image_path)
                os.replace(image_path, image_path + '.bak')
                image.save(image_path)
                reply_markup.keyboard[0][0] = telebot.types.InlineKeyboardButton("Undo fix", callback_data="undo_face")
            else:
                os.replace(image_path + '.bak', image_path)
                reply_markup.keyboard[0][0] = telebot.types.InlineKeyboardButton("Face fix", callback_data="fix_face")
            lines = call.message.caption.splitlines()
            caption = '\n'.join(['<code>{}</code>'.format(lines[0]), re.sub(r'\d+\.?\d+', r'<code>\g<0></code>', lines[1])])
            self.bot.edit_message_media(media=telebot.types.InputMediaPhoto(open(image_path, 'rb'), caption=caption, parse_mode='HTML'), chat_id=call.message.chat.id, message_id=call.message.message_id, reply_markup=reply_markup)
            if call.data == 'fix_face':
                self.bot.answer_callback_query(call.id, 'Face fixed')
            else:
                self.bot.answer_callback_query(call.id, 'Undo face fix')
            return
        sended = False
        if call.data == 'post_to_channel' or call.data == 'post_to_all':
            try:
                self.bot.copy_message(self.cfg['telegram_chat_id'], call.message.chat.id, call.message.message_id)
                if not call.data == 'post_to_all':
                    self.bot.answer_callback_query(call.id, 'Posted to Telegram channel')
            except Exception as e:
                self.logger.error(e)
                self.bot.answer_callback_query(call.id, 'Error posting to channel')
            else:
                sended = True
        if self.twitter_api_v1 and (call.data == 'post_to_twitter' or call.data == 'post_to_all'):
            sended = self.twitter_send(image_path, call.message.caption)
            if sended:
                self.bot.answer_callback_query(call.id, 'Posted to Twitter')
            else:
                self.bot.answer_callback_query(call.id, 'Error posting to Twitter')
        if sended and call.data == 'post_to_all':
            self.bot.edit_message_reply_markup(call.message.chat.id, call.message.message_id, reply_markup=None)

    def prompt_worker(self, chat_id, sleep_time=600):
        """Prompt worker"""
        while True:
            if not self.cfg['command_only_mode']:
                prmpt = self.prompt.generate(random_prompt_probability=self.cfg['random_prompt_probability'])
                if prmpt:
                    self.worker_queue.put(Job(prmpt, chat_id))
                else:
                    self.logger.warning('Prompt generation failed')
            time.sleep(sleep_time)

    def main_loop(self):
        """Main loop for image generation"""
        while True:
            job = self.worker_queue.get()
            is_admin_chat = int(job.target_chat) in self.cfg['telegram_admin_ids']
            is_turbo_mode = job.target_chat == self.cfg['telegram_turbo_chat_id']
            self.logger.info('Generating image for prompt: {} (seed={} scale={} steps={})'.format(job.prompt, job.seed, job.scale, job.steps))
            if not job.image:
                try:
                    job.image = self.pipe.generate(job.prompt,
                                            seed=job.seed,
                                            scale=job.scale,
                                            steps=job.steps,
                                            width=self.cfg['image_width'],
                                            height=self.cfg['image_height'],
                                            )
                except IndexError as e:
                    self.logger.error(e)
                    job.steps += 1
                    self.worker_queue.put(job)
                    continue
                except RuntimeError as e:
                    self.logger.error(e)
                    torch.cuda.empty_cache()
                    torch.clear_autocast_cache()
                    self.worker_queue.put(job)
                    continue
                except Exception as e:
                    self.logger.error(e)
                    self.worker_queue.put(job)
                    continue
            else:
                job.seed = job.image.info['seed']
                job.scale = job.image.info['scale']
                job.steps = job.image.info['steps']
            if not job.message_id:
                if self.cfg['upscaling']:
                    self.logger.info('Upscaling...')
                    try:
                        job.image = self.enhancement.upscale(job.image)
                    except Exception as e:
                        self.logger.error(e)
                if is_admin_chat or is_turbo_mode:
                    markup = telebot.types.InlineKeyboardMarkup()
                    buttons = [
                        telebot.types.InlineKeyboardButton("Fix face", callback_data="fix_face"),
                        telebot.types.InlineKeyboardButton("Tg", callback_data="post_to_channel"),
                        ]
                    if self.twitter_api_v1:
                        buttons.append(telebot.types.InlineKeyboardButton("Twtr", callback_data="post_to_twitter"))
                    if len(buttons) > 1:
                        buttons.append(telebot.types.InlineKeyboardButton("ALL", callback_data="post_to_all"))
                    markup.add(*buttons, row_width=len(buttons))
                else:
                    markup = None
                self.logger.info('Send image to Telegram...')
                message = '<code>{}</code>\nseed: <code>{}</code> | scale: <code>{}</code> | steps: <code>{}</code>'.format(job.prompt, job.seed, job.scale, job.steps)
                try:
                    resp = self.bot.send_photo(job.target_chat, photo=job.image, caption=message, reply_markup=markup)
                except Exception as e:
                    self.logger.error(e)
                else:
                    if resp.id:
                        self.logger.info("https://t.me/{}/{}".format(resp.chat.username, resp.message_id))
                        job.message_id = resp.message_id
                        image_path = os.path.join(self.cfg['image_cache_dir'], '{}.jpg'.format(job.message_id))
                        if not os.path.exists(image_path):
                            job.image.save(image_path)
                    else:
                        self.logger.error(resp)
            if job.message_id and not is_admin_chat and not is_turbo_mode:
                image_path = os.path.join(self.cfg['image_cache_dir'], '{}.jpg'.format(job.message_id))
                if self.twitter_api_v1:
                    if not self.twitter_send(image_path, job.prompt):
                        self.logger.error('Error posting to Twitter')
            if not is_admin_chat and not job.message_id:
                self.worker_queue.put(job)
            else:
                if job.delete_message:
                    self.bot.delete_message(job.target_chat, job.delete_message)

    def run(self):
        """Start bot"""
        self.logger.info('Starting bot...')
        if hasattr(self.pipe, 'device'):
            self.logger.info('Used device: {}'.format(self.pipe.device))
        self.clean_cache()
        user = self.bot.get_me()
        if len(sys.argv) > 1:
            self.worker_queue.put(Job(sys.argv[1], self.cfg['telegram_chat_id']))
        if not self.cfg['premoderation']:
            threading.Thread(target=self.prompt_worker, args=(self.cfg['telegram_chat_id'], self.cfg['sleep_time']), daemon=True).start()
        if self.cfg['telegram_turbo_chat_id']:
            threading.Thread(target=self.prompt_worker, args=(self.cfg['telegram_turbo_chat_id'], self.cfg['turbo_sleep_time']), daemon=True).start()
        if len(self.cfg['telegram_admin_ids']) > 0:
            threading.Thread(target=self.main_loop, daemon=True).start()
            self.logger.info('Starting bot with username: {}'.format(user.username))
            if self.cfg['command_only_mode']:
                self.logger.info('Command only mode enabled')
            self.bot.infinity_polling()
        else:
            if self.cfg['command_only_mode']:
                self.logger.error('Command only mode is enabled, but no admin ID is provided')
                sys.exit(1)
            self.main_loop()
        self.logger.info('Bot stopped')

if __name__ == '__main__':
    bot = Bot()
    bot.run()