import config
import dataclasses
import diffusion
import enhancement
import instagrapi
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
        if not self.prompt or self.prompt.endswith('+'):
            self.prompt = prompt.generate(self.prompt.removesuffix('+'), random_prompt_probability=cfg.random_prompt_probability)
        self.seed = params.get('seed', self.seed or random.randint(0, 2**32 - 1))
        self.scale = params.get('scale', self.scale or round(random.uniform(7,10), 1))
        self.steps = params.get('steps', self.steps or random.randint(20,100))


logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
bot_logger = logging.getLogger('bot')
bot_logger.setLevel(logging.INFO)
insta_logger = logging.getLogger('instagrapi')
insta_logger.setLevel(logging.ERROR)

cfg = config.cfg

pipe = diffusion.Pipeline(cfg.sd_model_id, cfg.sd_model_vae_id, cfg.sd_refiner_id, cfg.fp16, cfg.low_vram)

bot = telebot.TeleBot(cfg.telegram_token, parse_mode='HTML')
bot.add_custom_filter(telebot.custom_filters.ChatFilter())

insta = instagrapi.Client(logger=insta_logger)
insta_logged = False

if cfg.twitter_consumer_key and cfg.twitter_consumer_secret and cfg.twitter_access_token and cfg.twitter_access_token_secret:
    try:
        auth = tweepy.OAuthHandler(cfg.twitter_consumer_key, cfg.twitter_consumer_secret)
        auth.set_access_token(cfg.twitter_access_token, cfg.twitter_access_token_secret)
        twitter_api_v1 = tweepy.API(auth)
        tw_creds = twitter_api_v1.verify_credentials(skip_status=True, include_entities=False)
        twitter_client = tweepy.Client(
            consumer_key=cfg.twitter_consumer_key, consumer_secret=cfg.twitter_consumer_secret,
            access_token=cfg.twitter_access_token, access_token_secret=cfg.twitter_access_token_secret
        )
    except Exception as e:
        bot_logger.error("Twitter authentication: {}".format(e))
        twitter_api_v1 = None
    else:
        bot_logger.info('Logged in Twitter as {}'.format(tw_creds.screen_name))
else:
    twitter_api_v1 = None

worker_queue = queue.Queue()

if not os.path.exists(cfg.image_cache_dir):
    os.mkdir(cfg.image_cache_dir)


def clean_cache(age=86400, interval=3600):
    for f in os.listdir(cfg.image_cache_dir):
        if os.stat(os.path.join(cfg.image_cache_dir,f)).st_mtime < time.time() - age:
            os.remove(os.path.join(cfg.image_cache_dir,f))
    threading.Timer(interval, clean_cache).start()


def instagram_login():
    global insta_logged
    attempt_count = 0
    while not insta_logged:
        try:
            insta_logged = insta.login(cfg.instagram_username, cfg.instagram_password)
        except Exception as e:
            bot_logger.error(e)
            attempt_count += 1
            if attempt_count > 5:
                attempt_count = 0
                time.sleep(3600)
            else:
                time.sleep(random.randint(60, 600))
    bot_logger.info('Logged in Instagram as {}'.format(insta.username))


def instagram_send(image_path, message):
    global insta_logged
    if not insta_logged:
        return False
    bot_logger.info('Send image to Instagram...')
    try:
        resp = insta.photo_upload(image_path, caption=message)
    except (instagrapi.exceptions.ChallengeRequired, instagrapi.exceptions.ClientForbiddenError) as e:
        bot_logger.error(e)
        insta.logout()
        insta_logged = False
        threading.Thread(target=instagram_login).start()
    except Exception as e:
        bot_logger.error(e)
    else:
        if resp.code:
            bot_logger.info("https://www.instagram.com/p/{}/".format(resp.code))
            return True
        else:
            bot_logger.error(resp)
    return False


def twitter_send(image_path, message):
    message = message.splitlines()[0] + '\n#AIart #stablediffusion'
    status = textwrap.shorten(message, width=280, placeholder='...')
    bot_logger.info('Send image to Twitter...')
    try:
        media = twitter_api_v1.media_upload(image_path)
        resp = twitter_client.create_tweet(text=status, media_ids=[media.media_id])
    except Exception as e:
        bot_logger.error(e)
    else:
        bot_logger.info("https://twitter.com/{}/status/{}".format(tw_creds.screen_name, resp.data['id']))
        return True
    return False


@bot.message_handler(chat_id=cfg.telegram_admin_ids, commands=['start', 'help'])
def start(message):
    bot.send_message(message.chat.id, 'Just type the text prompt for image generation\n\nUse the <code>+</code> symbol at the end of the query to expand it with random data, for example:\n<code>cat+</code>')


@bot.message_handler(chat_id=cfg.telegram_admin_ids, commands=['die'])
def die(message):
    bot.send_message(message.chat.id, 'Bye!')
    os._exit(0)


@bot.message_handler(chat_id=cfg.telegram_admin_ids, commands=['config'])
def change_config(message):
    parameter = message.text.split()[1].lower()
    value = message.text.split()[2]
    old_value = getattr(cfg, parameter, None)
    if old_value:
        if type(old_value) == bool:
            value = True if value.lower() in ['true', 'on', 'yes', '1'] else False
        elif type(old_value) == int:
            value = int(value)
        elif type(old_value) == float:
            value = float(value)
        setattr(cfg, parameter, value)
        value = getattr(cfg, parameter)
        bot.send_message(message.chat.id, 'Parameter {} changed to {}'.format(parameter, value))
    else:
        bot.send_message(message.chat.id, 'Parameter {} not found'.format(parameter))


@bot.message_handler(chat_id=cfg.telegram_admin_ids, commands=['reset'])
def change_sleep(message):
    cfg._load()
    bot.send_message(message.chat.id, 'Config reseted')


@bot.message_handler(chat_id=cfg.telegram_admin_ids, commands=['random'])
def random_generate(message):
    job = Job(prompt.generate(random_prompt_probability=1), message.chat.id)
    msg = bot.send_message(message.chat.id, 'Put random prompt <code>{}</code> in queue: {}'.format(job.prompt, worker_queue.qsize()), disable_notification=True)
    job.delete_message = msg.message_id
    worker_queue.put(job)


@bot.message_handler(chat_id=cfg.telegram_admin_ids)
def command_generate(message):
    prompt = message.text.strip()
    if prompt == "":
        bot.send_message(message.chat.id, 'Please provide a prompt')
    else:
        batch = re.findall(r'batch=(\d+)', prompt)
        if batch:
            batch = int(batch[0])
            prompt = re.sub(r'batch=(\d+)', '', prompt).strip()
        else:
            batch = 1
        for i in range(batch):
            job = Job(prompt, message.chat.id)
            job.seed += i
            worker_queue.put(job)
            if prompt != job.prompt or i == batch - 1:
                time.sleep(0.1)
                msg = bot.send_message(message.chat.id, 'Put prompt <code>{}</code> in queue: {}'.format(job.prompt, worker_queue.qsize()), disable_notification=True)
                job.delete_message = msg.message_id


@bot.message_handler(chat_id=cfg.telegram_admin_ids, content_types=['document'], func=lambda m: m.document.file_name.endswith('.txt'))
def handle_ideas_update(message):
    if not os.path.exists(message.document.file_name):
        bot.send_message(message.chat.id, 'File {} not found!'.format(message.document.file_name))
    else:
        file_info = bot.get_file(message.document.file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        with open(message.document.file_name, 'wb') as f:
            f.write(downloaded_file)
        bot.send_message(message.chat.id, 'File {} updated'.format(message.document.file_name))
    bot.delete_message(message.chat.id, message.message_id)


@bot.callback_query_handler(func=lambda call: call.from_user.id in cfg.telegram_admin_ids)
def callback_query(call):
    image_path = os.path.join(cfg.image_cache_dir, '{}.jpg'.format(call.message.message_id))
    if not os.path.exists(image_path) and not call.data == 'post_to_channel':
        try:
            file_info = bot.get_file(call.message.photo[-1].file_id)
            downloaded_file = bot.download_file(file_info.file_path)
        except Exception as e:
            bot_logger.error(e)
            bot.answer_callback_query(call.id, 'Error downloading image')
            return
        else:
            with open(image_path, 'wb') as f:
                f.write(downloaded_file)
    if call.data == 'fix_face' or call.data == 'undo_face':
        if not os.path.exists(image_path):
            bot.answer_callback_query(call.id, 'Image not found')
            return
        reply_markup = call.message.reply_markup
        if call.data == 'fix_face':
            image = enhancement.fixface(image_path)
            os.replace(image_path, image_path + '.bak')
            image.save(image_path)
            reply_markup.keyboard[0][0] = telebot.types.InlineKeyboardButton("Undo fix", callback_data="undo_face")
        else:
            os.replace(image_path + '.bak', image_path)
            reply_markup.keyboard[0][0] = telebot.types.InlineKeyboardButton("Face fix", callback_data="fix_face")
        lines = call.message.caption.splitlines()
        caption = '\n'.join(['<code>{}</code>'.format(lines[0]), re.sub(r'\d+\.?\d+', r'<code>\g<0></code>', lines[1])])
        bot.edit_message_media(media=telebot.types.InputMediaPhoto(open(image_path, 'rb'), caption=caption, parse_mode='HTML'), chat_id=call.message.chat.id, message_id=call.message.message_id, reply_markup=reply_markup)
        if call.data == 'fix_face':
            bot.answer_callback_query(call.id, 'Face fixed')
        else:
            bot.answer_callback_query(call.id, 'Undo face fix')
        return
    sended = False
    if call.data == 'post_to_channel' or call.data == 'post_to_all':
        try:
            bot.copy_message(cfg.telegram_chat_id, call.message.chat.id, call.message.message_id)
            if not call.data == 'post_to_all':
                bot.answer_callback_query(call.id, 'Posted to Telegram channel')
        except Exception as e:
            bot_logger.error(e)
            bot.answer_callback_query(call.id, 'Error posting to channel')
        else:
            sended = True
    if insta_logged and (call.data == 'post_to_instagram' or call.data == 'post_to_all'):
        sended = instagram_send(image_path, call.message.caption + '\n#aiart #stablediffusion')
        if sended:
            bot.answer_callback_query(call.id, 'Posted to Instagram')
        else:
            bot.answer_callback_query(call.id, 'Error posting to Instagram')
    if twitter_api_v1 and (call.data == 'post_to_twitter' or call.data == 'post_to_all'):
        sended = twitter_send(image_path, call.message.caption)
        if sended:
            bot.answer_callback_query(call.id, 'Posted to Twitter')
        else:
            bot.answer_callback_query(call.id, 'Error posting to Twitter')
    if sended and call.data == 'post_to_all':
        bot.edit_message_reply_markup(call.message.chat.id, call.message.message_id, reply_markup=None)


def prompt_worker(chat_id, sleep_time=600):
    while True:
        if not cfg.command_only_mode:
            prmpt = prompt.generate()
            if prmpt:
                worker_queue.put(Job(prmpt, chat_id))
            else:
                bot_logger.warning('Prompt generation failed')
        time.sleep(sleep_time)


def main_loop():
    while True:
        job = worker_queue.get()
        is_admin_chat = int(job.target_chat) in cfg.telegram_admin_ids
        is_turbo_mode = job.target_chat == cfg.telegram_turbo_chat_id
        bot_logger.info('Generating image for prompt: {} (seed={} scale={} steps={})'.format(job.prompt, job.seed, job.scale, job.steps))
        if not job.image:
            try:
                job.image = pipe.generate(job.prompt, seed=job.seed, scale=job.scale, steps=job.steps)
            except IndexError as e:
                bot_logger.error(e)
                job.steps += 1
                worker_queue.put(job)
                continue
            except RuntimeError as e:
                bot_logger.error(e)
                torch.cuda.empty_cache()
                torch.clear_autocast_cache()
                worker_queue.put(job)
                continue
            except Exception as e:
                bot_logger.error(e)
                worker_queue.put(job)
                continue
        else:
            job.seed = job.image.info['seed']
            job.scale = job.image.info['scale']
            job.steps = job.image.info['steps']
        if not job.message_id:
            if cfg.upscaling:
                bot_logger.info('Upscaling...')
                try:
                    job.image = enhancement.upscale(job.image)
                except Exception as e:
                    bot_logger.error(e)
            if is_admin_chat or is_turbo_mode:
                markup = telebot.types.InlineKeyboardMarkup()
                buttons = [
                    telebot.types.InlineKeyboardButton("Fix face", callback_data="fix_face"),
                    telebot.types.InlineKeyboardButton("Tg", callback_data="post_to_channel"),
                    ]
                if insta_logged:
                    buttons.append(telebot.types.InlineKeyboardButton("Insta", callback_data="post_to_instagram"))
                if twitter_api_v1:
                    buttons.append(telebot.types.InlineKeyboardButton("Twtr", callback_data="post_to_twitter"))
                if len(buttons) > 1:
                    buttons.append(telebot.types.InlineKeyboardButton("ALL", callback_data="post_to_all"))
                markup.add(*buttons, row_width=len(buttons))
            else:
                markup = None
            bot_logger.info('Send image to Telegram...')
            message = '<code>{}</code>\nseed: <code>{}</code> | scale: <code>{}</code> | steps: <code>{}</code>'.format(job.prompt, job.seed, job.scale, job.steps)
            try:
                resp = bot.send_photo(job.target_chat, photo=job.image, caption=message, reply_markup=markup)
            except Exception as e:
                bot_logger.error(e)
            else:
                if resp.id:
                    bot_logger.info("https://t.me/{}/{}".format(resp.chat.username, resp.message_id))
                    job.message_id = resp.message_id
                    image_path = os.path.join(cfg.image_cache_dir, '{}.jpg'.format(job.message_id))
                    if not os.path.exists(image_path):
                        job.image.save(image_path)
                else:
                    bot_logger.error(resp)
        if job.message_id and not is_admin_chat and not is_turbo_mode:
            image_path = os.path.join(cfg.image_cache_dir, '{}.jpg'.format(job.message_id))
            if insta_logged:
                message = '{}\nseed: {} | scale: {} | steps: {}\n#aiart #stablediffusion'.format(job.prompt, job.seed, job.scale, job.steps)
                if not instagram_send(image_path, message):
                    bot_logger.error('Error posting to Instagram')
            if twitter_api_v1:
                if not twitter_send(image_path, job.prompt):
                    bot_logger.error('Error posting to Twitter')
        if not is_admin_chat and not job.message_id:
            worker_queue.put(job)
        else:
            if job.delete_message:
                bot.delete_message(job.target_chat, job.delete_message)


if __name__ == '__main__':
    bot_logger.info('Used device: {}'.format(pipe.device))
    clean_cache()
    user = bot.get_me()
    if cfg.instagram_username and cfg.instagram_password:
        threading.Thread(target=instagram_login).start()
    if len(sys.argv) > 1:
        worker_queue.put(Job(sys.argv[1], cfg.telegram_chat_id))
    if not cfg.premoderation:
        threading.Thread(target=prompt_worker, args=(cfg.telegram_chat_id, cfg.sleep_time), daemon=True).start()
    if cfg.telegram_turbo_chat_id:
        threading.Thread(target=prompt_worker, args=(cfg.telegram_turbo_chat_id, cfg.turbo_sleep_time), daemon=True).start()
    if len(cfg.telegram_admin_ids) > 0:
        threading.Thread(target=main_loop, daemon=True).start()
        bot_logger.info('Starting bot with username: {}'.format(user.username))
        if cfg.command_only_mode:
            bot_logger.info('Command only mode enabled')
        bot.infinity_polling()
    else:
        if cfg.command_only_mode:
            bot_logger.error('Command only mode is enabled, but no admin ID is provided')
            sys.exit(1)
        main_loop()
    bot_logger.info('Bot stopped')
