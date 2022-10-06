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
import twitter
import webui


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
        self.prompt = re.sub(r'(\w+)[:=]\s?(\d+\.\d+)', lambda m: params.update({m.group(1).lower(): float(m.group(2))}) or '', self.prompt)
        self.prompt = re.sub(r'(\w+)[:=]\s?(\d+)', lambda m: params.update({m.group(1).lower(): int(m.group(2))}) or '', self.prompt)
        self.prompt = re.sub(r'[(|]\s*[)|]','', self.prompt)
        self.prompt = self.prompt.strip()
        if not self.prompt or self.prompt.endswith('+'):
            self.prompt = prompt.generate(self.prompt.removesuffix('+'), random_prompt_probability=cfg.random_prompt_probability)
        self.seed = params.get('seed', self.seed or random.randint(0, 2**32 - 1))
        self.scale = params.get('scale', self.scale or round(random.uniform(7,10), 1))
        self.steps = params.get('steps', self.steps or random.randint(30,100))


logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
bot_logger = logging.getLogger('bot')
bot_logger.setLevel(logging.INFO)
insta_logger = logging.getLogger('instagrapi')
insta_logger.setLevel(logging.ERROR)

cfg = config.cfg

bot = telebot.TeleBot(cfg.telegram_token, parse_mode='HTML')
bot.add_custom_filter(telebot.custom_filters.ChatFilter())

insta = instagrapi.Client(logger=insta_logger)
insta_logged = False

if cfg.twitter_consumer_key and cfg.twitter_consumer_secret and cfg.twitter_access_token and cfg.twitter_access_token_secret:
    try:
        twitter_api = twitter.Api(consumer_key=cfg.twitter_consumer_key,
                                  consumer_secret=cfg.twitter_consumer_secret,
                                  access_token_key=cfg.twitter_access_token,
                                  access_token_secret=cfg.twitter_access_token_secret,
                                  sleep_on_rate_limit=True)
        tw_creds = twitter_api.VerifyCredentials()
    except Exception as e:
        bot_logger.error("Twitter authentication: {}".format(e))
        twitter_api = None
    else:
        bot_logger.info('Logged in Twitter as {}'.format(tw_creds.screen_name))
else:
    twitter_api = None

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
    except instagrapi.exceptions.ChallengeRequired as e:
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
        resp = twitter_api.PostUpdate(status, media=image_path)
    except Exception as e:
        bot_logger.error(e)
    else:
        bot_logger.info("https://twitter.com/{}/status/{}".format(resp.user.screen_name, resp.id))
        return True
    return False


@bot.message_handler(chat_id=cfg.telegram_admin_ids, commands=['start', 'help'])
def start(message):
    bot.send_message(message.chat.id, 'Just type the text prompt for image generation\n\nUse the <code>+</code> symbol at the end of the query to expand it with random data, for example:\n<code>cat+</code>')


@bot.message_handler(chat_id=cfg.telegram_admin_ids, commands=['chat'])
def change_chat(message):
    chat_id = message.text.split()[1]
    if chat_id == '2me':
        cfg.telegram_chat_id = message.chat.id
        bot.send_message(message.chat.id, 'Target chat changed on this private chat')
        return
    if not chat_id.isdigit() and not chat_id.startswith('@'):
        chat_id = '@' + chat_id
    try:
        resp = bot.get_chat(chat_id)
    except telebot.apihelper.ApiException as e:
        bot.send_message(message.chat.id, e)
    else:
        cfg.telegram_chat_id = resp.id
        bot.send_message(message.chat.id, 'Target chat changed on {}'.format(resp.title))


@bot.message_handler(chat_id=cfg.telegram_admin_ids, commands=['command_mode'])
def change_command_mode(message):
    cfg.command_only_mode = message.text.split()[1].lower() == 'on'
    if cfg.command_only_mode:
        bot.send_message(message.chat.id, 'Command only mode enabled')
    else:
        bot.send_message(message.chat.id, 'Command only mode disabled')


@bot.message_handler(chat_id=cfg.telegram_admin_ids, commands=['sleep'])
def change_sleep(message):
    sleep_text = message.text.split()[1].strip()
    if sleep_text == 'reset':
        cfg.sleep_time = float(os.getenv('SLEEP_TIME', 60))
        bot.send_message(message.chat.id, 'Sleep time changed on default value')
        return
    cfg.sleep_time = float(sleep_text)
    bot.send_message(message.chat.id, 'Sleep time changed on {:0.0f} sec'.format(cfg.sleep_time))


@bot.message_handler(chat_id=cfg.telegram_admin_ids, commands=['reset'])
def change_sleep(message):
    cfg._load()
    bot.send_message(message.chat.id, 'Config reseted')


@bot.message_handler(chat_id=cfg.telegram_admin_ids, commands=['webui'])
def change_webui_mode(message):
    cfg.webui = message.text.split()[1].lower() == 'on'
    if cfg.webui:
        _, _, share_url = webui.gr.launch(share=True, prevent_thread_lock=True)
        bot.send_message(message.chat.id, 'WebUI enabled on {}'.format(share_url))
    else:
        webui.gr.close(verbose=True)
        bot.send_message(message.chat.id, 'WebUI disabled')


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
        loop = re.findall(r'loop=(\d+)', prompt)
        if loop:
            loop = int(loop[0])
            prompt = re.sub(r'loop=(\d+)', '', prompt).strip()
        else:
            loop = 1
        for i in range(loop):
            job = Job(prompt, message.chat.id)
            job.seed += i
            worker_queue.put(job)
            if prompt != job.prompt or i == loop - 1:
                msg = bot.send_message(message.chat.id, 'Put prompt <code>{}</code> in queue: {}'.format(job.prompt, worker_queue.qsize()), disable_notification=True)
                job.delete_message = msg.message_id


@bot.message_handler(chat_id=cfg.telegram_admin_ids, content_types=['document'], func=lambda m: m.document.file_name == 'ideas.txt')
def handle_ideas_update(message):
    file_info = bot.get_file(message.document.file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    with open('ideas.txt', 'wb') as f:
        f.write(downloaded_file)
    bot.send_message(message.chat.id, 'Ideas updated')
    bot.delete_message(message.chat.id, message.message_id)


@bot.callback_query_handler(func=lambda call: call.from_user.id in cfg.telegram_admin_ids)
def callback_query(call):
    sended = False
    image_path = os.path.join(cfg.image_cache_dir, '{}.jpg'.format(call.message.message_id))
    if call.data == 'post_to_channel' or call.data == 'post_to_all':
        try:
            bot.copy_message(cfg.telegram_chat_id, call.message.chat.id, call.message.message_id)
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
    if twitter_api and (call.data == 'post_to_twitter' or call.data == 'post_to_all'):
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
            worker_queue.put(Job(prompt.generate(), chat_id))
        time.sleep(sleep_time)


def main_loop():
    while True:
        job = worker_queue.get()
        is_admin_chat = int(job.target_chat) in cfg.telegram_admin_ids
        is_turbo_mode = job.target_chat == cfg.telegram_turbo_chat_id
        bot_logger.info('Generating (seed={}, scale={}, steps={}) image for prompt: {}'.format(job.seed, job.scale, job.steps, job.prompt))
        if not job.image:
            try:
                job.image = diffusion.generate(job.prompt, seed=job.seed, scale=job.scale, steps=job.steps)
            except IndexError as e:
                bot_logger.error(e)
                job.steps += 1
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
                    face_restore = enhancement.face_presence_detection(job.image)
                    if face_restore:
                        bot_logger.info('Faces detected, restoring...')
                    job.image = enhancement.upscale(job.image, face_restore=face_restore)
                except Exception as e:
                    bot_logger.error(e)
            if is_admin_chat or is_turbo_mode:
                markup = telebot.types.InlineKeyboardMarkup()
                buttons = [telebot.types.InlineKeyboardButton("Post to channel", callback_data="post_to_channel")]
                if insta_logged:
                    buttons.append(telebot.types.InlineKeyboardButton("Post to Instagram", callback_data="post_to_instagram"))
                if twitter_api:
                    buttons.append(telebot.types.InlineKeyboardButton("Post to Twitter", callback_data="post_to_twitter"))
                if len(buttons) > 1:
                    buttons.append(telebot.types.InlineKeyboardButton("Post to all", callback_data="post_to_all"))
                markup.add(*buttons)
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
            if twitter_api:
                if not twitter_send(image_path, job.prompt):
                    bot_logger.error('Error posting to Twitter')
        if not is_admin_chat and not job.message_id:
            worker_queue.put(job)
        else:
            if job.delete_message:
                bot.delete_message(job.target_chat, job.delete_message)


if __name__ == '__main__':
    bot_logger.info('Used device: {}'.format(diffusion.device))
    clean_cache()
    user = bot.get_me()
    if cfg.webui:
        webui.gr.launch(share=True, prevent_thread_lock=True)
    if cfg.instagram_username and cfg.instagram_password:
        threading.Thread(target=instagram_login).start()
    if len(sys.argv) > 1:
        worker_queue.put(Job(sys.argv[1], cfg.telegram_chat_id))
    threading.Thread(target=prompt_worker, args=(cfg.telegram_chat_id, cfg.sleep_time), daemon=True).start()
    if cfg.telegram_turbo_chat_id:
        threading.Thread(target=prompt_worker, args=(cfg.telegram_turbo_chat_id, cfg.turbo_sleep_time), daemon=True).start()
    if len(cfg.telegram_admin_ids) > 0:
        threading.Thread(target=main_loop, daemon=True).start()
        bot_logger.info('Starting bot with username: {}'.format(user.username))
        if cfg.command_only_mode:
            bot_logger.info('Command only mode enabled')
        bot.set_my_commands([
            telebot.types.BotCommand('start', 'Start the bot'),
            telebot.types.BotCommand('help', 'Show help message'),
            telebot.types.BotCommand('chat', 'Change target chat ("2me" for this private chat)'),
            telebot.types.BotCommand('command_mode', 'On/Off command only mode'),
            telebot.types.BotCommand('sleep', 'Change sleep time'),
            telebot.types.BotCommand('reset', 'Reset config'),
            telebot.types.BotCommand('random', 'Generate random prompt'),
            telebot.types.BotCommand('webui', 'WebUI on/off'),
        ])
        bot.infinity_polling()
        bot.delete_my_commands()
    else:
        if cfg.command_only_mode:
            bot_logger.error('Command only mode is enabled, but no admin ID is provided')
            sys.exit(1)
        main_loop()
    bot_logger.info('Bot stopped')
