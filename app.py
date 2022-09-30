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
import random
import sys
import telebot
import threading
import time
import torch


@dataclasses.dataclass
class Job:
    prompt: str
    target_chat: str
    count: int = 1
    seed: int = 0
    scale: float = 0
    steps: int = 0
    images: list = dataclasses.field(default_factory=list)
    send_to_instagram: bool = False
    send_to_telegram: bool = False
    delete_message: int = 0

    def __post_init__(self):
        params = {}
        self.prompt = re.sub(r'(\w+)=(\d+\.\d+)', lambda m: params.update({m.group(1): float(m.group(2))}) or '', self.prompt)
        self.prompt = re.sub(r'(\w+)=(\d+)', lambda m: params.update({m.group(1): int(m.group(2))}) or '', self.prompt)
        self.prompt = self.prompt.strip()
        if not self.prompt or self.prompt.endswith('+'):
            self.prompt = prompt.generate(self.prompt.removesuffix('+'))
        self.count = params.get('count', self.count)
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

worker_queue = queue.Queue()

if not os.path.exists(cfg.image_cache_dir):
    os.mkdir(cfg.image_cache_dir)


def clean_cache(age=86400, interval=3600):
    for f in os.listdir(cfg.image_cache_dir):
        if os.stat(os.path.join(cfg.image_cache_dir,f)).st_mtime < time.now - age:
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
    if not insta_logged:
        return False
    bot_logger.info('Send image to Instagram...')
    try:
        resp = insta.photo_upload(image_path, caption=message)
    except Exception as e:
        bot_logger.error(e)
    else:
        if resp.code:
            bot_logger.info("https://www.instagram.com/p/{}/".format(resp.code))
            return True
        else:
            bot_logger.error(resp)
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
        cfg.telegram_chat_id = chat_id
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


@bot.message_handler(chat_id=cfg.telegram_admin_ids, commands=['random'])
def random_generate(message):
    job = Job('', message.chat.id)
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


@bot.callback_query_handler(func=lambda call: True)
def callback_query(call):
    sended = False
    if call.data == 'post_to_channel' or call.data == 'post_to_both':
        try:
            bot.copy_message(cfg.telegram_chat_id, call.message.chat.id, call.message.message_id)
            bot.answer_callback_query(call.id, 'Posted to channel')
        except Exception as e:
            bot_logger.error(e)
            bot.answer_callback_query(call.id, 'Error posting to channel')
        else:
            sended = True
    if call.data == 'post_to_instagram' or call.data == 'post_to_both':
        image_path = os.path.join(cfg.image_cache_dir, '{}.jpg'.format(call.message.message_id))
        sended = instagram_send(image_path, call.message.caption + '\n#aiart #stablediffusion')
        if sended:
            bot.answer_callback_query(call.id, 'Posted to Instagram')
        else:
            bot.answer_callback_query(call.id, 'Error posting to Instagram')
    if sended and call.data == 'post_to_both':
        bot.edit_message_reply_markup(call.message.chat.id, call.message.message_id, reply_markup=None)


def prompt_worker():
    while True:
        if not cfg.command_only_mode and worker_queue.empty():
             worker_queue.put(Job(prompt.generate(), cfg.telegram_chat_id))
        time.sleep(cfg.sleep_time)


def main_loop():
    while True:
        job = worker_queue.get()
        is_admin_chat = int(job.target_chat) in cfg.telegram_admin_ids
        bot_logger.info('Generating (count={}, seed={}, scale={}, steps={}) image for prompt: {}'.format(job.count, job.seed, job.scale, job.steps, job.prompt))
        if len(job.images) == 0:
            try:
                job.images = diffusion.generate(job.prompt, count=job.count, seed=job.seed, steps=job.steps)
            except IndexError as e:
                bot_logger.error(e)
                job.steps += 1
                worker_queue.put(job)
                continue
            except RuntimeError as e:
                bot_logger.error(e)
                torch.cuda.empty_cache()
                if job.count > 1:
                    job.count -= 1
                worker_queue.put(job)
                continue
            except Exception as e:
                bot_logger.error(e)
                worker_queue.put(job)
                continue
        for image in job.images:
            if enhancement.upscaling:
                bot_logger.info('Upscaling...')
                try:
                    face_restore = enhancement.face_presence_detection(image)
                    if face_restore:
                        bot_logger.info('Faces detected, restoring...')
                    image = enhancement.upscale(image, face_restore=face_restore)
                except Exception as e:
                    bot_logger.error(e)
            image_path = None
            if not job.send_to_telegram:
                markup = None
                if is_admin_chat:
                    markup = telebot.types.InlineKeyboardMarkup()
                    markup.add(telebot.types.InlineKeyboardButton("Post to channel", callback_data="post_to_channel"),
                               telebot.types.InlineKeyboardButton("Post to Instagram", callback_data="post_to_instagram"),
                               telebot.types.InlineKeyboardButton("Post to both", callback_data="post_to_both"),
                               row_width=2)
                bot_logger.info('Send image to Telegram...')
                message = '<code>{}</code>\nseed: <code>{}</code> | scale: <code>{}</code> | steps: <code>{}</code>'.format(job.prompt, job.seed, job.scale, job.steps)
                try:
                    resp = bot.send_photo(job.target_chat, photo=image, caption=message, reply_markup=markup)
                except Exception as e:
                    bot_logger.error(e)
                else:
                    if resp.id:
                        bot_logger.info("https://t.me/{}/{}".format(resp.chat.username, resp.message_id))
                        job.send_to_telegram = True
                        image_path = os.path.join(cfg.image_cache_dir, '{}.jpg'.format(resp.message_id))
                        if not os.path.exists(image_path):
                            image.save(image_path)
                    else:
                        bot_logger.error(resp)
            if insta_logged and image_path and not job.send_to_instagram and not is_admin_chat:
                message = '{}\nseed: {} | scale: {} | steps: {}\n#aiart #stablediffusion'.format(job.prompt, job.seed, job.scale, job.steps)
                job.send_to_instagram = instagram_send(image_path, message)
            if not is_admin_chat and (not job.send_to_telegram or (insta_logged and not job.send_to_instagram)):
                worker_queue.put(job)
            else:
                if job.delete_message:
                    bot.delete_message(job.target_chat, job.delete_message)


if __name__ == '__main__':
    bot_logger.info('Used device: {}'.format(diffusion.device))
    clean_cache()
    user = bot.get_me()
    if cfg.instagram_username and cfg.instagram_password:
        threading.Thread(target=instagram_login).start()
    if len(sys.argv) > 1:
        worker_queue.put(Job(sys.argv[1], cfg.telegram_chat_id))
    threading.Thread(target=prompt_worker, daemon=True).start()
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
        ])
        bot.infinity_polling()
        bot.delete_my_commands()
    else:
        if cfg.command_only_mode:
            bot_logger.error('Command only mode is enabled, but no admin ID is provided')
            sys.exit(1)
        main_loop()
    bot_logger.info('Bot stopped')
