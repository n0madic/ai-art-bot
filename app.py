import config
import dataclasses
import diffusion
import enhancement
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

    def __post_init__(self):
        params = {}
        self.prompt = re.sub(r'(\w+)=(\d+\.\d+)', lambda m: params.update({m.group(1): float(m.group(2))}) or '', self.prompt)
        self.prompt = re.sub(r'(\w+)=(\d+)', lambda m: params.update({m.group(1): int(m.group(2))}) or '', self.prompt)
        self.prompt = self.prompt.strip()
        if self.prompt.endswith('+'):
            self.prompt = prompt.generate(self.prompt.removesuffix('+'))
        self.count = params.get('count', self.count)
        self.seed = params.get('seed', self.seed or random.randint(0, 2**32 - 1))
        self.scale = params.get('scale', self.scale or round(random.uniform(7,10), 1))
        self.steps = params.get('steps', self.steps or random.randint(30,100))


logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
cfg = config.cfg

bot = telebot.TeleBot(cfg.telegram_token, parse_mode='HTML')
bot.add_custom_filter(telebot.custom_filters.ChatFilter())
worker_queue = queue.Queue()


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
        bot.send_message(message.chat.id, 'Put prompt <code>{}</code> in queue: {}'.format(prompt, worker_queue.qsize()))


@bot.message_handler(chat_id=cfg.telegram_admin_ids, content_types=['document'], func=lambda m: m.document.file_name == 'ideas.txt')
def handle_ideas_update(message):
    file_info = bot.get_file(message.document.file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    with open('ideas.txt', 'wb') as f:
        f.write(downloaded_file)
    bot.send_message(message.chat.id, 'Ideas updated')
    bot.delete_message(message.chat.id, message.message_id)


def prompt_worker():
    while True:
        if not cfg.command_only_mode and worker_queue.empty():
             worker_queue.put(Job(prompt.generate(), cfg.telegram_chat_id))
        time.sleep(cfg.sleep_time)


def main_loop():
    while True:
        job = worker_queue.get()
        logging.info('Generating (count={}, seed={}, scale={}, steps={}) image for prompt: {}'.format(job.count, job.seed, job.scale, job.steps, job.prompt))
        try:
            images = diffusion.generate(job.prompt, count=job.count, seed=job.seed, steps=job.steps)
            for image in images:
                if enhancement.upscaling:
                    logging.info('Upscaling...')
                    face_restore = enhancement.face_presence_detection(image)
                    if face_restore:
                        logging.info('Faces detected, restoring...')
                    image = enhancement.upscale(image, face_restore=face_restore)
                logging.info('Send image to Telegram...')
                message = '<code>{}</code>\nseed: <code>{}</code> | scale: <code>{}</code> | steps: <code>{}</code>'.format(job.prompt, job.seed, job.scale, job.steps)
                resp = bot.send_photo(job.target_chat, photo=image, caption=message)
                if resp.id:
                    logging.info("https://t.me/{}/{}".format(resp.chat.username, resp.message_id))
                else:
                    logging.error(resp)
        except IndexError as e:
            logging.error(e)
            job.steps += 1
            worker_queue.put(job)
        except RuntimeError as e:
            logging.error(e)
            torch.cuda.empty_cache()
            if job.count > 1:
                job.count -= 1
            worker_queue.put(job)
        except Exception as e:
            logging.error(e)
            worker_queue.put(job)


if __name__ == '__main__':
    logging.info('Used device: {}'.format(diffusion.device))
    user = bot.get_me()
    if len(sys.argv) > 1:
        worker_queue.put(Job(sys.argv[1], cfg.telegram_chat_id))
    threading.Thread(target=prompt_worker, daemon=True).start()
    if len(cfg.telegram_admin_ids) > 0:
        threading.Thread(target=main_loop, daemon=True).start()
        logging.info('Starting bot with username: {}'.format(user.username))
        if cfg.command_only_mode:
            logging.info('Command only mode enabled')
        bot.set_my_commands([
            telebot.types.BotCommand('start', 'Start the bot'),
            telebot.types.BotCommand('help', 'Show help message'),
            telebot.types.BotCommand('chat', 'Change target chat ("2me" for this private chat)'),
            telebot.types.BotCommand('command_mode', 'On/Off command only mode'),
            telebot.types.BotCommand('sleep', 'Change sleep time'),
            telebot.types.BotCommand('reset', 'Reset config'),
        ])
        bot.infinity_polling()
    else:
        if cfg.command_only_mode:
            logging.error('Command only mode is enabled, but no admin ID is provided')
            sys.exit(1)
        main_loop()
    logging.info('Bot stopped')
