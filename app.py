import diffusion
import enhancement
import logging
import os
import prompt
import queue
import sys
import telebot
import threading
import time


class Config:
    def __init__(self, env_file='.env'):
        self.env_file = env_file
        self._load()

    def _load(self):
        if os.path.exists(self.env_file):
            logging.info(f'Loading environment variables from {self.env_file}')
            with open(self.env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        key, value = line.split('=', 1)
                        os.putenv(key, value)
        self.command_only_mode = os.getenv('COMMAND_ONLY_MODE', 'false').lower() in ['true', 'on', 'yes', '1']
        self.sleep_time = float(os.getenv('SLEEP_TIME', 60))
        self.telegram_token = os.getenv('TELEGRAM_TOKEN')
        telegram_admin_id = os.getenv('TELEGRAM_ADMIN_ID')
        if telegram_admin_id:
            self.telegram_admin_ids = [int(i) for i in telegram_admin_id.split(',')]
        else:
            self.telegram_admin_ids = []
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')


logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
cfg = Config()

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
        worker_queue.put(prompt)
        bot.send_message(message.chat.id, 'Put prompt <code>{}</code> in queue: {}'.format(prompt, worker_queue.qsize()))


@bot.message_handler(chat_id=cfg.telegram_admin_ids, content_types=['document'], func=lambda m: m.document.file_name == 'prompt.json')
def handle_prompts_update(message):
    file_info = bot.get_file(message.document.file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    with open('prompt.json', 'wb') as f:
        f.write(downloaded_file)
    bot.send_message(message.chat.id, 'Prompts updated')


def main_loop():
    while True:
        if not cfg.command_only_mode and worker_queue.empty():
            random_prompt = prompt.get_prompt()
        else:
            random_prompt = worker_queue.get()
        if random_prompt.endswith('+'):
            random_prompt = prompt.get_prompt(random_prompt.removesuffix('+'))
        logging.info('Generating image for prompt: {}'.format(random_prompt))
        try:
            images = diffusion.generate(random_prompt)
            for image in images:
                if enhancement.upscaling:
                    logging.info('Upscaling...')
                    face_restore = enhancement.face_presence_detection(image)
                    if face_restore:
                        logging.info('Faces detected, restoring...')
                    image = enhancement.upscale(image, face_restore=face_restore)
                logging.info('Send image to Telegram...')
                resp = bot.send_photo(cfg.telegram_chat_id, photo=image, caption=random_prompt)
                if resp.id:
                    logging.info("https://t.me/{}/{}".format(resp.chat.username, resp.message_id))
                else:
                    logging.error(resp)
        except Exception as e:
            logging.error(e)
            worker_queue.put(random_prompt)
        if not cfg.command_only_mode:
            time.sleep(cfg.sleep_time)


if __name__ == '__main__':
    logging.info('Used device: {}'.format(diffusion.device))
    user = bot.get_me()
    if len(sys.argv) > 1:
        worker_queue.put(sys.argv[1])
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
