import diffusion
import enhancement
import io
import logging
import os
import prompt
import queue
import sys
import telebot
import threading
import time


command_only_mode = os.getenv('COMMAND_ONLY_MODE', 'false').lower() == 'true'
sleep_time = float(os.getenv('SLEEP_TIME', 60))
telegram_token = os.getenv('TELEGRAM_TOKEN')
telegram_admin_id = int(os.getenv('TELEGRAM_ADMIN_ID'))
telegram_chat_id = int(os.getenv('TELEGRAM_CHAT_ID'))
bot = telebot.TeleBot(telegram_token, parse_mode='MarkdownV2')
bot.add_custom_filter(telebot.custom_filters.ChatFilter())
worker_queue = queue.Queue()


@bot.message_handler(chat_id=[telegram_admin_id], commands=['start'])
def start(message):
    bot.send_message(message.chat.id, 'Just type the text prompt for image generation')


@bot.message_handler(chat_id=[telegram_admin_id])
def command_generate(message):
    prompt = message.text.strip()
    if prompt == "":
        bot.send_message(message.chat.id, 'Please provide a prompt')
    else:
        worker_queue.put(prompt)
        bot.send_message(message.chat.id, 'Put prompt `{}` in queue: {}'.format(prompt, worker_queue.qsize()))


def main_loop():
    logging.info('Used device: {}'.format(diffusion.device))
    while True:
        if not command_only_mode and worker_queue.empty():
            worker_queue.put(prompt.get_prompt())
        random_prompt = worker_queue.get()
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
                resp = bot.send_photo(telegram_chat_id, photo=image, caption=random_prompt)
                if resp.id:
                    logging.info("https://t.me/{}/{}".format(resp.chat.username, resp.message_id))
                else:
                    logging.error(resp)
        except Exception as e:
            logging.error(e)
            worker_queue.put(random_prompt)
        else:
            worker_queue.task_done()
        if not command_only_mode:
            time.sleep(sleep_time)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    user = bot.get_me()
    logging.info('Starting bot with username: {}'.format(user.username))
    if len(sys.argv) > 1:
        worker_queue.put(sys.argv[1])
    threading.Thread(target=main_loop, daemon=True).start()
    bot.infinity_polling()
    logging.info('Bot stopped')
