import diffusion
import enhancement
import io
import logging
import os
import prompt
import requests
import sys
import time


telegram_token = os.getenv('TELEGRAM_TOKEN')
telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')


def send_image_to_telegram(caption, image_data):
    response = requests.post(
        'https://api.telegram.org/bot{}/sendPhoto'.format(telegram_token),
        data={
            'chat_id': telegram_chat_id,
            'caption': caption,
        },
        files={
            'photo': io.BytesIO(image_data),
        },
    )

    return response.json()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logging.info('Used device: {}'.format(diffusion.device))
    retry = False
    while True:
        if len(sys.argv) > 1:
            random_prompt = sys.argv[1]
        elif not retry:
            random_prompt = prompt.get_prompt()
        logging.info('Generating image for prompt: {}'.format(random_prompt))
        try:
            images = diffusion.generate(random_prompt)
            retry = False
            for image in images:
                if enhancement.upscaling:
                    logging.info('Upscaling...')
                    face_restore = enhancement.face_presence_detection(image)
                    if face_restore:
                        logging.info('Faces detected, restoring...')
                    image = enhancement.upscale(image, face_restore=face_restore)
                with io.BytesIO() as output:
                    image.save(output, format='PNG')
                    byteImg = output.getvalue()
                logging.info('Send image to Telegram...')
                resp = send_image_to_telegram(random_prompt, byteImg)
                if resp['ok']:
                    logging.info("https://t.me/{}/{}".format(resp['result']['chat']['username'], resp['result']['message_id']))
                else:
                    logging.error(resp)
        except Exception as e:
            logging.error(e)
            retry = True
        time.sleep(float(os.getenv('SLEEP_TIME', 60)))
