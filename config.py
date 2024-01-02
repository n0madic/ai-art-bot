import os
import logging


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
                        os.environ[key] = value

        self.command_only_mode = os.getenv('COMMAND_ONLY_MODE', 'false').lower() in ['true', 'on', 'yes', '1']
        self.face_enhancer_arch = os.getenv('FACE_ENHANCER_ARCH', 'CodeFormer')
        self.face_enhancer_model_path = os.getenv('FACE_ENHANCER_MODEL_PATH', 'gfpgan/CodeFormer.pth')
        self.fp16 = os.getenv('FP16', 'false').lower() in ['true', 'on', 'yes', '1']
        self.image_cache_dir = os.getenv('IMAGE_CACHE_DIR', 'imagecache')
        self.low_vram = os.getenv('LOW_VRAM', 'false').lower() in ['true', 'on', 'yes', '1']
        self.premoderation = os.getenv('PREMODERATION', 'false').lower() in ['true', 'on', 'yes', '1']
        self.prompt_model_id = os.getenv('PROMPT_MODEL_ID', 'Gustavosta/MagicPrompt-Stable-Diffusion')
        self.prompt_model_tokenizer = os.getenv('PROMPT_MODEL_TOKENIZER', 'gpt2')
        self.prompt_prefix = os.getenv('PROMPT_PREFIX')
        self.random_prompt_probability = float(os.getenv('RANDOM_PROMPT_PROBABILITY', 0.5))
        self.realesrgan_model_path = os.getenv('REALESRGAN_MODEL_PATH', 'realesrgan/RealESRGAN_x4plus.pth')
        self.sd_model_id = os.getenv('SD_MODEL_ID', 'stabilityai/stable-diffusion-2-1')
        self.sd_model_vae_id = os.getenv('SD_MODEL_VAE_ID')
        self.sd_refiner_id = os.getenv('SD_REFINER_ID')
        self.sleep_time = float(os.getenv('SLEEP_TIME', 600))
        self.telegram_token = os.getenv('TELEGRAM_TOKEN')
        telegram_admin_id = os.getenv('TELEGRAM_ADMIN_ID')
        if telegram_admin_id:
            self.telegram_admin_ids = [int(i) for i in telegram_admin_id.split(',')]
        else:
            self.telegram_admin_ids = []
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.telegram_turbo_chat_id = os.getenv('TELEGRAM_TURBO_CHAT_ID')
        self.turbo_sleep_time = float(os.getenv('TURBO_SLEEP_TIME', 60))
        self.twitter_consumer_key = os.getenv('TWITTER_CONSUMER_KEY')
        self.twitter_consumer_secret = os.getenv('TWITTER_CONSUMER_SECRET')
        self.twitter_access_token = os.getenv('TWITTER_ACCESS_TOKEN')
        self.twitter_access_token_secret = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
        self.upscaling = os.getenv('UPSCALING', 'true').lower() in ['true', 'on', 'yes', '1']


cfg = Config()


if __name__ == '__main__':
    import json
    print(json.dumps(cfg.__dict__, indent=4))
