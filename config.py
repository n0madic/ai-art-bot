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
        self.sleep_time = float(os.getenv('SLEEP_TIME', 60))
        self.telegram_token = os.getenv('TELEGRAM_TOKEN')
        telegram_admin_id = os.getenv('TELEGRAM_ADMIN_ID')
        if telegram_admin_id:
            self.telegram_admin_ids = [int(i) for i in telegram_admin_id.split(',')]
        else:
            self.telegram_admin_ids = []
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.instagram_username = os.getenv('INSTAGRAM_USERNAME')
        self.instagram_password = os.getenv('INSTAGRAM_PASSWORD')
        self.image_cache_dir = os.getenv('IMAGE_CACHE_DIR', 'imagecache')
        self.random_prompt_probability = float(os.getenv('RANDOM_PROMPT_PROBABILITY', 0.5))


cfg = Config()


if __name__ == '__main__':
    import json
    print(json.dumps(cfg.__dict__, indent=4))
