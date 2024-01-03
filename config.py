import os


def load(env_file='.env'):
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    os.environ[key] = value

    config = {}

    config['command_only_mode'] = os.getenv('COMMAND_ONLY_MODE', 'false').lower() in ['true', 'on', 'yes', '1']
    config['face_enhancer_arch'] = os.getenv('FACE_ENHANCER_ARCH', 'CodeFormer')
    config['face_enhancer_model_path'] = os.getenv('FACE_ENHANCER_MODEL_PATH', 'gfpgan/CodeFormer.pth')
    config['fp16'] = os.getenv('FP16', 'false').lower() in ['true', 'on', 'yes', '1']
    config['image_cache_dir'] = os.getenv('IMAGE_CACHE_DIR', 'imagecache')
    config['low_vram'] = os.getenv('LOW_VRAM', 'false').lower() in ['true', 'on', 'yes', '1']
    config['premoderation'] = os.getenv('PREMODERATION', 'false').lower() in ['true', 'on', 'yes', '1']
    config['prompt_model_id'] = os.getenv('PROMPT_MODEL_ID', 'n0madic/ai-art-random-prompts')
    config['prompt_model_tokenizer'] = os.getenv('PROMPT_MODEL_TOKENIZER', 'distilgpt2')
    config['prompt_prefix'] = os.getenv('PROMPT_PREFIX')
    config['random_prompt_probability'] = float(os.getenv('RANDOM_PROMPT_PROBABILITY', 0.5))
    config['realesrgan_model_path'] = os.getenv('REALESRGAN_MODEL_PATH', 'realesrgan/RealESRGAN_x4plus.pth')
    config['image_width'], config['image_height'] = [int(i) for i in os.getenv('RESOLUTION', '512x512').lower().split('x')]
    config['sd_model_id'] = os.getenv('SD_MODEL_ID', 'stabilityai/stable-diffusion-2-1')
    config['sd_model_vae_id'] = os.getenv('SD_MODEL_VAE_ID')
    config['sd_refiner_id'] = os.getenv('SD_REFINER_ID')
    config['sleep_time'] = float(os.getenv('SLEEP_TIME', 600))
    config['telegram_token'] = os.getenv('TELEGRAM_TOKEN')
    config['telegram_admin_ids'] = [int(i) for i in os.getenv('TELEGRAM_ADMIN_ID').split(',')] if os.getenv('TELEGRAM_ADMIN_ID') else []
    config['telegram_chat_id'] = os.getenv('TELEGRAM_CHAT_ID')
    config['telegram_turbo_chat_id'] = os.getenv('TELEGRAM_TURBO_CHAT_ID')
    config['turbo_sleep_time'] = float(os.getenv('TURBO_SLEEP_TIME', 60))
    config['twitter_consumer_key'] = os.getenv('TWITTER_CONSUMER_KEY')
    config['twitter_consumer_secret'] = os.getenv('TWITTER_CONSUMER_SECRET')
    config['twitter_access_token'] = os.getenv('TWITTER_ACCESS_TOKEN')
    config['twitter_access_token_secret'] = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
    config['upscaling'] = os.getenv('UPSCALING', 'true').lower() in ['true', 'on', 'yes', '1']

    return config


if __name__ == '__main__':
    import json
    print(json.dumps(load(), indent=4))
