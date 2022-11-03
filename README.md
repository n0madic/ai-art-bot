# AI art bot

The app generates random images using AI [Stable Diffusion](https://github.com/CompVis/stable-diffusion) and posts them in Telegram https://t.me/ai_art_random and Twitter https://twitter.com/ai_art_random

## Configuring

Environment variables:

* `COMMAND_ONLY_MODE` - bot command mode only
* `FACE_ENHANCER_ARCH` - face enhancer architecture
* `FACE_ENHANCER_MODEL_PATH` - face enhancer model path for GFPGAN
* `FACE_RESTORING` - restoring faces in an image with [GFPGAN](https://github.com/TencentARC/GFPGAN) (required enabled `UPSCALING`)
* `HUGGING_FACE_HUB_TOKEN` - token for Hugging Face for downloading models
* `INSTAGRAM_USERNAME` - Instagram username
* `INSTAGRAM_PASSWORD` - Instagram password
* `LOW_VRAM` - low video RAM mode
* `RANDOM_PROMPT_PROBABILITY` - probability of generate full random prompt without ideas (default `0.5`)
* `REALESRGAN_MODEL_PATH` - model path for RealESRGAN
* `SD_MODEL_ID` - Hugging Face model id for Stable Diffusion
* `SLEEP_TIME` - how many seconds to sleep between generations (default 600s)
* `TELEGRAM_TOKEN` - Telegram bot token
* `TELEGRAM_ADMIN_ID` - user ID to manage the bot
* `TELEGRAM_CHAT_ID` - chat where images will be sent
* `TELEGRAM_TURBO_CHAT_ID` - chat where images will be sent in turbo mode
* `TURBO_SLEEP_TIME` - how many seconds to sleep between generations in turbo mode (default 60s)
* `TWITTER_CONSUMER_KEY` - Twitter consumer key
* `TWITTER_CONSUMER_SECRET` - Twitter consumer secret
* `TWITTER_ACCESS_TOKEN` - Twitter access token
* `TWITTER_ACCESS_TOKEN_SECRET` - Twitter access token secret
* `UPSCALING` - up to 4x image resolution with [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) (default `true`)

## Usage

Build and running in docker:

```
docker build -t ai-art-bot .
```
```
docker volume create ai-art-bot_huggingface
```
```
docker run --name ai-art-bot -d --env-file .env -v ai-art-bot_huggingface:/root/.cache/huggingface --gpus=all ai-art-bot
```

Running with docker-compose:

```
docker-compose up -d
```
