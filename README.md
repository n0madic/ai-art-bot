# AI art bot

The app generates random images using AI [Stable Diffusion](https://github.com/CompVis/stable-diffusion) and posts them in Telegram https://t.me/ai_art_random

## Configuring

Environment variables:

* `HUGGING_FACE_HUB_TOKEN` - token for Hugging Face for downloading models
* `TELEGRAM_TOKEN` - bot token
* `TELEGRAM_ADMIN_ID` - user ID to manage the bot
* `TELEGRAM_CHAT_ID` - chat where images will be sent
* `UPSCALING` - up to 4x image resolution with [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) (default `true`)
* `COMMAND_ONLY_MODE` - bot command mode only
* `FACE_RESTORING` - restoring faces in an image with [GFPGAN](https://github.com/TencentARC/GFPGAN) (required enabled upscaling)
* `SLEEP_TIME` - how many seconds to sleep between generations (default 60s)

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
