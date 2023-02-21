from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
from diffusers.utils.import_utils import is_xformers_available
import config
import logging
import os
import random
import torch
import sys
import threading


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

StableDiffusionSafetyChecker.forward = lambda self, clip_input, images: (
    images, False)
pipe = StableDiffusionPipeline.from_pretrained(
    config.cfg.sd_model_id,
    revision='fp16',
    torch_dtype=torch.float16,
)
if config.cfg.low_vram:
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    pipe.enable_attention_slicing()
if is_xformers_available():
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception as e:
        logging.warning(
            'Could not enable memory efficient attention. Make sure xformers is installed'
            ' correctly and a GPU is available: {e}'
        )
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.to(device)
lock = threading.Lock()


def get_negative_prompt(filename='negative.txt'):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return ', '.join([line.rstrip() for line in f])
    else:
        return ''


def generate(prompt, negative_prompt='', seed=0, scale=7.5, steps=50):
    if not negative_prompt:
        negative_prompt = get_negative_prompt()
    seed = seed or random.SystemRandom().randint(0, 2**32 - 1)
    try:
        lock.acquire()
        generator = torch.Generator(device=device).manual_seed(int(seed))
        image = pipe(
            prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=scale,
            generator=generator
        ).images[0]
    finally:
        lock.release()
    image.info['prompt'] = prompt
    image.info['seed'] = seed
    image.info['scale'] = scale
    image.info['steps'] = steps
    return image


if __name__ == "__main__":
    print(f'Used device: {device}')
    if len(sys.argv) > 1:
        prompt = sys.argv[1]
    else:
        prompt = "test"
    print(f'Generating image for prompt: {prompt}')
    image = generate(prompt)
    filename = f"{prompt}.png"
    print(f'Saving {filename}...')
    image.save(filename)
