from diffusers.models import AutoencoderKL
from diffusers import StableDiffusionPipeline,LMSDiscreteScheduler
import config
import os
import random
import torch
import sys
import threading


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
if config.cfg.low_vram:
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    pipe = StableDiffusionPipeline.from_pretrained(config.cfg.sd_model_id, vae=vae, revision="fp16", torch_dtype=torch.float16)
    pipe.enable_attention_slicing()
else:
    pipe = StableDiffusionPipeline.from_pretrained(config.cfg.sd_model_id, vae=vae)
pipe.safety_checker = lambda images, **kwargs: (images, False)
pipe.scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")
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
        if device.type == 'cuda':
            with torch.autocast('cuda'):
                image = pipe(
                    prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=steps,
                    guidance_scale=scale,
                    generator=generator
                ).images[0]
        else:
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
    images = generate(prompt)
    for i, image in enumerate(images):
        if len(images) > 1:
            filename = f"{prompt}_{i+1}.png"
        else:
            filename = f"{prompt}.png"
        print(f'Saving {filename}...')
        image.save(filename)
