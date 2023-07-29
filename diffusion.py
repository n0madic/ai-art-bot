from diffusers.models import AutoencoderKL
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
from diffusers.utils.import_utils import is_xformers_available
import logging
import os
import random
import torch
import sys
import threading


class Pipeline:
    '''Wrapper around DiffusionPipeline to make it thread-safe'''
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    lock = threading.Lock()

    def __init__(self, sd_model_id, sd_model_vae_id: str, fp16: bool = False, low_vram: bool = False):
        '''Initialize the pipeline'''
        revision = 'main'
        torch_dtype = None
        if fp16:
            revision='fp16'
            torch_dtype=torch.float16
        if sd_model_vae_id:
            self.pipe = StableDiffusionPipeline.from_pretrained(
                    sd_model_id,
                    revision=revision,
                    torch_dtype=torch_dtype,
                    safety_checker=None,
                    requires_safety_checker=False,
                    vae=AutoencoderKL.from_pretrained(sd_model_vae_id)
                )
        else:
            self.pipe = StableDiffusionPipeline.from_pretrained(
                sd_model_id,
                revision=revision,
                torch_dtype=torch_dtype,
                safety_checker=None,
                requires_safety_checker=False
            )
        if low_vram:
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            self.pipe.enable_attention_slicing()
        if is_xformers_available():
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
            except Exception as e:
                logging.warning(
                    'Could not enable memory efficient attention. Make sure xformers is installed'
                    ' correctly and a GPU is available: {e}'
                )
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.pipe.scheduler.config)
        self.pipe.to(self.device)

    def generate(self, prompt, negative_prompt='', seed=0, scale=7.5, steps=50):
        '''Generate an image for the given prompt'''
        if not negative_prompt:
            negative_prompt = get_negative_prompt()
        seed = seed or random.SystemRandom().randint(0, 2**32 - 1)
        try:
            self.lock.acquire()
            generator = torch.Generator(
                device=self.device).manual_seed(int(seed))
            image = self.pipe(
                prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=scale,
                generator=generator
            ).images[0]
        finally:
            self.lock.release()
        image.info['prompt'] = prompt
        image.info['seed'] = seed
        image.info['scale'] = scale
        image.info['steps'] = steps
        return image


def get_negative_prompt(filename='negative.txt'):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return ', '.join([line.rstrip() for line in f])
    return ''


if __name__ == "__main__":
    import config

    pipe = Pipeline(config.cfg.sd_model_id, config.cfg.sd_model_vae_id,
                    config.cfg.fp16, config.cfg.low_vram)
    print(f'Used device: {pipe.device}')

    if len(sys.argv) > 1:
        prompt = sys.argv[1]
    else:
        prompt = "test"
    print(f'Generating image for prompt: {prompt}')
    image = pipe.generate(prompt, steps=20)
    filename = f"{prompt}.png"
    print(f'Saving {filename}...')
    image.save(filename)
