from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
from diffusers.utils.import_utils import is_xformers_available
import logging
import os
import random
import torch
import sys
import threading


# Disable safety checks to allow for NSFW prompts
StableDiffusionSafetyChecker.forward = lambda self, clip_input, images: (
    images, False)


class Pipeline:
    '''Wrapper around StableDiffusionPipeline to make it thread-safe'''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lock = threading.Lock()

    def __init__(self, sd_model_id: str, fp16: bool = False, low_vram: bool = False):
        '''Initialize the pipeline'''
        if fp16:
            self.pipe = StableDiffusionPipeline.from_pretrained(
                sd_model_id,
                revision='fp16',
                torch_dtype=torch.float16,
            )
        else:
            self.pipe = StableDiffusionPipeline.from_pretrained(sd_model_id)
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

    pipe = Pipeline(config.cfg.sd_model_id,
                    config.cfg.fp16, config.cfg.low_vram)
    print(f'Used device: {pipe.device}')

    if len(sys.argv) > 1:
        prompt = sys.argv[1]
    else:
        prompt = "test"
    print(f'Generating image for prompt: {prompt}')
    image = pipe.generate(prompt)
    filename = f"{prompt}.png"
    print(f'Saving {filename}...')
    image.save(filename)
