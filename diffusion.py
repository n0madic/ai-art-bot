from diffusers.models import AutoencoderKL
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
from diffusers.utils.import_utils import is_xformers_available
import gc
import logging
import os
import random
import torch
import sys
import threading


class Pipeline:
    '''Wrapper around DiffusionPipeline to make it thread-safe'''
    device = torch.device('cuda' if torch.cuda.is_available(
    ) else 'mps' if torch.backends.mps.is_available() else 'cpu')
    lock = threading.Lock()

    def __init__(self,
                 sd_model_id: str,
                 sd_model_vae_id: str = None,
                 sd_refiner_id: str = None,
                 fp16: bool = False,
                 low_vram: bool = False,
                 ):
        self.sd_model_id = sd_model_id
        self.sd_refiner_id = sd_refiner_id
        self.sd_model_vae_id = sd_model_vae_id
        self.low_vram = low_vram
        self.variant = None
        self.torch_dtype = None
        if fp16:
            self.variant = 'fp16'
            self.torch_dtype = torch.float16
        if self.low_vram:
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True

    def __init_pipeline(self, model_id, vae=None):
        '''Initialize the pipeline'''
        if vae:
            pipe = DiffusionPipeline.from_pretrained(
                model_id,
                variant=self.variant,
                torch_dtype=self.torch_dtype,
                safety_checker=None,
                requires_safety_checker=False,
                vae=AutoencoderKL.from_pretrained(self.sd_model_vae_id)
            )
        else:
            pipe = DiffusionPipeline.from_pretrained(
                model_id,
                variant=self.variant,
                torch_dtype=self.torch_dtype,
                safety_checker=None,
                requires_safety_checker=False
            )
        if self.low_vram:
            pipe.enable_attention_slicing()
        if is_xformers_available():
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception as e:
                logging.warning(
                    'Could not enable memory efficient attention. Make sure xformers is installed'
                    ' correctly and a GPU is available: {e}'
                )
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipe.scheduler.config)
        try:
            pipe.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)
        except Exception as e:
            logging.warning('Could not enable FreeU: {e}')
        if self.low_vram:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(self.device)
        return pipe

    def load_pipe(self):
        self.pipe = self.__init_pipeline(
            self.sd_model_id, vae=self.sd_model_vae_id)

    def load_refiner(self):
        self.refiner = self.__init_pipeline(self.sd_refiner_id)

    def unload_pipe(self):
        '''Unload the pipeline'''
        if hasattr(self, 'pipe'):
            self.pipe = None
            del self.pipe
            torch.clear_autocast_cache()
        gc.collect()

    def unload_refiner(self):
        '''Unload the refiner'''
        if hasattr(self, 'refiner'):
            self.refiner = None
            del self.refiner
            torch.clear_autocast_cache()
        gc.collect()

    def __del__(self):
        '''Unload the pipeline and refiner'''
        self.unload_pipe()
        self.unload_refiner()

    def generate(self, prompt, negative_prompt='', seed=0, scale=7.5, steps=50, width=512, height=512):
        '''Generate an image for the given prompt'''
        if not hasattr(self, 'pipe'):
            self.load_pipe()
        if not negative_prompt:
            negative_prompt = get_negative_prompt()
        seed = seed or random.SystemRandom().randint(0, 2**32 - 1)
        output_type = 'pil'
        if self.sd_refiner_id:
            output_type = 'latent'
        try:
            self.lock.acquire()
            generator = torch.Generator(
                device=self.device).manual_seed(int(seed))
            image = self.pipe(
                prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=scale,
                output_type=output_type,
                generator=generator,
                width=width,
                height=height,
            ).images[0]
            if self.sd_refiner_id:
                if self.low_vram:
                    self.unload_pipe()
                if not hasattr(self, 'refiner'):
                    self.load_refiner()
                image = self.refiner(
                    prompt=prompt,
                    image=image,
                    negative_prompt=negative_prompt,
                    num_inference_steps=steps,
                    guidance_scale=scale,
                    generator=generator,
                    width=width,
                    height=height,
                ).images[0]
        finally:
            if self.low_vram:
                self.unload_refiner()
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

    pipe = Pipeline(config.cfg.sd_model_id, config.cfg.sd_model_vae_id, config.cfg.sd_refiner_id,
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
