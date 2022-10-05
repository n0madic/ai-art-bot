from diffusers import StableDiffusionPipeline,LMSDiscreteScheduler
import random
import torch
import sys
import threading

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sd_model_id = 'CompVis/stable-diffusion-v1-4'

scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")
pipe = StableDiffusionPipeline.from_pretrained(sd_model_id, use_auth_token=True)
pipe.safety_checker = lambda images, **kwargs: (images, False)
pipe.to(device)
lock = threading.Lock()


def generate(prompt, seed, scale=7.5, steps=50):
    seed = seed or random.randint(0, 2**32 - 1)
    try:
        lock.acquire()
        generator = torch.Generator(device=device).manual_seed(int(seed))
        if device.type == 'cuda':
            with torch.autocast('cuda'):
                image = pipe(
                    [prompt],
                    num_inference_steps=steps,
                    guidance_scale=scale,
                    generator=generator,
                    scheduler=scheduler
                ).images[0]
        else:
            image = pipe(
                [prompt],
                num_inference_steps=steps,
                guidance_scale=scale,
                generator=generator,
                scheduler=scheduler
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
