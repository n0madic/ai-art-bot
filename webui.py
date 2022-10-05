import diffusion
import enhancement
import gradio
import prompt


def webui_fn(prompt_input, scale, steps, seed, enhancements):
    if not prompt_input or prompt_input.endswith('+'):
        prompt_input = prompt.generate(prompt_input.removesuffix('+'))
    image = diffusion.generate(prompt_input, seed=seed, scale=scale, steps=steps)
    message = '{} (seed={} scale={} steps={})'.format(prompt_input, image.info['seed'], image.info['scale'], image.info['steps'])
    if 'Upscale' in enhancements or 'Face restore' in enhancements:
        image = enhancement.upscale(image, face_restore='Face restore' in enhancements)
    return image, message


gr = gradio.Interface(
    fn=webui_fn,
    inputs=[
        gradio.Textbox(placeholder="Place your input prompt here", label="Input Prompt"),
        gradio.Slider(0, 20, 7.5, step=0.5, label="Guidance Scale"),
        gradio.Slider(1, 200, 50, step=1, label="Steps"),
        gradio.Number(None, label="Seed", precision=0),
        gradio.CheckboxGroup(['Upscale', 'Face restore'], label="Enhancements"),
    ],
    outputs=[
        gradio.Image(type="pil"),
        gradio.Textbox(label='Output message').style(rounded=True),
    ],
    allow_flagging="never",
    examples=[
        [prompt.generate()]
    ],
)
