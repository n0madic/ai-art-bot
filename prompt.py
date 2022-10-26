import config
import random
import re
import string
import sys
import transformers


gpt2_pipe = transformers.pipeline('text-generation', model='Gustavosta/MagicPrompt-Stable-Diffusion', tokenizer='gpt2')
used_prompts = []


def generate(starting_text='', max_length=100, random_prompt_probability=config.cfg.random_prompt_probability):
    transformers.set_seed(random.SystemRandom().randint(100, 1000000))

    with open('ideas.txt', 'r') as f:
        ideas = f.readlines()

    with open('ignores.txt', 'r') as f:
        ignores = [line.rstrip() for line in f]

    if starting_text == '' and random.random() > random_prompt_probability:
        starting_text: str = ideas[random.randrange(0, len(ideas))].replace('\n', '')
        starting_text: str = re.sub(r'[,:\-–.!;?_]', '', starting_text)

    if len(ideas) == len(used_prompts):
        used_prompts.clear()
    if starting_text:
        used_prompts.append(starting_text)

    prompt = ''
    tries = 0
    while not prompt:
        if tries > 10:
            print('ERROR: Could not find a prompt!')
            return
        responses = []
        for r in gpt2_pipe(starting_text, max_length=random.randint(60, max_length), num_return_sequences=4):
            resp = r['generated_text'].strip()
            if resp and resp != starting_text and len(resp) > (len(starting_text) * 2) and not resp.endswith((':', '-', '—')) and not resp.find('--'):
                continue
            if not any([i.lower() in resp.lower() for i in ignores]):
                responses.append(resp)
        for r in responses:
            response_end = r.strip(string.punctuation)
            response_end = re.sub(r'[^ ]+\.[^ ]+','', response_end)
            response_end = re.sub(r'\(\s*\)','', response_end)
            response_end = response_end.replace(',,', ',')
            response_end = response_end.replace('| |', '|')
            response_end = response_end.replace('<', '').replace('>', '')
            response_end = response_end.replace('[[', '').replace(']]', '')
            response_end = response_end.replace('((', '').replace('))', '')
            response_end = ' '.join(response_end.split()).strip()
            if response_end and len(response_end) > 20:
                prompt = response_end
                break
        tries += 1
    return prompt


if __name__ == '__main__':
    starting_text = ''
    if len(sys.argv) > 1:
        starting_text = sys.argv[1]
    delimiter = '=' * 80
    for _ in range(10):
        prompt = generate(starting_text)
        if not prompt:
            raise Exception('No prompt generated')
        print('{}\n{}\n{}'.format(delimiter, prompt, delimiter), end='\n\n')
