import random
import re
import string
import sys
import transformers


class Prompt:
    def __init__(self, prompt_model_id, prompt_model_tokenizer, sd_model_id, prompt_prefix='') -> None:
        self.gpt2_pipe = transformers.pipeline(
            'text-generation',
            model=prompt_model_id,
            tokenizer=prompt_model_tokenizer,
        )
        self.tokenizer = transformers.CLIPTokenizer.from_pretrained(sd_model_id, subfolder='tokenizer')
        self.prompt_prefix = prompt_prefix
        self.used_ideas = []

    def token_count(self, prompt):
        return self.tokenizer(prompt, return_tensors='pt')['input_ids'].shape[-1]

    def generate(self, starting_text='', max_length=100, random_prompt_probability=0.5):
        transformers.set_seed(random.SystemRandom().randint(100, 1000000))

        with open('ideas.txt', 'r') as f:
            ideas = f.readlines()

        with open('ignores.txt', 'r') as f:
            ignores = [line.rstrip() for line in f]

        if len(ideas) == len(self.used_ideas):
            self.used_ideas.clear()

        if starting_text == '' and random.random() > random_prompt_probability:
            while True:
                starting_text = ideas[random.randrange(0, len(ideas))].replace('\n', '')
                if starting_text not in self.used_ideas:
                    self.used_ideas.append(starting_text)
                    break
            starting_text = re.sub(r'[,:\-–.!;?_]', '', starting_text)

        prompt = ''
        tries = 0
        while not prompt:
            if tries > 10:
                print('ERROR: Could not find a prompt!')
                return
            responses = []
            for r in self.gpt2_pipe(starting_text, max_length=random.randint(60, max_length), num_return_sequences=4):
                resp = r['generated_text'].strip()
                if resp and resp != starting_text and len(resp) > (len(starting_text) * 2) and not resp.endswith((':', '-', '—')) and not resp.find('--'):
                    continue
                if not any([i.lower() in resp.lower() for i in ignores]) and self.token_count(resp) <= 77:
                    responses.append(resp)
            for r in responses:
                response_end = r.strip(string.punctuation)
                response_end = response_end.encode('ascii', 'ignore').decode('ascii')
                response_end = re.sub(r'[^ ]+\.[^ ]+','', response_end)
                response_end = re.sub(r'\(\s*\)','', response_end)
                response_end = re.sub(r'^\W+|\W+$','', response_end)
                response_end = re.sub(r'!+', '!', response_end)
                response_end = response_end.replace(',,', ',')
                response_end = response_end.replace('| |', '|')
                response_end = response_end.replace('<', '').replace('>', '')
                response_end = response_end.replace('[[', '').replace(']]', '')
                response_end = response_end.replace('((', '').replace('))', '')
                if self.prompt_prefix:
                    response_end = self.prompt_prefix + ', ' + response_end
                response_end = ' '.join(response_end.split()).strip()
                if response_end and len(response_end) > 20:
                    prompt = response_end
                    break
            tries += 1
        return prompt


if __name__ == '__main__':
    starting_text = ''
    if len(sys.argv) > 1:
        starting_text = ' '.join(sys.argv[1:])
    delimiter = '=' * 80
    prompt = Prompt('n0madic/ai-art-random-prompts', 'distilgpt2', 'stabilityai/stable-diffusion-2-1')
    for _ in range(10):
        prompt = prompt.generate(starting_text)
        if not prompt:
            raise Exception('No prompt generated')
        print('{}\n{}\n{}'.format(delimiter, prompt, delimiter), end='\n\n')
