import json
import random


used_prompts = []


def get_random_string(list, probability=1.0, count=1):
    result = []
    for _ in range(random.randint(1, count)):
        # check duplicates
        while True:
            item = random.choice(list)
            if not item in result:
                break
        # check probability
        if random.random() < probability:
            result.append(item)
    return result


def get_prompt(init_prompt=''):
    with open('prompt.json') as f:
        keywords = json.load(f)
    generated = []
    if init_prompt:
        generated.append(init_prompt.strip().removesuffix(','))
    else:
        if len(keywords['Prompts']) == len(used_prompts):
            used_prompts.clear()
        while True:
            general_prompt = get_random_string(keywords['Prompts'])[0]
            if not general_prompt in used_prompts:
                generated.append(general_prompt)
                used_prompts.append(general_prompt)
                break
    generated.append('made by ' + ' and '.join(get_random_string(keywords['Artists'], count=3)))
    generated.extend(get_random_string(keywords['Lighting']))
    generated.extend(get_random_string(keywords['Style'], 0.9))
    generated.extend(get_random_string(keywords['Technology'], 0.8))
    generated.extend(get_random_string(keywords['Angles'], 0.2))
    generated.extend(get_random_string(keywords['Effects'], 0.2))
    generated.extend(get_random_string(keywords['Filters'], 0.1))
    generated.extend(get_random_string(keywords['Lenses'], 0.1))
    generated.extend(get_random_string(keywords['Material'], 0.1))
    generated.extend(get_random_string(keywords['Photo styles'], 0.1))
    generated.extend(get_random_string(keywords['Photography'], 0.2))
    generated.extend(get_random_string(keywords['Technique'], 0.1))
    generated.extend(get_random_string(keywords['Textures'], 0.1))
    generated.extend(get_random_string(keywords['Modifiers'], count=8))
    return ', '.join([g for g in generated if g is not None])


if __name__ == '__main__':
    for _ in range(10):
        print(get_prompt(), end='\n\n')
