def load_prompt(prompt_path):
    with open(prompt_path, 'r') as f:
        prompt = f.read()
    return prompt

