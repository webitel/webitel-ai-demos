

def load_prompt(prompt_path):
    with open(prompt_path, 'r') as f:
        prompt = f.read()
    return prompt

load_prompt('prompts/query_analyzer_prompt.txt')