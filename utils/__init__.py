import json
import re
from .prompt_template import *
import tiktoken

tokenizer = tiktoken.encoding_for_model("gpt-4")


def extract_question(question):
    new_question = question.replace(
        "以下是一道关于数学的单项选择题，请你一步一步推理，并在最后用“所以答案为选项X”给出答案，其中“X”为选项A，B，C，D中你认为正确的选项。下面是你要回答的问题\n", ''
    ).replace(
        "Here is a multiple-choice question about mathematics. Please reason through it step by step, and at the end, "
        "provide your answer option with 'Therefore, the correct answer is option X', Where 'X' is the correct option "
        "you think from A，B，C，D. Here is the question you need to answer:\n", ''
    ).replace(
        "请严格按照如下格式回答：[[选项]]，例如：选项: [[A]]。\n让我们一步一步思考：\n", ''
    ).replace(
        "Please Give the response by strictly following this format: [[choice]],"
        "for example: Choice: [[A]].\nLet's think step by step:\n", ''
    ).replace(
        "Please answer the following question directly and give the answer directly without any reasoning process. "
        "Please strictLy follow the format: [[choice]],for example: Choice: [[A]].\n", ''
    )
    return new_question


def add_budget(question, budget):
    new_question = question \
        .replace("Let's think step by step:\n", f"Let's think step by step and use less than {budget} tokens:\n") \
        .replace("让我们一步一步思考：\n", f"让我们一步一步思考并使用少于 {budget} tokens:\n")
    return new_question


def token_measure(text):
    return len(tokenizer.encode(text))


def create_zero_shot_context():
    zero_shot_context = """Task: Analyze the given question and estimate the minimum number of tokens required to generate a complete and accurate response. Please Give the response by strictly following this format: [[budget]],for example: Budget: [[12]]."""
    return zero_shot_context


def save_config(nested_dict_list, filepath='demo_config.py', name='demo_config'):
    config_content = f"{name} = " + json.dumps(nested_dict_list, indent=4, ensure_ascii=False)
    with open(filepath, 'w', encoding='utf-8') as file:
        file.write(config_content)


def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data


def get_number(options):
    result_string = ''
    for i, option in enumerate(options, start=ord('A')):
        result_string += f'{chr(i)}. {option}\n'
    return result_string


def extract_number(text):
    import re
    pattern = r"\[\[(\d+)\]\]"
    match = re.search(pattern, text)
    if match:
        result = match.group(1)
        return result
    else:
        return -1


def save_to_jsonl(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        for item in data:
            json_line = json.dumps(item, ensure_ascii=False)
            file.write(json_line + '\n')


mathbench_name_list = ['mathbench-college-single_choice_cn', 'mathbench-college-single_choice_en',
                       'mathbench-high-single_choice_cn', 'mathbench-high-single_choice_en',
                       'mathbench-middle-single_choice_cn',
                       'mathbench-middle-single_choice_en', 'mathbench-primary-cloze_cn', 'mathbench-primary-cloze_en',
                       'mathbench-arithmetic-cloze_en', 'mathbench-college_knowledge-single_choice_cn',
                       'mathbench-college_knowledge-single_choice_en', 'mathbench-high_knowledge-single_choice_cn',
                       'mathbench-high_knowledge-single_choice_en', 'mathbench-middle_knowledge-single_choice_cn',
                       'mathbench-middle_knowledge-single_choice_en', 'mathbench-primary_knowledge-single_choice_cn',
                       'mathbench-primary_knowledge-single_choice_en']
