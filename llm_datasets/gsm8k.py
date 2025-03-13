import os
from torch.utils.data import Dataset
import logging
from utils import *

logger = logging.getLogger(__name__)


class GSM8K(Dataset):
    def __init__(self, args, with_reasoning=True, cache=True, name=None, budget=None, split='train'):
        from utils import create_gsm8k_prompt
        self.args = args
        self.cache = cache
        self.split = split
        self.with_reasoning = with_reasoning
        if budget is not None:
            global gsm8k_prompts
            gsm8k_prompts = create_gsm8k_prompt(budget)
        self.gsm8k_std_data_sets = self._load_data()
        logger.info(f"Loading dataset from the GSM8k-{split}!")
        self.dataset = sum(self.gsm8k_std_data_sets.values(), [])

    def _generate_configs(self):
        config = [dict(abbr='GSM8K',
                       path=f'./data/GSM8K',
                       name=f'GSM8K-{self.split}',
                       reader_cfg=dict(
                           input_column='question',
                           output_column='answer'
                       ),
                       meta_prompt=dict(
                           round=gsm8k_prompts['reasoning'] if self.with_reasoning else gsm8k_prompts['no_reasoning'],
                       ),
                       )
                  ]
        return config

    @staticmethod
    def _generate_std_subset(raw_data, cfg):
        examples = []
        prompt_template = cfg["meta_prompt"]["round"][0]['prompt']
        for item in raw_data:
            examples.append(dict(
                gold=item['answer'],
                reasoning_process_main=item['reasoning_process_main'],
                reasoning_process_socratic=item['reasoning_process_socratic'],
                round=[
                    {
                        "role": "HUMAN",
                        "prompt": prompt_template.replace("{question}", item['question'])
                    },
                    {
                        "role": "BOT",
                        "prompt": "{answer}"
                    },
                ]
            ))
        return examples

    def _generate_formal_info(self, cfg):
        from utils import read_jsonl
        def find_answer(text):
            pattern = r"#### (-?\d+(?:\.\d+)?|\d+/\d+)"
            return re.findall(pattern, text)[-1]

        data_main = read_jsonl(f'data/GSM8K/gsm8k-main-{self.split}.jsonl')
        data_socratic = read_jsonl(f'data/GSM8K/gsm8k-socratic-{self.split}.jsonl')
        data = []
        for idx in range(len(data_main)):
            assert data_main[idx]['question'] == data_socratic[idx]['question']
            data.append({
                'question': data_main[idx]['question'].strip(),
                'answer': find_answer(data_main[idx]['answer']),
                'reasoning_process_main': data_main[idx]['answer'],
                'reasoning_process_socratic': data_socratic[idx]['answer'],
            })
            assert data[-1]['answer'] in data_main[idx]['answer'] and data[-1]['answer'] in data_socratic[idx]['answer']
        return data

    def _load_data(self):
        from utils import save_to_jsonl
        cfgs = self._generate_configs()
        std_data_sets = {}
        for cfg in cfgs:
            # [{question, answer, reasoning_process_main, reasoning_process_socratic}]
            info = self._generate_formal_info(cfg)
            std_subset = self._generate_std_subset(info, cfg)
            save_to_jsonl(std_subset, os.path.join('./.cache', cfg["abbr"]) + f'-{self.split}.jsonl')
            # print(f"the path is {os.path.join('./.cache', cfg['abbr']) + f'-{self.split}.jsonl'}")
            std_data_sets[cfg["abbr"]] = std_subset
        return std_data_sets

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        return self.dataset[i]
