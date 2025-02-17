import os
from torch.utils.data import Dataset
import logging
from utils import *

logger = logging.getLogger(__name__)


class GPQA(Dataset):
    def __init__(self, args, with_reasoning=True, name=None, cache=True, budget=None):
        self.args = args
        self.cache = cache
        self.with_reasoning = with_reasoning
        if budget is not None:
            global gpqa_prompts
            gpqa_prompts = create_gpqa_prompt(budget)
        self.gpqa_std_data_sets = self._load_data()
        logger.info("Loading dataset from the GPQA!")
        self.dataset = sum(self.gpqa_std_data_sets.values(), [])

    def _generate_configs(self):
        config = [dict(abbr='GPQA',
                       path=f'data/gpqa',
                       name='GPQA',
                       reader_cfg=dict(
                           input_column='question',
                           output_column='answer'
                       ),
                       meta_prompt=dict(
                           round=gpqa_prompts['reasoning'] if self.with_reasoning else gpqa_prompts['no_reasoning'],
                       ),
                       )
                  ]
        save_config(config[0])
        return config

    @staticmethod
    def _generate_std_subset(raw_data, cfg):
        examples = []
        prompt_template = cfg["meta_prompt"]["round"][0]['prompt']
        for item in raw_data:
            new_question = f"""{item['question']}\nA. {item['answer']}\nB. {item['wa1']}\nC. {item['wa2']}\nD. {item['wa3']}"""
            examples.append(dict(
                gold='A',
                # gold=item['answer'],
                round=[
                    {
                        "role": "HUMAN",
                        "prompt": prompt_template.replace("{question}", new_question)
                    },
                    {
                        "role": "BOT",
                        "prompt": "{answer}"
                    }
                ]
            ))
        return examples

    @staticmethod
    def _generate_formal_info(cfg):
        data = []
        filename = f"{os.path.join(cfg['path'], cfg['name'])}.jsonl"
        with open(filename, 'r', encoding='utf-8') as infile:
            for ids, line in enumerate(infile):
                entry = json.loads(line)
                data.append({
                    'question': entry['question'].strip(),
                    'answer': entry['answer'].strip(),
                    'wa1': entry['wrong_answer_1'].strip(),
                    'wa2': entry['wrong_answer_2'].strip(),
                    'wa3': entry['wrong_answer_3'].strip(),
                })
        return data

    def _load_data(self):
        cfgs = self._generate_configs()
        save_config(cfgs[0])
        std_data_sets = {}
        for cfg in cfgs:
            info = self._generate_formal_info(cfg)  # [{question, answer}]
            std_subset = self._generate_std_subset(info, cfg)
            save_to_jsonl(std_subset, os.path.join('./.cache', cfg["abbr"]) + '.jsonl')
            std_data_sets[cfg["abbr"]] = std_subset
        return std_data_sets

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):

        return self.dataset[i]


def test():
    os.chdir(r'the path')
    dataset = GPQA(1, with_reasoning=False, name=None, cache=True, budget=512)

# test()

