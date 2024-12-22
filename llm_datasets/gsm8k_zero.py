import os
from torch.utils.data import Dataset
import logging
from utils import *

logger = logging.getLogger(__name__)


class GSM8KZero(Dataset):
    def __init__(self, args, with_reasoning=True, name=None, cache=True, budget=None):
        self.args = args
        self.cache = cache
        self.with_reasoning = with_reasoning
        if budget is not None:
            global gsm8k_prompts
            gsm8k_prompts = create_gsm8k_prompt(budget)

        self.gsm8k_std_data_sets = self._load_data()
        logger.info("Loading dataset from the GSM8k-Zero!")
        self.dataset = sum(self.gsm8k_std_data_sets.values(), [])

    def _generate_configs(self):
        config = [dict(abbr='GSM8K',
                       path=f'./data/GSM8K-Zero',
                       name='GSM8K-Zero',
                       reader_cfg=dict(
                           input_column='question',
                           output_column='answer'
                       ),
                       meta_prompt=dict(
                           round=gsm8k_prompts['reasoning'] if self.with_reasoning else gsm8k_prompts['no_reasoning'],
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
            examples.append(dict(
                gold=item['answer'],
                round=[
                    {
                        "role": "HUMAN",
                        "prompt": prompt_template.replace("{question}", item['question'])
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
                    'answer': str(int(entry['answer'])).strip()
                })
        return data

    def _load_data(self):
        cfgs = self._generate_configs()
        save_config(cfgs[0])
        std_data_sets = {}
        for cfg in cfgs:
            info = self._generate_formal_info(cfg)
            std_subset = self._generate_std_subset(info, cfg)
            save_to_jsonl(std_subset, os.path.join('./.cache', cfg["abbr"]) + '.jsonl')
            std_data_sets[cfg["abbr"]] = std_subset
        return std_data_sets

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):

        return self.dataset[i]


def test():
    os.chdir('..')
    dataset = GSM8KZero(1)

# test()


