import os
from torch.utils.data import Dataset
import logging
from utils import *

logger = logging.getLogger(__name__)


class MathBenchDataset(Dataset):
    def __init__(self, args, with_reasoning=True, name=None, cache=True, budget=None):
        self.args = args
        self.cache = cache
        self.with_reasoning = with_reasoning
        if budget is not None:
            global single_choice_prompts, cloze_prompts
            cloze_prompts = create_cloze_prompt(budget)
            single_choice_prompts = create_prompt(budget)

        self.mathbench_std_data_sets = self._load_data()
        if name is None:
            logger.info("Loading all subsets from the mathbenchmark!")
            self.dataset = sum(self.mathbench_std_data_sets.values(), [])
        else:
            logger.info(f'Loading {name}')
            self.dataset = self.mathbench_std_data_sets[name]

    def _generate_configs(self):
        mathbench_cfgs = []
        for _split in list(mathbench_sets.keys()):
            for _name in mathbench_sets[_split]:
                mathbench_cfgs.append(
                    dict(
                        abbr='mathbench-' + _split + '-' + _name,
                        path=f'./data/mathbench_v1/{_split}',
                        name=_name,
                        reader_cfg=dict(
                            input_column='question',
                            output_column='answer'
                        ),
                        meta_prompt=dict(
                            round=[
                                dict(
                                    role='HUMAN',
                                    prompt=single_choice_prompts[_name + '_with_reasoning'] if self.with_reasoning else
                                    single_choice_prompts[_name],
                                ), dict(role='BOT', prompt='{answer}')] if 'choice' in _name else (
                                cloze_prompts[_name + '_with_reasoning'] if self.with_reasoning else cloze_prompts[
                                    _name]),
                        ),
                    )
                )

        save_config(mathbench_cfgs[0])
        return mathbench_cfgs

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
            for id, line in enumerate(infile):
                entry = json.loads(line)
                if 'cloze' in cfg['name']:
                    data.append({
                        'question': entry['question'].strip(),
                        'answer': entry['answer'].strip()
                    })
                else:
                    question = entry['question'].strip(
                    ) + '\n' + get_number(entry['options'])
                    info = {
                        'question': question,
                        'answer': entry['answer'].strip()
                    }

                    data.append(info)
        return data

    def _load_data(self):

        mathbench_cfgs = self._generate_configs()
        save_config(mathbench_cfgs[0])

        mathbench_std_data_sets = {}
        if os.path.exists('./.cache') and self.cache:
            for cfg in mathbench_cfgs:
                std_subset = read_jsonl(os.path.join('./.cache', cfg["abbr"]) + '.jsonl')
                mathbench_std_data_sets[cfg["abbr"]] = std_subset
        else:
            if not os.path.exists('./.cache'):
                os.mkdir('./.cache')
            for cfg in mathbench_cfgs:
                info = self._generate_formal_info(cfg)
                std_subset = self._generate_std_subset(info, cfg)
                save_to_jsonl(std_subset, os.path.join('./.cache', cfg["abbr"]) + '.jsonl')
                mathbench_std_data_sets[cfg["abbr"]] = std_subset
        return mathbench_std_data_sets

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):

        return self.dataset[i]


def test():
    os.chdir('..')

    dataset = MathBenchDataset(1)

# test()
