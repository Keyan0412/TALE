import os
import random
import argparse
from utils import *
from torch.utils.data import Subset
from llm_datasets import *
from llm_models import LLMModel
import time
import logging
from evaluator import AccEvaluator

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


# TODO: add more number of generated responses?


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", default=None, help="=The budget token for our tech.")
    parser.add_argument("--reasoning", action='store_true', help="If we use LLM reasoning.")
    parser.add_argument("--model", default='gpt-4o-mini', help="The model name on huggingface.")
    # gpt-4, gpt-4o-2024-05-13, GPT-3.5-turbo-0613, gpt-4o-mini, yi-lightning
    parser.add_argument("--output_path", default='./temp/100-test',
                        help="The output path to save the model output.")
    parser.add_argument("--n", default=1, type=int, help="Number of samples from LLM.")
    parser.add_argument("--start_index", default=0, type=int, help="The start index for the dataset.")
    parser.add_argument("--end_index", default=100, type=int, help="The end index for the dataset.")
    parser.add_argument("--key_index", default=1, type=int, help="The key index for the dataset.")
    # 'mathbench-college-single_choice_en, GSM8K, GPQA, GSM8K-Zero'
    parser.add_argument("--data_name", default='GSM8K',
                        type=str, help="The dataset name used during our evaluation.")
    args = parser.parse_args()
    args.reasoning = True
    # args.output_path = os.path.join(args.output_path, args.model, 'with_budget' if args.budget else 'without_reasoning')
    args.output_path = os.path.join(args.output_path, args.model, args.data_name)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    if args.budget is not None:
        # args.output_path = os.path.join(args.output_path, args.model, 'output_with_budget.jsonl')
        args.output_path = os.path.join(args.output_path, 'output_with_budget.jsonl')
    else:
        args.output_path = os.path.join(args.output_path,
                                        'output_with_reasoning.jsonl' if args.reasoning else 'output_without_reasoning_new_prompt.jsonl')
    logger.info(f'Saving to {args.output_path}')
    keys = {
        'yi-lightning': ['your_api_key', 'your_api_key'],
        'gpt-4o-mini': ['your_api_key', 'your_api_key'],
        'gpt-4o-2024-05-13': ['your_api_key', 'your_api_key'],
    }
    key = keys[args.model][args.key_index]
    # Prepare dataset
    if 'math' in args.data_name:
        if args.data_name == 'math':
            dataset = MathBenchDataset(args, with_reasoning=args.reasoning, budget=args.budget, cache=False)
        else:
            dataset = MathBenchDataset(args, with_reasoning=args.reasoning, budget=args.budget,
                                       name=args.data_name, cache=False)
    elif args.data_name == 'GSM8K-Zero':
        dataset = GSM8KZero(args, with_reasoning=args.reasoning, budget=args.budget,
                            name=args.data_name, cache=False)
    elif args.data_name == 'GSM8K':
        dataset = GSM8K(args, with_reasoning=args.reasoning, budget=args.budget,
                        name=args.data_name, cache=False)
    else:
        dataset = None
        ValueError(f"Not supported for {args.data_name}")
    # dataset = Subset(dataset, list(range(1500, 2000)))
    # Prepare evaluator
    evaluator = AccEvaluator(dataset)
    # Prepare llm model
    model = LLMModel(args)
    acc_num = 0
    results = []
    start_time = time.time()
    logger.info("=" * 30 + 'Requesting' + "=" * 30 + '\n')
    if args.end_index is None:
        args.end_index = len(dataset)
    for idx, instance in enumerate(dataset):
        # logger.info(idx)
        if args.start_index <= idx < args.end_index:
            cur_sample = instance['round']
            ground_truth = instance['gold']
            logger.info('=' * 30 + f"Step: {idx + 1} / {args.end_index}" + '=' * 30)
            # pred = model.query(cur_sample, key=keys[args.key_index])
            logger.info(f"Question: {cur_sample[0]['prompt']}")
            pred = model.query(cur_sample, key=key)
            results.append({
                "ground truth": ground_truth,
                "question": cur_sample[0]['prompt'],
                "prediction": pred[0][0],
            })
            acc_num += evaluator.evaluate_sample(results[-1],
                                                 cloze=('cloze' in args.data_name) or (args.data_name in ['GSM8K', 'GSM8K-Zero']))
            logger.info(f'Accuracy: {acc_num / len(results)}')
            # logger.info(f"Ground truth: {ground_truth}")
            # logger.info(f"Prediction: {pred[0][0]}")
            save_to_jsonl(results, args.output_path)
            logger.info(f'Time cost: {time.time() - start_time}')


if __name__ == "__main__":
    main()
