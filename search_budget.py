#!/usr/bin/env python3
import os
import argparse
from utils import *
from llm_datasets import *
from llm_models import LLMModel
import logging
from evaluator import AccEvaluator
from settings import OPENAI_API_KEY
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    """
    Parse command line arguments for the budget search script.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", default=None, help="=The budget token for our tech.")
    parser.add_argument("--do_search", action='store_true', help="If we search the best budget.")
    parser.add_argument("--model", default='gpt-4o-mini', help="yi-lightning, gpt-4o-mini")
    parser.add_argument("--output_path", default='tmp/search_budget/gpt-4o-mini',
                        help="The output path to save the model output.")
    parser.add_argument("--n", default=1, type=int, help="Number of samples from LLM.")
    parser.add_argument("--start_index", default=0, type=int, help="The start index for the dataset.")
    parser.add_argument("--end_index", default=None, type=int, help="The end index for the dataset.")
    parser.add_argument("--key_index", default=0, type=int, help="The key index for the dataset.")
    parser.add_argument("--data_name", default='GSM8K',
                        type=str, help="The dataset name used during our evaluation.")
    return parser.parse_args()


def search_budget(instance, budget, model, evaluator, key='your_api_key'):
    """
    Search for optimal token budget that maintains prediction accuracy while minimizing tokens.
    
    Args:
        instance: Dictionary containing question and ground truth
        budget: Initial token budget (upper bound)
        model: LLM model instance
        evaluator: AccEvaluator instance for accuracy checking
        key: API key for model access (default: 'your_api_key')
        
    Returns:
        tuple: (updated_instance, final_budget) where:
            updated_instance: Original instance with added budget information
            final_budget: The optimal budget found
    """
    pred_flag = evaluator.evaluate_sample(instance)
    if not pred_flag:
        logger.info("Incorrect prediction, skipping...")
        return instance, budget
    upper_bound = budget
    pre_token_cost = upper_bound
    instance['question_budget'] = 'None'
    instance['prediction_budget'] = 'None'
    instance['budget'] = upper_bound
    res_budget_list = [upper_bound]
    res_token_list = []
    while pred_flag:
        new_question = add_budget(instance['question'], budget // 2)
        cur_sample = [{'prompt': new_question}]
        cur_answer = model.query(cur_sample, key=key)[0][0]
        cur_token_cost = token_measure(cur_answer)
        res_token_list.append(cur_token_cost)
        pred_flag = evaluator.evaluate_sample({
            'ground truth': instance['ground truth'],
            'prediction': cur_answer
        })
        # condition for the next iteration
        if pred_flag and cur_token_cost < pre_token_cost:
            # update current best answer and budget
            logger.info(f'Searching budget from {budget} to {budget // 2}.')
            logger.info(f'Token costs from {pre_token_cost} to {cur_token_cost}')
            instance['question_budget'] = new_question
            instance['prediction_budget'] = cur_answer
            instance['budget'] = budget // 2
            budget //= 2
            pre_token_cost = cur_token_cost
            res_budget_list.append(budget)
        else:
            break
    if not pred_flag:
        logger.info("Incorrect prediction, ending search...")
    if cur_token_cost > pre_token_cost:
        logger.info("Token cost increased, ending search...")
    instance['budget_list'] = res_budget_list
    instance['token_list'] = res_token_list
    return instance, budget


def main():
    args = parse_args()
    args.do_search = True
    args.output_path = os.path.join(args.output_path, 'searched_budget.jsonl')
    logger.info(f'Saving to {args.output_path}')
    
    dataset = read_jsonl("./tmp/gpt-4o-mini/GSM8K-Test/output_with_reasoning_nb.jsonl")
    model = LLMModel(args)
    keys = {
        'yi-lightning': [OPENAI_API_KEY],
        'gpt-4o-mini': [OPENAI_API_KEY],
        'gpt-4o-2024-05-13': [OPENAI_API_KEY],
    }
    key = keys[args.model][args.key_index]
    res_budget = []
    logger.info("start searching budget...")
    if args.end_index is None:
        args.end_index = len(dataset)

    # search for the best budget
    evaluator = AccEvaluator(dataset)
    idx = args.start_index
    if args.do_search:
        def search_process_instance(instance, model, evaluator, key):
            nonlocal res_budget
            token_cost = token_measure(instance['prediction'])
            new_instance, budget = search_budget(instance, token_cost, model, evaluator, key=key)
            new_token_cost = token_measure(new_instance['prediction_budget'])
            logger.info(f"Updating Budget: {budget}.")
            logger.info(f"Updating Token costs: {token_cost} ==> {new_token_cost}.")
            with threading.Lock():
                res_budget.append(new_instance)

        with ThreadPoolExecutor(max_workers=32) as executor:
            futures = []
            while idx < args.end_index:
                instance = dataset[idx]
                logger.info('=' * 30 + f"Step: {idx - args.start_index + 1} / {args.end_index-args.start_index}" + '=' * 30)

                # check correctness of prediction
                if not evaluator.evaluate_sample(instance): 
                    logger.info("Incorrect prediction, skipping...")
                    idx += 1
                    continue

                future = executor.submit(search_process_instance, instance, model, evaluator, key)
                futures.append(future)
                idx += 1

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    logger.error(f"处理实例时发生错误: {exc}")
            save_to_jsonl(res_budget, args.output_path)

    # simply reduce the token cost by half
    else:  
        while idx < args.end_index:
            instance = dataset[idx]
            logger.info('=' * 30 + f"Step: {idx - args.start_index + 1} / {args.end_index-args.start_index}" + '=' * 30)

            # check correctness of prediction
            if not evaluator.evaluate_sample(instance): 
                logger.info("Incorrect prediction, skipping...")
                idx += 1
                continue

            token_cost = token_measure(instance['prediction'])
            new_question = add_budget(instance['question'], token_cost // 2)  # add budget to the question
            cur_sample = [{'prompt': new_question}]
            cur_answer = model.query(cur_sample, key=key)[0][0]
            cur_token_cost = token_measure(cur_answer)

            # check prediction and record the instance
            if evaluator.evaluate_sample({
                'ground truth': instance['ground truth'],
                'prediction': cur_answer
            }):
                instance['question_budget'] = new_question
                instance['prediction_budget'] = cur_answer
                instance['budget'] = token_cost // 2
                logger.info("Updating Token costs: {}/{}.".format(cur_token_cost, token_cost))
                res_budget.append(instance)
            idx += 1
        save_to_jsonl(res_budget, args.output_path)


if __name__ == "__main__":
    main()
