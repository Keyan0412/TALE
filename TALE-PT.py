from transformers import (
    AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments,
    TrainerCallback, TrainerState, TrainerControl
)
from peft import PeftModel, get_peft_model, LoraConfig, TaskType
from evaluator import AccEvaluator
from transformers import pipeline
import torch
from trl import DPOTrainer, DPOConfig
from utils import read_jsonl, save_to_jsonl, token_measure
import argparse
import time
from llm_models import LLMModel, inference_lora
import re
import os
from datasets import Dataset
import setproctitle
import logging
import json

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
proc_title = "your_proc_name"
setproctitle.setproctitle(proc_title)
MAX_LENGTH = MAX_NEW_Tokens = 512
SEED = 1024
TRAIN_EVAL_SPLIT = 1.0
EPOCH = 2

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    # filename='tmp/search_budget/gpt-4o-mini/log',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class LoggingCallback(TrainerCallback):
    def __init__(self, log_file):
        self.log_file = log_file

    def on_log(self, local_args, state: TrainerState, control: TrainerControl, **kwargs):
        save_to_jsonl(state.log_history, self.log_file)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=None,
                        help="=The path for base model(hugging face model)")
    parser.add_argument("--batch_size", default=64, type=int, help="=The budget token for our tech.")
    parser.add_argument("--lora_path", default=None, help="The path for lora model")
    parser.add_argument("--data_path", default=None, help="The path for loaded data")

    parser.add_argument("--output_dir", default=None,
                        help="The output directory")
    # hugging face model, DeepSeek-R1-Distill-Qwen-1.5B
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--strategy", default="lora", help="dpo, lora")
    parser.add_argument("--save", action='store_true', help="If we save the finetuned model.")
    parser.add_argument("--eval", action='store_true', help="If we just eval the model.")
    return parser.parse_args()


def prepare_data():
    def clean(text):
        # extract question
        return re.sub(r"Let's.*?:$", "", text).strip()

    data = read_jsonl(args.data_path)
    cleaned_data = [{
        "question": clean(item['question']),
        # "question": item['question'],
        "prediction": item['prediction_budget']
    } for item in data]

    def format_example(example):
        return {"input_text": f"{example['question']}",
                "target_text": f"{example['prediction']}"}

    # 80% for training, 20% for evaluation
    num_train = int(TRAIN_EVAL_SPLIT * len(cleaned_data))
    train_dataset = Dataset.from_list([format_example(d) for d in cleaned_data[:num_train]])
    eval_dataset = Dataset.from_list([format_example(d) for d in cleaned_data[num_train:]])
    return train_dataset, eval_dataset


def prepare_dpo_data():
    def clean(text):
        # extract question
        return re.sub(r"Let's.*?:$", "", text).strip()

    data = read_jsonl(args.data_path)
    cleaned_data = [{
        "prompt": clean(item['prompt']),
        "chosen": item['chosen'],
        "rejected": item['rejected']
    } for item in data]

    def format_example(example):
        return {"prompt": f"{example['prompt']}",
                "chosen": f"{example['chosen']}",
                "rejected": f"{example['rejected']}"}

    num_train = int(TRAIN_EVAL_SPLIT * len(cleaned_data))
    train_dataset = Dataset.from_list([format_example(d) for d in cleaned_data[:num_train]])
    eval_dataset = Dataset.from_list([format_example(d) for d in cleaned_data[num_train:]])
    return train_dataset, eval_dataset


def evaluate():
    def AvgLength(sample_list, gt_list, train=True):
        if train:
            logger.info("Evaluating on train set.")
        else:
            logger.info("Evaluating on test set.")
        logger.info(f"Data size: {len(sample_list)}")
        args.model = args.model_name
        if args.lora_path is None:
            logger.info("Evaluating base model!")
            args.n = 1  # Reasoning path
            model = LLMModel(args)
            pred_list = model.query_batch(sample_list)
        else:
            logger.info("Evaluating lora model!")
            lora_model, tokenizer = load_model(args.model_path, args.lora_path)
            pred_list = inference_lora(
                sample_list, n_reasoning_paths=1,
                model=lora_model, tokenizer=tokenizer, batch_size=args.batch_size
            )
        # dump to results
        assert len(sample_list) == len(gt_list) == len(pred_list)
        acc_num = 0
        total_length = 0
        results = []
        start_time = time.time()
        for i in range(len(pred_list)):
            results.append({
                "ground truth": gt_list[i],
                "question": sample_list[i],
                "prediction": pred_list[i],
            })
            total_length += token_measure(pred_list[i])
            if evaluator.evaluate_sample(results[-1], cloze=True):
                acc_num += 1
        logger.info("=" * 30 + 'Evaluation Results' + "=" * 30 + '\n')
        logger.info(f'Accuracy: {100 * (acc_num / len(results)):.2f}')
        logger.info(f"Average length: {total_length / acc_num:.2f}")
        # logger.info(f"Ground truth: {ground_truth}")
        # logger.info(f"Prediction: {pred[0][0]}")
        logger.info(f'Time cost: {time.time() - start_time}')
        return results


    data = read_jsonl(args.data_path)
    evaluator = AccEvaluator()
    all_sample_list = [
        item['question'].replace("Let's think step by step:\n", '') for item in data
        # item['question'] for item in data
    ]
    if args.lora_path is None:
        all_sample_list = [
            item['question'] + "\nPlease Give the response by strictly following " \
                               "this format: [[answer]], for example: Answer: [[" \
                               "50]]." for item in data
        ]
    all_gt_list = [item['ground truth'] for item in data]

    val_res = AvgLength(all_sample_list, all_gt_list, train=False)
    logger.info("=" * 30 + 'END' + "=" * 30 + '\n')

    logger.info(f"Saved to internalize-{args.strategy}.jsonl")
    save_to_jsonl(val_res, os.path.join(args.output_dir, f'internalize-{args.strategy}.jsonl'))


def evaluate_dpo():
    def AvgLength(sample_list, gt_list, train=True):
        if train:
            logger.info("Evaluating on train set.")
        else:
            logger.info("Evaluating on test set.")
        logger.info(f"Data size: {len(sample_list)}")
        args.model = args.model_name
        if args.lora_path is None:
            logger.info("Evaluating base model!")
            args.n = 1  # Reasoning path
            model = LLMModel(args)
            pred_list = model.query_batch(sample_list)
        else:
            logger.info("Evaluating DPO model with LoRA!")
            lora_model, tokenizer = load_model(args.model_path, args.lora_path)
            pred_list = inference_lora(
                sample_list, n_reasoning_paths=1,
                model=lora_model, tokenizer=tokenizer, batch_size=args.batch_size
            )
        # dump to results
        assert len(sample_list) == len(gt_list) == len(pred_list)
        acc_num = 0
        total_length = 0
        results = []
        start_time = time.time()
        for i in range(len(pred_list)):
            results.append({
                "ground truth": gt_list[i],
                "question": sample_list[i],
                "prediction": pred_list[i],
            })
            total_length += token_measure(pred_list[i])
            if evaluator.evaluate_sample(results[-1], cloze=True):
                acc_num += 1
        logger.info("=" * 30 + 'Evaluation Results' + "=" * 30 + '\n')
        logger.info(f'Accuracy: {acc_num / len(results)}')
        logger.info(f"Average length: {total_length / acc_num}")
        # logger.info(f"Ground truth: {ground_truth}")
        # logger.info(f"Prediction: {pred[0][0]}")
        logger.info(f'Time cost: {time.time() - start_time}')
        return results

    data = read_jsonl(args.data_path)
    evaluator = AccEvaluator()
    if args.lora_path is not None:
        all_sample_list = [
            item['prompt'].replace("Let's think step by step:\n", '') for item in data
        ]
    else:
        all_sample_list = [
            item['prompt'] for item in data
        ]
    all_gt_list = [item['ground truth'] for item in data]
    num_train = int(TRAIN_EVAL_SPLIT * len(all_sample_list))
    # train_sample_list, train_gt_list = all_sample_list[:num_train], all_gt_list[:num_train]
    val_sample_list, val_gt_list = all_sample_list[num_train:], all_gt_list[num_train:]
    val_res = AvgLength(val_sample_list, val_gt_list, train=False)
    logger.info("=" * 30 + 'END' + "=" * 30 + '\n')
    # train_res = AvgLength(train_sample_list, train_gt_list, train=True)
    # train_res.extend(val_res)
    logger.info(f"Saved to {args.output_dir}/internalize-{args.strategy}.jsonl")
    save_to_jsonl(val_res, os.path.join(args.output_dir, f'internalize-{args.strategy}.jsonl'))


def tokenize_data(dataset, tokenizer):
    def tokenize_function(examples):
        input_texts = examples["input_text"]
        output_texts = examples["target_text"]
        full_texts = [inp + "\n" + out for inp, out in zip(input_texts, output_texts)]
        tokenized = tokenizer(full_texts, padding="max_length", truncation=True, max_length=2048)
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset


def load_model(model_path, lora_path=None):
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if args.model_name in ['Qwen2.5-14B', 'Qwen2.5-7B-Instruct-1M']:
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
    base_model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
    base_model.eval()
    logger.info(f"Load base model from {model_path}")
    logger.info(f"Tokenizer max length: {tokenizer.model_max_length}")
    logger.info(f"Model config max length: {base_model.config.max_position_embeddings}")

    if lora_path is None:
        return base_model, tokenizer
    lora_model = PeftModel.from_pretrained(base_model, lora_path)
    logger.info(f"Load LoRA model from {lora_path} Successfully!")
    merged_model = lora_model.merge_and_unload()
    assert not isinstance(merged_model, PeftModel), "merge_and_unload failed"
    merged_model.half()
    merged_model.eval()
    return merged_model, tokenizer


def train_model_with_lora(model, tokenizer, dataset):

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"]
    )
    model = get_peft_model(model, lora_config)

    tokenized_dataset = tokenize_data(dataset, tokenizer)
    os.makedirs(os.path.join(args.output_dir, 'logs'), exist_ok=True)
    training_args = TrainingArguments(
        log_level="info",
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=8,
        num_train_epochs=EPOCH,
        save_steps=1,
        save_strategy="steps",
        save_total_limit=100,
        eval_strategy="no",
        logging_dir=os.path.join(args.output_dir, 'logs'),
        logging_steps=1,
        learning_rate=1e-4,
        weight_decay=0.01,
        fp16=True
    )
    with open(os.path.join(args.output_dir, 'logs', 'train-args.jsonl'), "w") as f:
        json_line = json.dumps(training_args.to_dict())
        f.write(json_line + "\n")
    log_file = os.path.join(args.output_dir, 'logs', 'log.jsonl')
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        callbacks=[LoggingCallback(log_file)],
        processing_class=tokenizer,
    )
    logger.info("Lora initialized!")
    trainer.train()
    # import sys
    # sys.exit(-1)
    # logger.info(type(trainer.state.log_history))
    save_to_jsonl(trainer.state.log_history, log_file)
    if args.save:
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)


def train_model_with_dpo(model, tokenizer, dataset):
    os.makedirs(os.path.join(args.output_dir, 'logs'), exist_ok=True)
    logger.info(f"Saving to {args.output_dir}")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"]
    )
    model = get_peft_model(model, lora_config)
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         logger.info(f"Updated params, {name}: {param.shape}")

    training_args = DPOConfig(
        output_dir=args.output_dir,
        log_level="info",
        logging_dir=os.path.join(args.output_dir, 'logs'),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=16,
        num_train_epochs=EPOCH,
        learning_rate=3e-5,
        logging_steps=1,
        save_strategy="steps",
        save_steps=1,
        report_to="none",
        fp16=False,
        save_total_limit=200,
        weight_decay=0.001,
        max_length=MAX_LENGTH,
        seed=SEED,
        data_seed=SEED,
        logging_first_step=True,
        beta=0.5,
        max_grad_norm=5.0
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        ref_model=None,
        train_dataset=dataset,
        eval_dataset=None,
    )
    trainer.train()
    logger.info(trainer.state.log_history)
    save_to_jsonl(trainer.state.log_history, os.path.join(args.output_dir, 'logs', 'log.jsonl'))
    if args.save:
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)


def inference(model_path, tokenizer_path, question):

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    prompt = f"Let’s think step by step:\n{question}"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    output = model.generate(input_ids, max_length=200, do_sample=True, temperature=0.7)
    return tokenizer.decode(output[0], skip_special_tokens=True)


def inference_pipeline():
    data = read_jsonl('temp/craft-data-GSM8K-Zero/cleansed_data.jsonl')
    logger.info(data[0].keys())
    logger.info(data[0]['question'])
    logger.info(data[0]['budget'])
    logger.info(token_measure(data[0]['prediction']))
    logger.info(token_measure(data[0]['prediction_budget']))
    question = data[0]['question'].replace("Let's think step by step:", "")
    # gt = data[0]['ground truth']
    messages = [
        {"role": "system",
         "content": "You are a helpful assistant."
         },
        {"role": "user", "content": f"""{question}\nLet's think step by step and use less than 250 tokens:\n"""},
    ]
    pipe = pipeline("text-generation", model=".cache/DeepSeek-R1-Distill-Qwen-1.5B",
                    torch_dtype=torch.float16, device_map="auto")
    output = pipe(
        messages,
        max_new_tokens=5000,
        temperature=0.1,
        top_p=0.1,
        repetition_penalty=1.2,
        do_sample=True,
        num_return_sequences=1,
    )
    logger.info(output[0])
    answer = output[0]['generated_text'][-1]['content']
    from evaluator import AccEvaluator
    evaluator = AccEvaluator()
    logger.info("The answer is: ", evaluator.extract_predicted_answer(answer))
    logger.info(f"Token costs is: {token_measure(answer)}")


def inference_pipeline_batch():
    texts = [
        "I love this movie!",
        "This book is terrible.",
        "The food was great!",
        "I don't like the weather today."
    ]
    pipe = pipeline("text-generation", model=".cache/DeepSeek-R1-Distill-Qwen-1.5B",
                    torch_dtype=torch.float16, device_map="auto")
    output = pipe(
        texts,
        max_new_tokens=5000,
        temperature=0.1,
        top_p=0.1,
        repetition_penalty=1.2,
        do_sample=True,
        num_return_sequences=1,
        batch_size=4
    )
    logger.info(len(output))


def main():
    args.model_path = os.path.join('.cache', args.model_name)
    if args.eval:
        if args.strategy == "lora":
            args.output_dir = os.path.join(args.output_dir, args.model_name, "LoRA")
            evaluate()
        else:
            args.output_dir = os.path.join(args.output_dir, args.model_name, "DPO")
            evaluate_dpo()
    elif args.strategy == "lora":
        args.output_dir = os.path.join(args.output_dir, args.model_name, "LoRA")
        if args.save:
            logger.info(f"Saving to {args.output_dir}")
        # inference_pipeline() # just for test
        # inference_pipeline_batch()
        dataset, _ = prepare_data()
        logger.info("Data Prepared Successfully!")
        model, tokenizer = load_model(args.model_path, args.lora_path)
        logger.info("Model Loaded Successfully!")
        train_model_with_lora(model, tokenizer, dataset)
    elif args.strategy == "dpo":
        args.output_dir = os.path.join(args.output_dir, args.model_name, "DPO")
        dataset, _ = prepare_dpo_data()
        logger.info("Data Prepared Successfully!")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(args.model_path, local_files_only=True)
        train_model_with_dpo(model, tokenizer, dataset)

    # logger.info(inference(output_dir, output_dir, test_question))


if __name__ == "__main__":
    args = parse_args()
    main()
