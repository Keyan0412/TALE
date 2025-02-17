from openai import OpenAI
from google.api_core import retry
import json
import time
import torch
import requests
import google.generativeai as palm
from tqdm import tqdm
from utils import *
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

logger = logging.getLogger(__name__)

API_BASE = "https://api.lingyiwanwu.com/v1"
palm_api_key = ''
claude_api_key = ""
max_tokens = 5000


def inference_lora(sample_list, n_reasoning_paths, model, tokenizer, batch_size):
    logger.info("Inference processing...")
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer,
                    torch_dtype=torch.float16,
                    device_map="auto", batch_size=batch_size)
    system_prompt = {"role": "system", "content": "You are a helpful assistant."}
    messages = [[system_prompt, {"role": "user", "content": sample_list[i]}]
                for i in range(len(sample_list))]

    # logger.info(cur_sample[0]['prompt'])
    output = []
    for i in tqdm(range(0, len(sample_list), batch_size), desc="Processing", unit="batch"):
        batch = messages[i:i + batch_size]
        output.extend(pipe(batch,
                           max_new_tokens=1024,
                           temperature=0.1,
                           repetition_penalty=1.2,
                           do_sample=True,
                           num_return_sequences=n_reasoning_paths))
        logger.info(messages[i])
        logger.info(output[i][0]['generated_text'][-1]['content'])
        logger.info(f"Token costs is {token_measure(output[i][0]['generated_text'][-1]['content'])}")
    assert len(output) == len(sample_list)
    results = [output[i][0]['generated_text'][-1]['content'] for i in range(len(output))]
    return results


class HuggingFaceModel:
    def __init__(self, args):
        self.args = args
        self.model_path = f'.cache/{self.args.model}'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, torch_dtype=torch.float16,
                                                       local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, torch_dtype=torch.float16,
                                                          local_files_only=True)
        if args.model in ['Qwen2.5-14B', 'Qwen2.5-7B-Instruct-1M']:
            self.tokenizer.padding_side = "left"
            self.tokenizer.pad_token = self.tokenizer.eos_token if self.tokenizer.pad_token is None else self.tokenizer.pad_token
        self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer,
                             torch_dtype=torch.float16,
                             device_map="auto", batch_size=self.args.batch_size)
        logger.info("Model loaded!")

    def inference_batch(self, sample_list, n_reasoning_paths):
        logger.info("Inference processing...")
        system_prompt = {"role": "system", "content": "You are a helpful assistant."}
        messages = [[system_prompt, {"role": "user", "content": sample_list[i]}] for i in range(len(sample_list))]

        # logger.info(cur_sample[0]['prompt'])
        results = []
        for i in tqdm(range(0, len(sample_list), self.args.batch_size), desc="Processing", unit="batch"):
            batch = messages[i:i + self.args.batch_size]
            outputs = self.pipe(
                batch, max_new_tokens=max_tokens,
                temperature=0.1,
                repetition_penalty=1.2,
                do_sample=True,
                num_return_sequences=n_reasoning_paths)
            results.extend([output[0]['generated_text'][-1]['content'] for output in outputs])
            logger.info(messages[i])
            logger.info(results[-1])
        assert len(results) == len(sample_list)
        # results = [output[i][0]['generated_text'][-1]['content'] for i in range(len(output))]
        return results

    def inference_batch_new(self, sample_list, n_reasoning_paths):
        logger.info("Inference processing...")
        system_prompt = {"role": "system", "content": "You are a helpful assistant."}
        messages = [[system_prompt, {"role": "user", "content": sample}] for sample in sample_list]
        tokenized_data = []
        for message in messages:
            formatted_message = self.format_message(message)
            encoded_input = self.tokenizer(
                formatted_message,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=512
            )
            tokenized_data.append(encoded_input)

        results = []

        for i in tqdm(range(0, len(messages), self.args.batch_size), desc="Processing Batches", unit="batch"):
            batch = tokenized_data[i:i + self.args.batch_size]
            batch_inputs = {k: torch.cat([b[k] for b in batch], dim=0).to("cuda") for k in batch[0].keys()}
            with torch.no_grad():
                output = self.model.generate(
                    **batch_inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.1,
                    repetition_penalty=1.2,
                    num_return_sequences=n_reasoning_paths,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            decoded_outputs = [self.tokenizer.decode(o, skip_special_tokens=True) for o in output]
            results.extend(decoded_outputs)
            logger.info(decoded_outputs)

        assert len(results) == len(sample_list)
        return results

    @staticmethod
    def format_message(message):
        formatted_message = ""
        for msg in message:
            role = msg["role"]
            content = msg["content"]
            formatted_message += f"{role}: {content}\n"
        return formatted_message.strip()

    def inference(self, cur_sample, n_reasoning_paths):
        logger.info("Inference processing...")
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": cur_sample[0]['prompt']},
        ]
        # logger.info(cur_sample[0]['prompt'])
        output = self.pipe(
            messages,
            max_new_tokens=max_tokens,
            temperature=0.1,
            top_p=0.1,
            repetition_penalty=1.2,
            do_sample=True,
            num_return_sequences=n_reasoning_paths
        )

        n_input, n_output = [], []
        return [output[0]['generated_text'][-1]['content']], n_input, n_output


def call_claude(prompt, model):
    prompt = prompt.replace('\\n', '\n')
    while True:
        #    try:
        data = {
            "max_tokens_to_sample": max_tokens,
            "model": model,
            "prompt": prompt,
            'temperature': 0,
        }
        response = requests.post(
            url="https://api.anthropic.com/v1/complete",
            headers={
                'accept': 'application/json',
                'anthropic-version': '2023-06-01',
                'content-type': 'application/json',
                'x-api-key': claude_api_key,
            },
            json=data,
        )
        if response.status_code == 200:
            break
    response = json.loads(response.text)["completion"]
    return response


def call_gpt(cur_sample, n_reasoning_paths,
             api_key="your_api_key",
             model='gpt-4o-mini'):
    while True:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        data = {
            "model": model,
            "messages": [
                {"role": "system",
                 "content": "You are a helpful assistant."
                 },
                {"role": "user",
                 "content": cur_sample[0]['prompt']
                 }
            ],
            "max_tokens": max_tokens,
            "seed": 1024,
            "n": n_reasoning_paths
        }

        response = requests.post("https://aigptx.top/v1/chat/completions", headers=headers, json=data)
        if response.status_code == 200:
            responses = response.json()
            answers = []
            for i in range(n_reasoning_paths):
                answers.append(responses['choices'][0]['message']['content'])
            n_input = responses['usage']['prompt_tokens']
            n_output = responses['usage']['completion_tokens']
            return answers, n_input, n_output
        else:
            logger.info(f"Error: {response.status_code} - {response.text}")
            time.sleep(random.uniform(1, 3))


def call_Yi_Lightning(cur_sample, n_reasoning_paths,
                      api_key='your_api_key',
                      model='yi-lightning'):
    client = OpenAI(
        api_key=api_key,
        base_url=API_BASE
    )
    while True:
        try:
            responses = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=[
                    {'role': "system", "content": "You are a helpful assistant."},
                    {'role': "user", "content": cur_sample[0]['prompt']},
                ],
                # stop = stop,
                temperature=0,
                n=n_reasoning_paths,
            )
            break
        except Exception as e:
            logger.info('Error!', e)
            time.sleep(random.uniform(1, 3))
    answers = []
    for i in range(n_reasoning_paths):
        answers.append(responses.choices[i].message.content)
    n_input = responses.usage.prompt_tokens
    n_output = responses.usage.completion_tokens
    return answers, n_input, n_output


@retry.Retry()
def call_palm(prompt, candidate_count, model):
    completion = palm.generate_text(
        model=model,
        prompt=prompt,
        temperature=0,
        candidate_count=candidate_count,
        max_output_tokens=512,
    )
    answers = [c['output'] for c in completion.candidates]
    return answers


class LLMModel:
    def __init__(self, args):
        self.local = False
        self.args = args
        if "bison" in args.model:
            palm.configure(api_key=palm_api_key)
        if 'bison' in args.model:  # Google models
            self.model = call_palm
        elif 'claude' in args.model:
            self.model = call_claude
        elif 'gpt' in args.model or 'o3' in args.model:  # OpenAI models
            self.model = call_gpt
        elif 'yi' in args.model:
            self.model = call_Yi_Lightning
        elif 'Qwen' in args.model or 'DeepSeek' in args.model or 'Llama' in args.model:
            self.model = HuggingFaceModel(args=args)
            self.local = True
        else:
            raise NotImplementedError

    def query_batch(self, sample_list):
        # return self.model.inference_batch_new(sample_list, self.args.n)
        return self.model.inference_batch(sample_list, self.args.n)

    def query(self, sample, key='your_api_key'):
        if self.local:
            return self.model.inference(sample, n_reasoning_paths=self.args.n)
        else:
            return self.model(sample, n_reasoning_paths=self.args.n,
                              api_key=key, model=self.args.model)

