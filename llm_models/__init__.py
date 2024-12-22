from openai import OpenAI
from google.api_core import retry
import json
import time
import requests
import google.generativeai as palm
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

logger = logging.getLogger(__name__)

API_BASE = "https://api.lingyiwanwu.com/v1"
palm_api_key = ''
claude_api_key = ""
max_tokens = 4096


def call_claude(prompt, model):
    prompt = prompt.replace('\\n', '\n')
    while True:

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

    logger.info(response.text)
    response = json.loads(response.text)["completion"]
    return response


def call_gpt(cur_sample, n_reasoning_paths,
             api_key="you_api_key",
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
            "temperature": 0.1,
            "top_p": 0.1,
            "seed": 1024,
            "n": n_reasoning_paths
        }
        response = requests.post("https://aigptx.top/v1/chat/completions", headers=headers, json=data)
        if response.status_code == 200:
            responses = response.json()
            answers = []
            n_input = []
            n_output = []
            for i in range(n_reasoning_paths):
                answers.append(responses['choices'][0]['message']['content'])
            n_input = responses['usage']['prompt_tokens']
            n_output = responses['usage']['completion_tokens']
            logger.info(responses['choices'][0]['message']['content'])

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
                    {'role': "system",
                     "content": "You are a helpful assistant. You need to answer the questions of the user accurately."},
                    {'role': "user", "content": cur_sample[0]['prompt']},
                ],

                temperature=0,
                n=n_reasoning_paths,
            )
            break
        except Exception as e:
            logger.info('Error!', e)
            time.sleep(random.uniform(1, 3))
    answers = []
    n_input = []
    n_output = []
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
        self.args = args
        if "bison" in args.model:
            palm.configure(api_key=palm_api_key)
        if 'bison' in args.model:
            self.model = call_palm
        elif 'claude' in args.model:
            self.model = call_claude
        elif 'gpt' in args.model:
            self.model = call_gpt
        elif 'yi' in args.model:
            self.model = call_Yi_Lightning
        else:
            raise NotImplementedError
        OPENAI_API_KEY = 'your_api_key'
        Yi_Lightning_API_KEY = "your_api_key"

    def query(self, sample, key='your_api_key'):
        return self.model(sample, n_reasoning_paths=self.args.n,
                          api_key=key, model=self.args.model)

    def query_multi(self, samples):
        keys = ['your_api_key', 'your_api_key']
        results = []
        with ThreadPoolExecutor(max_workers=len(keys)) as executor:

            futures = [
                executor.submit(self.model, samples[i], self.args.n, keys[i % len(keys)])
                for i in range(len(samples))
            ]

            for future in as_completed(futures):
                try:
                    result = future.result()
                except Exception as exc:
                    result = {"error": str(exc)}
                results.append(result)

        new_res = [item[0] for item in results]
        return new_res
