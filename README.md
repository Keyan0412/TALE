# <center> README </center>

## 1. Overview

This is the official repo for our in-progress work, “Token-Budget-Aware LLM Reasoning”.

Reasoning is crucial for LLMs to perform complex tasks, but methods like Chain-of-Thought (CoT) reasoning often lead to significant token overhead and increased costs. We identify substantial token redundancy in the reasoning process of state-of-the-art LLMs and propose a token-budget-aware reasoning framework. This approach dynamically allocates token budgets based on problem complexity to guide the reasoning process. Experiments demonstrate that our method reduces token usage in CoT reasoning with minimal performance trade-offs, striking a practical balance between efficiency and accuracy.

<img src="images%20in%20text/image-20241222191739762.png" alt="image-20241222191739762" width="30%“>





## 2. Environment

Please see requirements.txt.



## Inference for `Directly Answering` and `Vanilla CoT`

### `Directly Answering`

```sh
python -u inference.py --data_name GSM8K-Zero --model gpt-4o-mini 
```



### `Vanilla CoT`

```sh
python -u inference.py --data_name GSM8K-Zero --model gpt-4o-mini --reasoning
```



### Output token costs between Directly Answering and Vanilla CoT

<img src="images%20in%20text/Figure_1-1734865063632-4.png" alt="Figure_1" style="zoom:50%;" />



## 🧰 Search for optimal budget

```sh
python -u search_budget.py --do_search --data_name GSM8K-Zero
```

#### Output token costs between Vanilla CoT and CoT with optimal searched budget

<img src="images%20in%20text/Figure_1-1734865200621-6.png" alt="Figure_1" style="zoom:50%;" />



## ⚙ TALE

We have introduced three different budget estimation methods in our paper.

TALE with Zero-shot Estimator:

```sh
python -u TALE.py --data_name GSM8K-Zero --model gpt-4o-mini
```

TALE with Regression Estimator and Token-Budget Awareness Internalization via Fine-tuning is on the way!



## 📃 Note

This project is in progress, and the following implementation is coming soon!

