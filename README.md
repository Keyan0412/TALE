# <center> README </center>

## 🚀1. Overview

This is the script of our paper, [“Token-Budget-Aware LLM Reasoning”.](https://arxiv.org/pdf/2412.18547)

Reasoning is crucial for LLMs to perform complex tasks, but methods like Chain-of-Thought (CoT) reasoning often lead to significant token overhead and increased costs. We identify substantial token redundancy in the reasoning process of state-of-the-art LLMs and propose a token-budget-aware reasoning framework. This approach dynamically allocates token budgets based on problem complexity to guide the reasoning process. Experiments demonstrate that our method reduces token usage in CoT reasoning with minimal performance trade-offs, striking a practical balance between efficiency and accuracy.



## 📖 2. Environment

Please see requirements.txt.



## 🏗️3. Inference 

We provide the implementation for Directly Answering and Vanilla CoT.

### ⚡Directly Answering

```sh
python -u inference.py --data_name GSM8K-Zero --model gpt-4o-mini 
```

### 🔗Vanilla CoT

```sh
python -u inference.py --data_name GSM8K-Zero --model gpt-4o-mini --reasoning
```



### 💰Output token costs 

The output token costs between Directly Answering and Vanilla CoT are illustrated as follows:

<img src="images%20in%20text/Figure_1-1734865063632-4.png" width="50%">



## 🔍4. Search for optimal budget

```sh
python -u search_budget.py --do_search --data_name GSM8K-Zero
```

#### 💰Output token costs

The output token costs between Vanilla CoT and CoT with optimal searched budget are illustrated as follows:

<img src="images%20in%20text/Figure_1-1734865200621-6.png" width="50%">



## ⚙️5. TALE

We provide two implementations of TALE, TALE-EP and TALE-PT.

### 🧠TALE-EP

TALE with Zero-shot Estimator: 

```sh
python -u TALE-EP.py --data_name GSM8K-Zero --model gpt-4o-mini
```

### 🎯TALE-PT

#### 📚 TALE-PT-SFT

<img src="images%20in%20text/image-20250216125709485.png" width="67%">

```sh
# for training
python -u TALE-PT.py --strategy lora --model_name llama-3.1-8B-Instruct --data_path <your_training_data_path> --output_dir <your_output_dir> --batch_size 2 --save

# for eval
python -u TALE-PT.py --eval --strategy lora --model_name llama-3.1-8B-Instruct --data_path <your_eval_data_path> --output_dir <your_output_dir> --batch_size 2 --save
```

#### 🔄TALE-PT-DPO

<img src="images%20in%20text/image-20250216125725621.png" width="67%">

```sh
# for training
python -u TALE-PT.py--strategy dpo --model_name llama-3.1-8B-Instruct --data_path <your_training_data_path> --output_dir <your_output_dir> --batch_size 2 --save

# for eval
python -u TALE-PT.py --eval --strategy dpo --model_name llama-3.1-8B-Instruct --data_path <your_eval_data_path> --output_dir <your_output_dir> --batch_size 2 --save
```



## 🤝6. Cite our work

```tex
@article{han2024token,  
  title={Token-Budget-Aware LLM Reasoning},  
  author={Han, Tingxu and Wang, Zhenting and Fang, Chunrong and Zhao, Shiyu and Ma, Shiqing and Chen, Zhenyu},  
  journal={arXiv preprint arXiv:2412.18547},  
  year={2024}  
}
```

