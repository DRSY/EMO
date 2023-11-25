<p align="center">
<img src="./logo.png" width="250" height="250">
</p>

# EMO
<a href="https://arxiv.org/abs/2310.04691">
<img alt="Static Badge" src="https://img.shields.io/badge/paper-arxiv-red?link=https%3A%2F%2Farxiv.org%2Fabs%2F2310.04691">
</a>
<a href="https://pypi.org/project/EMOLoss/0.0.1/">
<img alt="Static Badge" src="https://img.shields.io/badge/pip-pypi-blue?link=https%3A%2F%2Fpypi.org%2Fproject%2FEMOLoss%2F0.0.1%2F">
</a>


This is the public codebase for arxiv paper: [EMO: Earth Mover Distance Optimization for Auto-regressive Language Modeling](https://arxiv.org/abs/2310.04691).

## TABLE OF CONTENTS
- [EMO](#emo)
  - [TABLE OF CONTENTS](#table-of-contents)
  - [Abstract](#abstract)
- [Usage](#usage)
  - [Standalone Package](#standalone-package)
  - [Use EMO as an independent loss function](#use-emo-as-an-independent-loss-function)
  - [Use EMO as a patch to existing models](#use-emo-as-a-patch-to-existing-models)
  - [Setup](#setup)
  - [Code Structure](#code-structure)
  - [🏫 Language Modeling Experiments](#-language-modeling-experiments)
  - [📑 NLU Experiments](#-nlu-experiments)
    - [Run continual fine-tuning on WikiText-103](#run-continual-fine-tuning-on-wikitext-103)
    - [Run downstream tasks using few-shot in-context learning](#run-downstream-tasks-using-few-shot-in-context-learning)
  - [📚 Instruction-Tuning](#-instruction-tuning)
- [🌐 Acknowledgements](#-acknowledgements)
- [Citation](#citation)

## Abstract
Neural language models are probabilistic models of human text. They are predominantly trained using maximum likelihood estimation (MLE), which is equivalent
to minimizing the forward cross-entropy between the empirical data distribution and the model distribution. However, various degeneration phenomena are still
widely observed when decoding from the distributions learned by such models. We establish that the forward cross-entropy is suboptimal as a distance metric for aligning human and model distribution due to its (1) recall-prioritization (2) negative diversity ignorance and (3) train-test mismatch. In this paper, we propose Earth Mover Distance Optimization (EMO) for auto-regressive language modeling. EMO capitalizes on the inherent properties of earth mover distance to address the aforementioned challenges. Due to the high complexity of direct computation, we further introduce a feasible upper bound for EMO to ease end-to-end training. Upon extensive evaluation of language models trained using EMO and MLE. We find that EMO demonstrates a consistently better language modeling performance than MLE across domains. Moreover, EMO demonstrates noteworthy enhancements in downstream performance with minimal fine-tuning on merely 25,000 sentences. This highlights the tremendous potential of EMO as a lightweight calibration method for enhancing large-scale pre-trained language models.

# Usage
## Standalone Package
We provide PyPi package of EMO as a easy-to-use loss function. Before install EMO, make sure you have installed `torch`.
```bash
pip install EMOLoss==0.0.3
```
## Use EMO as an independent loss function
EMO requires three input fields, namely `logits`, `labels`, and `cost_embedding`. 
```python
import torch
from emo import EMOLoss
logits = torch.rand(32, 1024, 32000, requires_grad=True)
labels = torch.ones(32, 1024, dtype=torch.long)
cost_embedding = torch.rand(32000, 4096)
emo_loss = EMOLoss(logits, labels, cost_embedding, ignore_index=-100, mode=1)
```
> **Signature of EMOLoss**
> - `logits` (Tensor, requried): the output logits after lm_head, before applying softmax
> - `labels` (Tensor, required): ids of ground truth next token
> - `cost_embedding` (Tensor, required) can be a fixed matrix extracted from a pre-trained model(more suitable for lightweight continual pre-training on general corpus), or can be the lm_head of the model currently being trained to inject task/domain-specific information(more suitable for domain-specific adaptation or instruction tuning). 
> - `ignore_index` (Int, Optional): default to -100.
> - `mode` (Int, Optional): loss weighting mode. 1 means that MLE is emphasized(more suitable for domain-specific adaptation) and 2 means that EMO is emphasized(more suitable for lightweight continual pre-training).

The `cost_embedding` must share the same vocabulary size as `logits`, e.g., 32000 for LLaMa. However, the hidden size of `cost_embedding` is not required to be identical to the model you want to train.
## Use EMO as a patch to existing models
EMO can also be integrated into HuggingFace's `transformers` using `emo_patch.py`. Below is an example of replacing the original forward function of `transformers.LlamaForCausalLM` with EMO:
```python
from transformers import LlamaForCausalLM
from emo_patch import (
  replace_llama_forward_with_emo_1_adaptive_forward,
  replace_llama_forward_with_emo_2_adaptive_forward,
  replace_llama_forward_with_emo_2_fixed_forward,
)
from copy import deepcopy

# EMO-1-Adaptive
replace_llama_forward_with_emo_1_adaptive_forward()

# Or
# EMO-2-Adaptive
replace_llama_forward_with_emo_2_adaptive_forward()

# Or
# EMO-2-Fixed
replace_llama_forward_with_emo_2_fixed_forward()
model = LlamaForCausalLM.from_pretrained(...)
cost_embedding = deepcopy(model.lm_head.weight.data)
model.register_buffer("cost_embedding", cost_embedding)


# Training code
...
```
## Setup
We recommend using `python>=3.10.0`, `torch>=2.0.1`, `transformers>=4.34.0`.
```bash
git clone https://github.com/DRSY/EMO.git
cd EMO
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```
You also need to install `DeepSpeed` if you prefer it over `FSDP` for distributed training.
```bash
pip install deepspeed
```
## Code Structure
This repository provide training scripts for three different scenarios, i.e., language modeling, continual fine-tuning, and instruction tuning, as discussed in the paper. Detailed instructions for each scenario are described in the following sections.
```
.
├── README.md
├── accelerate_configs
│   └── accelerate_config_0.yaml
├── continual_finetuning
│   ├── emo_llama.py
│   ├── finetune_fsdp.sh
│   ├── finetune_lora.sh
│   ├── icl.py
│   ├── llama_flash_attn_monkey_patch.py
│   ├── merge.sh
│   ├── merge_lora.py
│   ├── run_clm_trainer_emo.py
│   └── run_clm_trainer_emo_fsdp.py
├── emo_patch.py
├── instruction_tuning
│   ├── alpaca_gpt4_data.json
│   ├── deepspeed_zero2.json
│   ├── emo_llama.py
│   ├── flash_attention_patch.py
│   ├── train.py
│   ├── train_alpaca_gpt4_deepspeed.sh
│   └── train_alpaca_gpt4_fsdp.sh
├── language_modeling
│   ├── gpt2.py
│   ├── run_lm.py
│   ├── run_lm_gpt2.sh
│   └── test_utils.py
```
## 🏫 Language Modeling Experiments
The core code and scripts for language modeling experiments in the paper are located at [language_modeling](./language_modeling/). Model file that implements various training objective can be found at [gpt2.py](./language_modeling/gpt2.py).Training hyper-parameters can be adjusted in [run_lm_gpt2.sh](./language_modeling/run_lm_gpt2.sh). The argument "mode" specifies the training objective selected from `mle|mixce|tvd|emo|adaptive_emo`.
```bash
cd language_modeling
bash run_lm_gpt2.sh
```
We use [Mauve](https://github.com/krishnap25/mauve) as the primary evaluation metrics to assess the distributional similarity between model outputs and reference texts. Make sure you install it before running the above script.

## 📑 NLU Experiments
Scripts related to continual fine-tuning and downstream NLU evaluations are located under [continual_finetuning](./continual_finetuning/).
```bash
cd continual_finetuning
```
### Run continual fine-tuning on WikiText-103
The core script for lightweight continual fine-tuning on a single GPU using LoRA is named [finetune_lora.sh](./finetune_lora.sh). Training hyper-parameters are defined in the script and can be adjusted as needed.
```bash
bash finetune_lora.sh MODEL_PATH OUTPUT_PATH
```
MODEL_PATH points to the model name on HuggingFace or path to a local directory. OUTPUT_PATH specifies the output directory.
if the model is fine-tuned using LoRA, we need to first merge the trained LoRA weights into the original model checkpoint.
```bash
bash merge.sh OUTPUT_PATH MERGED_PATH
```
Specify your desired path for saving the merged model checkpoint at MERGED_PATH.

The core script for lightweight continual fine-tuning in a distributed setting using FSDP with FP16 mixed-precision training is named [finetune_fsdp.sh](./finetune_fsdp.sh). Training hyper-parameters are defined in the script and can be adjusted as needed.
```bash
bash finetune_fsdp.sh MODEL_PATH OUTPUT_PATH
```
### Run downstream tasks using few-shot in-context learning
The fine-tuned model can be evaluated on downstream natural language understanding tasks using few-shot in-context learning. Before running evaluation, make sure you have installed OpenICL:
```bash
git clone https://github.com/Shark-NLP/OpenICL
cd OpenICL
pip install -e .
```
Afterwards, we can run evaluation using the following command:
```bash
CUDA_VISIBLE_DEVICES=0, python icl.py --model_path OUTPUT_PATH/MERGED_PATH
```
> **Note**
> you may have to modify the model initialization part of OpenICL in order to run inference in torch.float16 data type.

## 📚 Instruction-Tuning
***Update!!!***: We recommend to use [Open-Instrcut](https://github.com/allenai/open-instruct) codebase for conducting instruction-tuning experiments owning to its advantages in terms of prompt design(allow for multi-turn dialogue) and handy train/eval dataset downloading/processing. After training, we obtain the following results using the official evaluation scripts in Open-Instruct:
|  | **CoT+UI** |  | **CoT+UI+FLANv2** |  | **Code+GPT4_Alpaca+CoT+FLANv2** |  | **Tulu_v2_sft_mix** |  |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|  | **MMLU** | **BBH** | **MMLU** | **BBH** | **MMLU** | **BBH** | **MMLU** | **BBH** |
| **MLE** | 46.3 | 36.7 | 49.0 | 37.8 | 47.6 | 38.3 | 50.4 | 41.1 |
| **EMO** | 47.7 | 37.3 | 49.7 | 38.6 | 47.8 | 41.7 | 51.4 | 43.1 |

CoT: Chain-of-Thought submix of FLANv2
UI: Unnatural Instruction
Code: Code Alpaca

***Below is the old version which uses Stanford Alpaca codebase***
EMO is also applicable in supervised instruction-tuning stage. We have tested on LLaMa-7B/13B and LLaMa2-7B/13B on the [Alpaca-GPT4](https://huggingface.co/datasets/vicgalle/alpaca-gpt4) and a recycled version of [Evol-Instruct-70K](https://huggingface.co/datasets/umd-zhou-lab/recycled_wiz70_v1) datasets. The responses of EMO-tuned models are more frequently deemed as better than those produced by MLE-tuned ones, judged by GPT-4, [Auto-J](https://github.com/GAIR-NLP/auto-j), and [PandaLM](https://github.com/WeOpenML/PandaLM).
We provide distributed training script(`FSDP` full fine-tuning using 4 GPUs) in [instruction_tuning](./instruction_tuning/) folder. Run the following command to launch training of specified model using the alpaca-gpt4 dataset:
```bash
cd instruction_tuning
mode=[emo|mle]
bash train_alpaca_gpt4_fsdp.sh MODEL_PATH OUTPUT_DIR $mode
```
Hyper-parameters such as training epochs, and global batch size are defined in the bash script. Feel free to adjust them as needed.
We also provide training script using `DeepSpeed` at [train_alpaca_gpt4_deepspeed.sh](./instruction_tuning/train_alpaca_gpt4_deepspeed.sh), which we empirically find to produce higher-quality models and faster training speed.
```bash
cd instruction_tuning
mode=[emo|mle]
bash train_alpaca_gpt4_deepspeed.sh MODEL_PATH OUTPUT_DIR $mode
```

# 🌐 Acknowledgements
+ Evaluation on NLU tasks is implemented using [OpenICL](https://github.com/Shark-NLP/OpenICL).
+ Instruction-tuning code is adapted from [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca).
+ Implementation of baselines are based on:
  + [TaiLr](https://github.com/thu-coai/TaiLr)
  + [MixCE](https://github.com/bloomberg/mixce-acl2023)


# Citation
If you find that our paper or code useful, please cite the paper as follows:
```latex
@misc{ren2023emo,
      title={EMO: Earth Mover Distance Optimization for Auto-Regressive Language Modeling}, 
      author={Siyu Ren and Zhiyong Wu and Kenny Q. Zhu},
      year={2023},
      eprint={2310.04691},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
