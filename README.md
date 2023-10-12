# EMO

## Abstract
Neural language models are probabilistic models of human text. They are predominantly trained using maximum likelihood estimation (MLE), which is equivalent
to minimizing the forward cross-entropy between the empirical data distribution and the model distribution. However, various degeneration phenomena are still
widely observed when decoding from the distributions learned by such models. We establish that the forward cross-entropy is suboptimal as a distance metric for aligning human and model distribution due to its (1) recall-prioritization (2) negative diversity ignorance and (3) train-test mismatch. In this paper, we propose Earth Mover Distance Optimization (EMO) for auto-regressive language modeling. EMO capitalizes on the inherent properties of earth mover distance to address the aforementioned challenges. Due to the high complexity of direct computation, we further introduce a feasible upper bound for EMO to ease end-to-end training. Upon extensive evaluation of language models trained using EMO and MLE. We find that EMO demonstrates a consistently better language modeling performance than MLE across domains. Moreover, EMO demonstrates noteworthy enhancements in downstream performance with minimal fine-tuning on merely 25,000 sentences. This highlights the tremendous potential of EMO as a lightweight calibration method for enhancing large-scale pre-trained language models.

## Usage
### Setup
We recommend using Python>=3.10.0 and Pytorch>=2.0.1.
```bash
git clone https://github.com/DRSY/EMO.git
cd EMO
pip install -r requirements.txt
```
### Run continual fine-tuning on WikiText-103 using LLaMa2-7B
The core script for lightweight continual fine-tuning is named finetune.sh. Training hyper-parameters are defined in the script and can be adjusted as needed.
```bash
bash finetune.sh MODEL_PATH
```
MODEL_PATH points to the model name on HuggingFace or path to a local directory.
### Merge and export the trained model
if the model is fine-tuned using LoRA, we need to first merge the trained LoRA weights into the original model checkpoint.
```bash
bash merge.sh MODEL_PATH OUTPUT_MODEL_PATH
```
Specify your desired path for saving the merged model checkpoint at OUTPUT_MODEL_PATH.

### Run downstream tasks using few-shot in-context learning
The fine-tuned model can be evaluated on downstream natural language understanding tasks using few-shot in-context learning.
```bash
CUDA_VISIBLE_DEVICES=0, python icl.py --model_path OUTPUT_MODEL_PATH
```

### Supervised Fine-tuning
We provide the scripts for running EMO on Alpaca via supervised fine-tuning(SFT). Necessary scource code and dataset are located under the [stanford_alpaca](./stanford_alpaca/).
```bash
cd stanford_alpaca
bash train_emo.sh MODEL_PATH emo
```
By default, we use FSDP to fine-tune LLaMa-7B using 4 A100-80G GPUs. Hyper-parameters related to the training setting can be adjusted in [train_emo.sh](./stanford_alpaca/train_emo.sh).

## Acknowledgements
+ Evaluation on NLU tasks is implemented using [OpenICL](https://github.com/Shark-NLP/OpenICL).
+ SFT is implemented based on [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca).
