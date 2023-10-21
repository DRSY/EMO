# EMO
This is the public codebase for arxiv paper: [EMO: Earth Mover Distance Optimization for Auto-regressive Language Modeling](https://arxiv.org/abs/2310.04691).

## Abstract
Neural language models are probabilistic models of human text. They are predominantly trained using maximum likelihood estimation (MLE), which is equivalent
to minimizing the forward cross-entropy between the empirical data distribution and the model distribution. However, various degeneration phenomena are still
widely observed when decoding from the distributions learned by such models. We establish that the forward cross-entropy is suboptimal as a distance metric for aligning human and model distribution due to its (1) recall-prioritization (2) negative diversity ignorance and (3) train-test mismatch. In this paper, we propose Earth Mover Distance Optimization (EMO) for auto-regressive language modeling. EMO capitalizes on the inherent properties of earth mover distance to address the aforementioned challenges. Due to the high complexity of direct computation, we further introduce a feasible upper bound for EMO to ease end-to-end training. Upon extensive evaluation of language models trained using EMO and MLE. We find that EMO demonstrates a consistently better language modeling performance than MLE across domains. Moreover, EMO demonstrates noteworthy enhancements in downstream performance with minimal fine-tuning on merely 25,000 sentences. This highlights the tremendous potential of EMO as a lightweight calibration method for enhancing large-scale pre-trained language models.

# Usage
## Setup
We recommend using Python>=3.10.0 and Pytorch>=2.0.1.
```bash
git clone https://github.com/DRSY/EMO.git
cd EMO
pip install -r requirements.txt
```
## Language Modeling Experiments
The core code and scripts for language modeling experiments in the paper are located at [language_modeling](./language_modeling/). Model file that implements various training objective can be found at [gpt2.py](./language_modeling/gpt2.py).Training hyper-parameters can be adjusted in [run_lm_gpt2.sh](./language_modeling/run_lm_gpt2.sh). The argument "mode" specifies the training objective(mle|mixce|tvd|emo).
```bash
cd language_modeling
bash run_lm_gpt2.sh
```
We use [Mauve](https://github.com/krishnap25/mauve) as the primary evaluation metrics, make sure you install it before running the above script.

## NLU Experiments
### Run continual fine-tuning on WikiText-103
The core script for lightweight continual fine-tuning on a single GPU using LoRA is named [finetune.sh](./finetune.sh). Training hyper-parameters are defined in the script and can be adjusted as needed.
```bash
bash finetune.sh MODEL_PATH OUTPUT_PATH
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
**Note**: you may have to modify the model initialization part of OpenICL in order to run inference in torch.float16 data type.

## Instruction-tuning
EMO is also applicable in supervised instruction-tuning stage. We provide distributed training script(FSDP full fine-tuning using 4 GPUs) in [instruction_tuning](./instruction_tuning/) folder. Run the following command to launch training of specified model using the alpaca-gpt4 dataset:
```bash
cd instruction_tuning
bash train_emo_alpaca_gpt4.sh MODEL_PATH OUTPUT_DIR
```
Training hyper-parameters such as training objective(mle|emo), epochs, and global batch size are defined in [train_emo_alpaca_gpt4.sh](./instruction_tuning/train_emo_alpaca_gpt4.sh) and are kept the same as in Stanford Alpaca codebase, feel free to adjust them as needed.

## Acknowledgements
+ Evaluation on NLU tasks is implemented using [OpenICL](https://github.com/Shark-NLP/OpenICL).
+ Instruction-tuning code is adapted from [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca).
+ Implementation of baselines are from:
  + [TaiLr](https://github.com/thu-coai/TaiLr)
  + [MixCE](https://github.com/bloomberg/mixce-acl2023)


## Citation
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
