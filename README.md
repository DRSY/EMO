# EMO

## Abstract
Neural language models are probabilistic models of human text. They are predominantly trained using maximum likelihood estimation (MLE), which is equivalent
to minimizing the forward cross-entropy between the empirical data distribution and the model distribution. However, various degeneration phenomena are still
widely observed when decoding from the distributions learned by such models. We establish that the forward cross-entropy is suboptimal as a distance metric for aligning human and model distribution due to its (1) recall-prioritization (2) negative diversity ignorance and (3) train-test mismatch. In this paper, we propose Earth Mover Distance Optimization (EMO) for auto-regressive language modeling. EMO capitalizes on the inherent properties of earth mover distance to address the aforementioned challenges. Due to the high complexity of direct computation, we further introduce a feasible upper bound for EMO to ease end-to-end training. Upon extensive evaluation of language models trained using EMO and MLE. We find that EMO demonstrates a consistently better language modeling performance than MLE across domains. Moreover, EMO demonstrates noteworthy enhancements in downstream performance with minimal fine-tuning on merely 25,000 sentences. This highlights the tremendous potential of EMO as a lightweight calibration method for enhancing large-scale pre-trained language models.

## Usage
### Setup
```bash
git clone https://github.com/DRSY/EMO.git
cd EMO
pip install requirements.txt
```
### Run continual fine-tuning on WikiText-103 using LLaMa2-7B
The core script for lightweight continual fine-tuning is named run_emo.sh. Training hyper-parameters can be adjusted as needed.
```bash
bash run_emo.sh
```
### Merge and export the trained model
```bash
bash merge.sh MODEL_PATH OUTPUT_MODEL_PATH
```

### Run downstream tasks using few-shot in-context learning
The fine-tuned model can be evaluated on downstream natural language understanding tasks using few-shot in-context learning.
```bash
CUDA_VISIBLE_DEVICES=0, python icl.py --model_path OUTPUT_MODEL_PATH
```