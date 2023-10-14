import numpy as np
import torch
import transformers
from transformers import AutoModel, AutoTokenizer
from collections import defaultdict
from scipy.spatial.distance import cosine
from evaluate import load
import mauve
from collections import defaultdict
from tqdm.auto import tqdm
from prettytable import PrettyTable
from gpt2 import GPT2MIXModel
from opt import OPTMIXModel
import os
import math
perplexity = load("perplexity", module_type="metric")
rouge = load('rouge')
bertscore = load("bertscore")
# mauve = load('mauve')


def green(text):
    return '\033[92m' + text + '\033[0m'


def repetitiveness(continuation, n: int):
    """
    n-gram repetitiveness, the lower the better
    """
    counter = defaultdict(lambda: 0)

    def ngram(string, n_gram):
        tokens = string.split(" ")
        for i in range(0, len(tokens)-n_gram+1, 1):
            yield tokens[i:i+n_gram]
    for span in ngram(continuation, n):
        counter["_".join(span)] += 1
    if sum(counter.values()) == 0:
        return 0.0
    return 1 - len(list(counter.keys())) / sum(counter.values())


@torch.no_grad()
def semantic_sim(prefixes, continuations, model, tokenizer):
    inputs = tokenizer(prefixes, padding=True, truncation=True,
                       return_tensors="pt").to(model.device)
    prefix_embeddings = model(**inputs, output_hidden_states=True,
                        return_dict=True).pooler_output
    inputs = tokenizer(continuations, padding=True, truncation=True,
                       return_tensors="pt").to(model.device)
    continuation_embeddings = model(**inputs, output_hidden_states=True,
                        return_dict=True).pooler_output
    prefix_embeddings = prefix_embeddings / torch.linalg.vector_norm(prefix_embeddings, ord=2, dim=-1, keepdim=True)
    continuation_embeddings = continuation_embeddings / torch.linalg.vector_norm(continuation_embeddings, ord=2, dim=-1, keepdim=True)
    # cosine similarity
    cos_sim = (prefix_embeddings*continuation_embeddings).sum(dim=-1).mean().item()
    return cos_sim

def kl_div(logits_q, logits_p):
    """
    q: distributions being evaluated
    p: reference distribution
    """
    log_q = torch.log_softmax(logits_q, dim=-1)
    q = log_q.exp()
    log_p = torch.log_softmax(logits_p, dim=-1)
    res = (-q*log_p).sum(dim=-1) + (q*log_q).sum(dim=-1)
    return res

@torch.no_grad()
def compute_ppl(oracle_model, oracle_tokenizer, texts):
    """
    compute perpplexity using oracle model
    """
    import statistics, math
    BS = 64
    oracle_tokenizer.padding_side = 'right'
    losses = []
    for i in tqdm(range(0, len(texts), BS), total=(len(texts)//BS)):
        input_text = texts[i:(i+BS)]
        inputs = oracle_tokenizer(input_text, padding=True, return_tensors='pt').to('cuda')
        inputs['labels'] = inputs['input_ids'].clone()
        inputs['labels'][inputs['labels'].eq(oracle_tokenizer.pad_token_id)] = -100
        outputs = oracle_model(**inputs)
        ce_loss = outputs.loss
        losses.append(ce_loss.cpu().item())
    mean_ce_loss = statistics.mean(losses)
    ppl = math.exp(mean_ce_loss)
    print(f"PPL using oracle model: {ppl:.2f}")
    return ppl


@torch.no_grad()
def distribution_measures(args, model, raw_datasets, split='validation'):
    """
    Computes a set of distributional measures that reflect the recall/precision/middle of learned probablistic model distribution
    """
    from transformers import AutoModelForCausalLM
    device = 'cuda:0'
    # if args.dataset_name is None:
    #     reference_model_id = "gpt2-medium"
    # else:
    # reference_model_id = "EleutherAI/gpt-neo-125m"
    # reference_model_id = "./gpt_neo_1.3b"
    # reference_model_id = "gpt2-large"
    reference_model_id = "./gpt_neo_1.3b/gptneo-1.3b"
    reference_model = AutoModelForCausalLM.from_pretrained(reference_model_id).to(device)
    reference_model.eval()
    if isinstance(model, GPT2MIXModel) or isinstance(model, OPTMIXModel):
        model = model.lm
    eval_model = transformers.AutoModelForCausalLM.from_pretrained(args.model_name_or_path).to(device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    eval_model.load_state_dict(model.state_dict())
    eval_model.eval()
    eval_set = raw_datasets[split]
    try:
        corpus = eval_set['text']
    except:
        corpus = eval_set['sentence']
    measures = defaultdict(lambda: list())
    losses = []
    for i in tqdm(range(len(corpus))):
        text = corpus[i]
        if len(text.split(" "))<20 or tokenizer(text, return_tensors='pt').input_ids.shape[-1]>500:
            continue
        inputs = tokenizer([text], return_tensors='pt').to(device)
        inputs['labels'] = inputs['input_ids'].clone()
        outputs = eval_model(**inputs)
        forward_ce_loss = outputs.loss.item()
        losses.append(forward_ce_loss)
        logits_model = outputs.logits[:, 1:, :]
        prob_model = torch.softmax(logits_model, dim=-1)
        logits_reference = reference_model(**inputs).logits[:, 1:, :]
        prob_reference = torch.softmax(logits_reference, dim=-1)
        assert (prob_model.shape==prob_reference.shape)
        # forward ce
        forward_ce = (-prob_reference*torch.log_softmax(logits_model, dim=-1)).sum(dim=-1).mean(dim=-1).item()
        # reverse ce
        reverse_ce = (-prob_model*torch.log_softmax(logits_reference, dim=-1)).sum(dim=-1).mean(dim=-1).item()
        # tvd
        tvd = 0.5*(torch.abs(prob_model-prob_reference)).sum(dim=-1).mean(dim=-1).item()
        # reverse KL div
        reverse_kl_div = ((-prob_model*torch.log_softmax(logits_reference, dim=-1)).sum(dim=-1) + (prob_model*torch.log_softmax(logits_model, dim=-1)).sum(dim=-1)).mean(dim=-1).item()
        measures['forward_ce'].append(forward_ce)
        measures['reverse_ce'].append(reverse_ce)
        measures['tvd'].append(tvd)
        measures['reverse_kl'].append(reverse_kl_div)
    # average
    from statistics import mean
    for key in measures:
        measures[key] = mean(measures[key])
    print("Distribution measures:")
    for k in measures:
        print(f"{k}: {measures[k]:.3f}")
    mean_ce_loss = mean(losses)
    ppl = math.exp(mean_ce_loss)
    print(f"Test set perplexity: {ppl:.3f}")
    return measures

@torch.no_grad()
def test_batch(iteration, args, model, raw_datasets, split='validation', do_sample=False, p=0.95):
    """
    args: parsed arguments
    model: the model that is being trained
    raw_datasets: the dataset from which prefixes are sampled from
    """
    if isinstance(model, GPT2MIXModel) or isinstance(model, OPTMIXModel):
        model = model.lm
    global perplexity, rouge, bertscore, mauve
    device = 'cuda:0'
    eval_model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path).to(device)
    eval_model.load_state_dict(model.state_dict())
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'
    tokenizer.truncation_side = 'left'
    eval_model.eval()
    val_set = raw_datasets[split]
    cnt = 0
    if args.dataset_name is None:
        if 'webtext' in args.train_file:
            eval_num = 5000
            prefix_len = 20
            continuation_len = 80
        elif 'writing' in args.train_file:
            eval_num = 5000
            prefix_len = 35
            continuation_len = 80
        else:
            # wikitext-103
            eval_num = 5000
            prefix_len = 20
            continuation_len = 80
    elif 'wiki' in args.dataset_name:
        eval_num = 4000
        prefix_len = 20
        continuation_len = 80
    elif 'ptb' in args.dataset_name:
        eval_num = 4000
        prefix_len = 5
        continuation_len = 25
    elif 'ag' in args.dataset_name:
        eval_num = 4000
        prefix_len = 10
        continuation_len = 30
    # for text in tqdm(prefixs, total=len(prefixs)):
    gold_continuations = []
    pred_continuations = []
    eval_num = min(eval_num, len(val_set))
    BS = 64
    for i in tqdm(range(0, eval_num, BS), total=(eval_num//BS)):
        try:
            texts = val_set['text'][i:(i+BS)]
        except:
            texts = val_set['sentence'][i:(i+BS)]
        texts_ = []
        for text in texts:
            if len(text) < 100 or len(text.split(" ")) < (prefix_len+continuation_len+10) or "==" in text: continue
            texts_.append(text)
        prefixs = []
        cur_gold_continuations = []
        for text in texts_:
            prefix = " ".join(text.split(" ")[:prefix_len])
            gold_continuation = " ".join(text.split(
                " ")[prefix_len:prefix_len+continuation_len])
            prefixs.append(prefix)
            cur_gold_continuations.append(prefix+" "+gold_continuation)
        cnt += 1
        if len(prefixs)<=0: continue
        inputs = tokenizer(prefixs, return_tensors='pt', padding=True, truncation=True).to(device)
        new_len = args.decode_newlen
        generate_ids = eval_model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            do_sample=do_sample,
            top_k=50 if args.decoding_mode=='top_k' else 0,
            top_p=p if args.decoding_mode=='top_p' else 1.0,
            typical_p=0.2 if args.decoding_mode=='typical' else 1.0,
            min_new_tokens=new_len,
            max_new_tokens=new_len+1,
        )
        output = tokenizer.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        for j in range(len(prefixs)):
            continuation = output[j][len(prefixs[j]):]
            pred_continuations.append(prefixs[j]+" "+continuation)
        gold_continuations.extend(cur_gold_continuations)
    mauve_score = mauve.compute_mauve(
        p_text=gold_continuations, q_text=pred_continuations, device_id=0, max_text_length=256, verbose=True, batch_size=32, featurize_model_name='./gpt2-large').mauve
    rouge_results = rouge.compute(
        predictions=pred_continuations, references=gold_continuations)
    table = PrettyTable()
    table.field_names = ["Mauve↑", "R-1↑", "R-2↑", "R-L↑"]
    table.add_row([
                   round(mauve_score, 3),
                   round(rouge_results['rouge1'], 4),
                   round(rouge_results['rouge2'], 4),
                   round(rouge_results['rougeL'], 4),
                   ])
    print(table)
    del eval_model
    torch.cuda.empty_cache()
    # write generation to file
    with open(os.path.join(args.output_dir, f"completion_{do_sample}sample_{iteration}.json"), "w") as f:
        import json
        data = [{"gold": gold, "pred": pred}
                for gold, pred in zip(gold_continuations, pred_continuations)]
        json.dump(data, f, indent=4)
    # return mauve_score
    return mauve_score
