from openicl import PPLInferencer, AccEvaluator, TopkRetriever
from openicl import PromptTemplate
from datasets import load_from_disk
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--model_path', type=str, required=True)
args = parser.parse_args()

# SST-2
sst2_tp_dict = {
    0: '</E>Review: \"<X>\" It is positive.', 
    1: '</E>Review: \"<X>\" It is negative.', 
}
sst2_template = PromptTemplate(sst2_tp_dict, column_token_map={'text' : '<X>'}, ice_token='</E>')

# CR
cr_tp_dict = {
    1: "</E>Review: <X> It is positive.",
    0: "</E>Review: <X> It is negative." 
}
cr_template = PromptTemplate(cr_tp_dict, {'text': '<X>'}, ice_token='</E>')

# emo
emo_tp_dict = {
    0: "</E><X> It is unclear.",
    1: "</E><X> It is happy.",
    2: "</E><X> It is sad.",
    3: "</E><X> It is angry.",
}
emo_template = PromptTemplate(emo_tp_dict, {'text': '<X>'}, ice_token='</E>')

# subj
subj_tp_dict = {
    0: "</E><X> It is objective.",
    1: "</E><X> It is subjective.",
}
subj_template = PromptTemplate(subj_tp_dict, {'text': '<X>'}, ice_token='</E>')

# tweet_eval
tweet_eval_emotion_tp_dict = {
    0: "</E><X> It is anger.",
    1: "</E><X> It is joy.",
    2: "</E><X> It is optimism.",
    3: "</E><X> It is sadness.",
}
tweet_eval_emotion_template = PromptTemplate(tweet_eval_emotion_tp_dict, {'text': '<X>'}, ice_token='</E>')

# rotten tomatoes
rt_tp_dict = {
    1: "</E>Review: <X> It is positive",
    0: "</E>Review: <X> It is negative" 
}
rt_template = PromptTemplate(rt_tp_dict, {'text': '<X>'}, ice_token='</E>')

# SST-5
sst5_tp_dict = {
    0: "</E>Review: <X>\nSentiment: terrible",
    1: "</E>Review: <X>\nSentiment: bad",
    2: "</E>Review: <X>\nSentiment: okay",
    3: "</E>Review: <X>\nSentiment: good",
    4: "</E>Review: <X>\nSentiment: great",
}
sst5_template = PromptTemplate(sst5_tp_dict, column_token_map={'text' : '<X>'}, ice_token='</E>')

# AG_NEWS
ag_news_tp_dict = {
    0: "</E>\"<X>\" It is about world.",
    1: "</E>\"<X>\" It is about sports.",
    2: "</E>\"<X>\" It is about business.",
    3: "</E>\"<X>\" It is about science and technology.",
}
ag_news_template = PromptTemplate(ag_news_tp_dict, column_token_map={'text' : '<X>'}, ice_token='</E>')

# TREC
trec_tp_dict = {
    0: "</E>\"<X>\" It is about abbreviation.",
    1: "</E>\"<X>\" It is about entity.",
    2: "</E>\"<X>\" It is about description and abstract concept.",
    3: "</E>\"<X>\" It is about human being.",
    4: "</E>\"<X>\" It is about location.",
    5: "</E>\"<X>\" It is about numeric value."
}
trec_template = PromptTemplate(trec_tp_dict, column_token_map={'text' : '<X>'}, ice_token='</E>')

# SNLI & MNLI
xnli_tp_dict = {
    0: '</E><X1>? Yes, <X2>',
    1: '</E><X1>? Maybe, <X2>',
    2: '</E><X1>? No, <X2>'
}
xnli_template = PromptTemplate(xnli_tp_dict, column_token_map={'premise' : '<X1>', 'hypothesis' : '<X2>'}, ice_token='</E>')

# QNLI 
qnli_tp_dict = {
    0: "</E><X1> Can we know <X2>? Yes.",
    1: "</E><X1> Can we know <X2>? No.",
}
qnli_template = PromptTemplate(qnli_tp_dict, column_token_map={'sentence' : '<X1>', 'question' : '<X2>'}, ice_token='</E>')

# SICK 
sick_tp_dict = {
    0: "</E><X1> , <X2>",
    1: "</E><X1> Maybe, <X2>",
    2: "</E><X1> No, <X2>",
}
sick_template = PromptTemplate(sick_tp_dict, column_token_map={'sentence_A' : '<X1>', 'sentence_B' : '<X2>'}, ice_token='</E>')

rte_tp_dict = {
    0: "</E><X1> we can know: <X2>",
    1: "</E><X1> we can not know: <X2>",
}
rte_template = PromptTemplate(rte_tp_dict, column_token_map={'sentence1' : '<X1>', 'sentence2' : '<X2>'}, ice_token='</E>')

# Commonsense QA
cmsqa_template=PromptTemplate(
    {
        'A': "</E>Answer the following question:\n</Q>\nAnswer: </Ans1>",
        'B': "</E>Answer the following question:\n</Q>\nAnswer: </Ans2>",
        'C': "</E>Answer the following question:\n</Q>\nAnswer: </Ans3>",
        'D': "</E>Answer the following question:\n</Q>\nAnswer: </Ans4>",
        'E': "</E>Answer the following question:\n</Q>\nAnswer: </Ans5>",
    },
    {'question':'</Q>', 'A': '</Ans1>', 'B': '</Ans2>', 'C': '</Ans3>', 'D': '</Ans4>', 'E': '</Ans5>'},
    ice_token='</E>' 
)

fb_tp_dict = {
    0: "</E><X> It is negative.",
    1: "</E><X> It is neutral.",
    2: "</E><X> It is positive.",
}
fb_template = PromptTemplate(fb_tp_dict, {'sentence': '<X>'}, ice_token='</E>')

templates = {'sst2': sst2_template,
             'snli': xnli_template,
             'mnli': xnli_template,
             "qnli": qnli_template,
             "sst5": sst5_template,
             "ag_news": ag_news_template,
             "trec": trec_template,
             "commonsense_qa": cmsqa_template,
             "cr": cr_template,
             "rt": rt_template,
             "fb": fb_template,
             "subj": subj_template,
             "rte": rte_template,
             "sick": sick_template,
             "tweet_eval": tweet_eval_emotion_template,
             "emo": emo_template
            }


from datasets import load_dataset
from openicl import DatasetReader

data_path = {'sst2': ["gpt3mix/sst2", None],
             'snli': ['snli', None],
             'mnli': ['LysandreJik/glue-mnli-train', None],
             "qnli": ["glue", "qnli"],
             "sst5": ["SetFit/sst5", None],
             "ag_news": ["ag_news", None],
             "trec": ["trec", None],
             "commonsense_qa": ["commonsense_qa", None],
             "cr": ["SetFit/CR", None],
             "subj": ["SetFit/subj", None],
             "rt": ["rotten_tomatoes", None],
             "fb": ["financial_phrasebank", "sentences_allagree"],
             "sick": ["sick", None],
             "rte": ["glue", "rte"],
             "tweet_eval": ["tweet_eval", "emotion"],
             "emo": ['emo', None]
            }

input_columns={'sst2': ["text"],
             'snli': ['premise', 'hypothesis'],
             'mnli': ['premise', 'hypothesis'],
             "qnli": ["sentence", "question"],
             "sst5": ["text"],
             "ag_news": ["text"],
             "trec": ["text"],
             "commonsense_qa": ['question', 'A', 'B', 'C', 'D', 'E'],
             "cr": ["text"],
             "subj": ["text"],
             "rt": ["text"],
             "fb": ["sentence"],
             "sick": ["sentence_A", "sentence_B"],
             "rte": ["sentence1", "sentence2"],
             "tweet_eval": ["text"],
             "emo": ["text"]
            }

output_column={'sst2': 'label',
             'snli': 'label',
             'mnli': 'label',
             "qnli": 'label',
             "sst5": 'label',
             "ag_news": 'label',
             "trec": 'coarse_label',
             "commonsense_qa": "answerKey",
             "cr": 'label',
             "subj": 'label',
             "rt": 'label',
             "fb": 'label',
             "sick": 'label',
             "rte": 'label',
             "tweet_eval": 'label',
             "emo": 'label'
            }

# Change it for other tasks
for task_name in ['tweet_eval', 'trec', 'sst2', 'subj', 'cr', 'rt', 'ag_news']:
    path,name=data_path[task_name]
    dataset = load_dataset(path=path,name=name)

    if task_name == 'fb':
        dataset['train'] = load_dataset(path=path, name=name, split='train[:50%]')
        dataset['test'] = load_dataset(path=path, name=name, split='train[50%:]')
    if task_name == 'ag_news':
        dataset['train'] = load_dataset(path=path, name=name, split='train[:30%]')

    # Preprocess for commonsense_qa
    def pre_process(example):
        for i in range(5):
            example[chr(ord('A') + i)] = example['choices']['text'][i]
        return example

    if task_name=='commonsense_qa':
        dataset=dataset.map(pre_process).remove_columns(['question_concept', 'id', 'choices'])

    test_split={
        'sst2': 'test',
        'snli': 'test',
        "sst5": 'test',
        "ag_news": 'test',
        "trec": 'test',
        'mnli': 'validation', # cannot get gold labels for the test split
        "qnli": 'validation',
        "commonsense_qa": "validation",
        "cr": 'test',
        "rt": 'test',
        "subj": 'test',
        "fb": 'test',
        "sick": 'test',
        "rte": 'validation',
        "tweet_eval": 'test',
        "emo": 'test'
    }

    data=DatasetReader(dataset, input_columns=input_columns[task_name], output_column=output_column[task_name], test_split=test_split[task_name])


    # If you only want to test part of the test set for faster running, you can use the following codes
    # dataset['test'] = dataset['test'].select(list(range(100)))
    # dataset['validation'] = dataset['validation'].select(list(range(100))) # trec,agnews don't have validation
    # dataset['train'] = dataset['train'].select(list(range(100)))

    retriever = TopkRetriever(data, ice_num=8, test_split=test_split[task_name])

    inferencer = PPLInferencer(model_name=args.model_path, batch_size=8)
    predictions = inferencer.inference(retriever, ice_template=templates[task_name], output_json_filename=f'topk_{task_name}')
    scores = AccEvaluator().score(predictions=predictions, references=data.references)
    print(f'{task_name}:', scores)