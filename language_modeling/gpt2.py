# Copyright 2023 Bloomberg Finance L.P.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from transformers import GPT2LMHeadModel, GPT2PreTrainedModel
from transformers.utils import add_start_docstrings
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map

PARALLELIZE_DOCSTRING = r"""
    This is an experimental feature and is a subject to change at a moment's notice.
    Uses a device map to distribute attention modules of the model across several devices. If no device map is given,
    it will evenly distribute blocks across all devices.
    Args:
        device_map (`Dict[int, list]`, optional, defaults to None):
            A dictionary that maps attention modules to devices. Note that the embedding module and LMHead are always
            automatically mapped to the first device (for esoteric reasons). That means that the first device should
            have fewer attention modules mapped to it than other devices. For reference, the gpt2 models have the
            following number of attention modules:
                - gpt2: 12
                - gpt2-medium: 24
                - gpt2-large: 36
                - gpt2-xl: 48
    Example:
    ```python
    # Here is an example of a device map on a machine with 4 GPUs using gpt2-xl,
    # which has a total of 48 attention modules:
    model = GPT2LMHeadModel.from_pretrained("gpt2-xl")
    device_map = {
        0: [0, 1, 2, 3, 4, 5, 6, 7, 8],
        1: [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
        2: [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34],
        3: [35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47],
    }
    model.parallelize(device_map)
    ```
"""
DEPARALLELIZE_DOCSTRING = r"""
    Moves the model to cpu from a model parallel state.
    Example:
    ```python
    # On a 4 GPU machine with gpt2-large:
    model = GPT2LMHeadModel.from_pretrained("gpt2-large")
    device_map = {
        0: [0, 1, 2, 3, 4, 5, 6, 7],
        1: [8, 9, 10, 11, 12, 13, 14, 15],
        2: [16, 17, 18, 19, 20, 21, 22, 23],
        3: [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35],
    }
    model.parallelize(device_map)  # Splits the model across several devices
    model.deparallelize()  # Put the model back on cpu and cleans memory by calling torch.cuda.empty_cache()
    ```
"""
from typing import Optional, List, Tuple

class GPT2MIXModel(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"attn.masked_bias",
        r"attn.bias",
        r"lm_head.weight",
    ]

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.lm = GPT2LMHeadModel(config)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()
        
        # loss mode
        self.mode = None

        self.cost_embedding = self.lm.lm_head.weight.data.clone()

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.lm.transformer.h), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.lm.transformer.h))
        self.lm.transformer.parallelize(self.device_map)
        self.lm.lm_head = self.lm.lm_head.to(self.lm.transformer.first_device)
        self.model_parallel = True

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        self.lm.transformer.deparallelize()
        self.lm.transformer = self.lm.transformer.to("cpu")
        self.lm.lm_head = self.lm.lm_head.to("cpu")
        self.model_parallel = False
        torch.cuda.empty_cache()

    def get_output_embeddings(self):
        return self.lm.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm.lm_head = new_embeddings

    def get_input_embeddings(self):
        return self.lm.transformer.wte

    def set_input_embeddings(self, new_embeddings):
        self.lm.transformer.wte = new_embeddings

    def forward(
        self,
        input_ids,
        attention_mask,
        mode='mle',
        use_cache=False,
    ):
        labels = input_ids * attention_mask - 100 * (1 - attention_mask)
        outputs = self.lm.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
        )
        logits = outputs.logits
        seq_len = logits.shape[1]-1
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        logits = logits[:, :-1, :].contiguous().view(-1, logits.shape[-1])
        labels = labels[:, 1:].contiguous().view(-1)
        mle_loss = loss_fct(logits, labels)
        mask = attention_mask[:, 1:].contiguous().view(-1).to(logits.dtype)
        with torch.no_grad():
            q = torch.exp(-mle_loss.detach())
        if self.mode is not None:
            mode = self.mode
        if mode == 'mixce':
            """
            MixCE from ACL2023 https://aclanthology.org/2023.acl-long.502.pdf
            mixing_ratio==1 => standard MLE loss
            """
            mle_loss = (
                self.config.mixing_ratio * mle_loss
                + (1.0 - self.config.mixing_ratio) * (q) * mle_loss
            )
        elif mode == 'tvd':
            """
            Total variation distance from ICLR2023 https://openreview.net/pdf?id=VELL0PlWfc
            mixing_ratio==1 => standard MLE loss
            """
            gamma = self.config.mixing_ratio
            mle_loss =  self.config.mixing_ratio * mle_loss + (1.0-self.config.mixing_ratio)* torch.clamp((q / ((1-gamma) + gamma*q)), min=0.2, max=1.0) * mle_loss
        elif mode == 'emo':
            bsz = input_ids.shape[0]
            labels_tmp = labels.clone()
            labels_tmp[labels_tmp==(-100)] = 0
            one_hot = torch.nn.functional.one_hot(labels_tmp, num_classes=self.lm.config.vocab_size)
            stable_onehot = (one_hot+1e-15) / torch.linalg.vector_norm((one_hot+1e-15), ord=1, dim=-1, keepdim=True) # (bsz*seq_len, vocab_size)
            embedding_matrix = self.cost_embedding # (vocab_size, hidden_size)
            embedding_matrix = embedding_matrix / torch.linalg.vector_norm(embedding_matrix, ord=2, dim=1, keepdim=True)
            p_contextual_repr = stable_onehot @ embedding_matrix # (bsz*seq_len, hidden_size)
            q_grad = torch.log_softmax(logits, dim=-1).exp() # (bsz*seq_len, vocab_size)
            q_contextual_repr = q_grad @ embedding_matrix # (bsz*seq_len, hidden_size)
            cosine_dist = 1 - torch.sum(p_contextual_repr*q_contextual_repr, dim=-1) # (bsz*seq_len,)
            cosine_dist = cosine_dist.reshape(bsz, seq_len)
            emo_loss = cosine_dist.reshape(-1)
            mle_loss = (mle_loss + (mle_loss / (emo_loss+1e-10)).detach() * emo_loss) * 0.5
        elif mode == 'mle':
            """
            Standard Maximum likelihood estimation
            """
            mle_loss = mle_loss
        if self.config.reduction == "sum":
            # sum over sequence length and batch size
            outputs.loss = mle_loss.sum()
        else:
            # average over sequence length and batch size
            outputs.loss = (mle_loss * mask).sum() / (1e-15 + mask.sum())
        return outputs