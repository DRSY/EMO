from typing import List, Optional, Tuple

import torch
from torch import nn

import transformers
from transformers.models.llama.modeling_llama import CausalLMOutputWithPast

def emo_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
):
    """
    forward function of EMO
    """

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    # ======================================================================== #
    #                   Compute the MLE loss
    # ======================================================================== #
    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states)
    mask = labels[:, 1:].contiguous().view(-1)
    mask = (mask!=-100).to(logits.dtype)
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    logits = logits[:, :-1, :].contiguous().view(-1, logits.shape[-1])
    labels = labels[:, 1:].contiguous().view(-1)
    mle_loss = loss_fct(logits, labels)

    # ======================================================================== #
    #                   Compute the EMO loss
    # ======================================================================== #
    labels_tmp = labels.clone()
    labels_tmp[labels_tmp==(-100)] = 0
    one_hot = torch.nn.functional.one_hot(labels_tmp, num_classes=self.config.vocab_size).to(logits.dtype)
    stable_onehot = (one_hot+1e-15) / torch.linalg.vector_norm((one_hot+1e-15), ord=1, dim=-1, keepdim=True) # (bsz*seq_len, vocab_size)
    embedding_matrix = self.cost_embedding # (vocab_size, hidden_size)
    embedding_matrix = embedding_matrix / torch.linalg.vector_norm(embedding_matrix, ord=2, dim=1, keepdim=True)
    p_contextual_repr = stable_onehot @ embedding_matrix # (bsz*seq_len, hidden_size)
    q_grad = torch.log_softmax(logits, dim=-1).exp() # (bsz*seq_len, vocab_size)
    q_contextual_repr = q_grad @ embedding_matrix # (bsz*seq_len, hidden_size)
    emo_loss = (1 - torch.sum(p_contextual_repr*q_contextual_repr, dim=-1)) # (bsz*seq_len,)
    emo_loss = emo_loss * mask

    # ======================================================================== #
    #                   Compose the final loss
    # ======================================================================== #
    loss = (torch.min((mle_loss / (emo_loss+1e-10)).detach(), torch.ones_like(mle_loss, dtype=mle_loss.dtype, device=mle_loss.device)*3.0) * emo_loss + mle_loss) * 0.5
    loss = (loss * mask).sum() / (1e-15 + mask.sum())

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output
    
    return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def replace_llama_forward_with_emo_forward():
    transformers.models.llama.modeling_llama.LlamaForCausalLM.forward = emo_forward