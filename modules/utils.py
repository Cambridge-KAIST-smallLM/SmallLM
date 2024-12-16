
# coding=utf-8
# Copyright 2023 The HuggingFace Team. All rights reserved.
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
import random
import numpy as np
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM
from typing import Any, List, Literal


DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"



def maybe_insert_system_message(messages, tokenizer):
    if messages[0]["role"] == "system":
        return

    # chat template can be one of two attributes, we check in order
    chat_template = tokenizer.chat_template
    if chat_template is None:
        chat_template = tokenizer.default_chat_template

    # confirm the jinja template refers to a system message before inserting
    if "system" in chat_template or "<|im_start|>" in chat_template:
        messages.insert(0, {"role": "system", "content": ""})

def is_openai_format(messages: Any) -> bool:
    """
    Check if the input messages are in OpenAI format.
    Args:
        messages (`Any`):
            Messages to check.
    Returns:
        `bool`: Whether the messages are in OpenAI format.
    """
    if isinstance(messages, list) and all(isinstance(message, dict) for message in messages):
        return all("role" in message and "content" in message for message in messages)
    return False

def map_chat_template_by_task(
    example,
    tokenizer,
    training_type: Literal["SFT", "RM", "ORPO", "DPO", "LINEAR", "PRETRAIN", "PROPOSED"],
    auto_insert_empty_system_msg: bool = False,
):
    ### TODO: Handle chat templates with inherent errors
    if training_type.lower() == "sft":
        messages = example["messages"]
        # We add an empty system message if there is none
        if auto_insert_empty_system_msg:
            maybe_insert_system_message(messages, tokenizer)
        example["text"] = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    elif training_type.lower() in ['pretrain','proposed']:
        example['input_ids'] = torch.tensor(example['input_ids'])
        example['labels'] = torch.tensor(example['input_ids'])
    elif training_type.lower() == "rm":
        example["input_ids_chosen"] = []
        example["attention_mask_chosen"] = []
        example["input_ids_rejected"] = []
        example["attention_mask_rejected"] = []
        
        for chosen, rejected in zip(example["chosen"], example["rejected"]):
            tokenized_chosen_ = tokenizer.apply_chat_template(chosen, add_generation_prompt=False, tokenize=False)
            tokenized_rejected_ = tokenizer.apply_chat_template(rejected, add_generation_prompt=False, tokenize=False)
            
            tokenized_chosen = tokenizer(tokenized_chosen_, add_special_tokens=False)
            tokenized_rejected = tokenizer(tokenized_rejected_, add_special_tokens=False)

            example["input_ids_chosen"].append(tokenized_chosen["input_ids"])
            example["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
            example["input_ids_rejected"].append(tokenized_rejected["input_ids"])
            example["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])

    elif training_type.lower() in ['dpo', 'orpo', 'linear']:
        if all(k in example.keys() for k in ("chosen", "rejected")):
            if not is_openai_format(example["chosen"]) or not is_openai_format(example["rejected"]):
                raise ValueError(
                    f"Could not format example as dialogue for `{training_type}` training_type! Require OpenAI format for all messages"
                )

            # For DPO/ORPO, the inputs are triples of (prompt, chosen, rejected), where `chosen` and `rejected` are the final turn of a dialogue
            # We therefore need to extract the N-1 turns to form the prompt
            if "prompt" in example and is_openai_format(example["prompt"]):
                prompt_messages = example["prompt"]
                chosen_messages = example["chosen"]
                rejected_messages = example["rejected"]
            else:
                prompt_messages = example["chosen"][:-1]
                # Now we extract the final turn to define chosen/rejected responses
                chosen_messages = example["chosen"][-1:]
                rejected_messages = example["rejected"][-1:]

            # Prepend a system message if the first message is not a system message
            if auto_insert_empty_system_msg:
                maybe_insert_system_message(prompt_messages, tokenizer)
            
            example["text_prompt"] = tokenizer.apply_chat_template(prompt_messages, tokenize=False)
            example["text_chosen"] = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
            example["text_rejected"] = tokenizer.apply_chat_template(rejected_messages, tokenize=False)

            if tokenizer.bos_token:
                if example["text_chosen"].startswith(tokenizer.bos_token): 
                    example["text_chosen"] = example["text_chosen"][len(tokenizer.bos_token):] 
                if example["text_rejected"].startswith(tokenizer.bos_token): 
                    example["text_rejected"] = example["text_rejected"][len(tokenizer.bos_token):] 
        else:
            raise ValueError(
                f"Could not format example as dialogue for `{training_type}` training_type! Require either the "
                f"`[chosen, rejected]` or `[prompt, chosen, rejected]` keys but found {list(example.keys())}"
            )
    else:
        raise ValueError(
            f"training_type {training_type} not supported, please ensure that the provided training_type is one of ['sft', 'generation', 'rm', 'dpo', 'orpo']"
        )
    return example

def print_sample_items(
    data,
    logger,
    training_type: str,
    sample_num: int = 3,
):
    if training_type.lower() in ["orpo", "dpo", "linear"]:
        for index in random.sample(range(len(data)), sample_num):
            logger.info(f"Prompt sample {index} of the raw training set:\n\n{data[index]['prompt']}")
            logger.info(f"Chosen sample {index} of the raw training set:\n\n{data[index]['chosen']}")
            logger.info(f"Rejected sample {index} of the raw training set:\n\n{data[index]['rejected']}")
    elif training_type.lower() == "sft":
        for index in random.sample(range(len(data)), sample_num):
            logger.info(f"Sample {index} of the processed training set:\n\n{data[index]['text']}")
    elif training_type.lower() == 'rm':
        pass
    else:
        raise Exception("Check the training type.")

def get_batches(items, batch_size):
    num_batches = (len(items) + batch_size - 1) // batch_size
    batches = []

    for i in range(num_batches):
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, len(items))
        batch = items[start_index:end_index]
        batches.append(batch)

    return batches


def initialize_reward_model_head(model: AutoModel, tokenizer: AutoTokenizer):
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.truncation_side = "left"

    model.config.pad_token_id = tokenizer.pad_token_id
    model.resize_token_embeddings(len(tokenizer))
    
    model.score = nn.Linear(model.config.hidden_size, 1, bias=False)

    print(">>> Classification head initialized to with normal distribution.: ", model.score.weight.size())
    nn.init.normal_(model.score.weight, mean=0.0, std=1/np.sqrt(model.config.hidden_size+1))

    return model, tokenizer

def initialize_model(attn_implementation, torch_dtype, tokenizer):
    # Define the configuration
    config_dict = {
        "_name_or_path": "JW17/SmolLM-14m-v0.1",
        "architectures": ["LlamaForCausalLM"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 0,
        "eos_token_id": 0,
        "hidden_act": "gelu",
        "hidden_size": 128,
        "initializer_range": 0.02,
        "intermediate_size": 512,
        "is_llama_config": True,
        "max_position_embeddings": 2048,
        "model_type": "llama",
        "num_attention_heads": 4,
        "num_hidden_layers": 6,
        "num_key_value_heads": 4,
        "pretraining_tp": 1,
        "rms_norm_eps": 1e-05,
        "rope_interleaved": False,
        "rope_scaling": None,
        "rope_theta": 100000,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "use_cache": True,
        "vocab_size": tokenizer.vocab_size,
        "flash_attn": True
    }

    # Create config object
    config = LlamaConfig.from_dict(
        config_dict=config_dict
    )

    # Initialize model with custom config
    model = AutoModelForCausalLM.from_config(
        config=config,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
    )

    return model



class OutputEmbeddingSelectiveUpdate(LlamaForCausalLM):
    def __init__(self, config, vocab_size, num_tokens_to_update):
        super().__init__(config)
        self.num_tokens_to_update = num_tokens_to_update
        self.token_frequencies = torch.zeros(
            vocab_size,
            dtype=torch.float32,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        self.world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

    def calculate_global_token_frequencies(self, input_ids):
        """
        Calculate token frequencies for the entire batch across all GPUs.
        """
        # Step 1: Calculate local token frequencies
        batch_tokens = input_ids.view(-1)  # Flatten the input tensor
        local_token_counts = torch.bincount(batch_tokens, minlength=self.token_frequencies.size(0)).float()

        # Step 2: Synchronize token frequencies across GPUs
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(local_token_counts, op=torch.distributed.ReduceOp.SUM)

        # Now `local_token_counts` contains the global token frequencies across all GPUs
        return local_token_counts

    def forward(self, input_ids, *args, **kwargs):
        # Step 1: Calculate and accumulate global token frequencies
        labels = input_ids[:, 1:].contiguous()  # Shift input_ids left for labels
        input_ids = input_ids[:, :-1].contiguous()  # Remove the last token from input_ids
        batch_token_frequencies = self.calculate_global_token_frequencies(input_ids)
        self.token_frequencies += batch_token_frequencies

        # Forward pass
        outputs = super().forward(input_ids=input_ids, *args, **kwargs)

        # Hook to modify gradients of the output embeddings
        def gradient_weighting_hook(grad):
            token_probs = self.token_frequencies ** 0.75
            token_probs /= token_probs.sum()

            top_probs, _ = torch.topk(token_probs, 100, largest=True)
            lowest_probs, _ = torch.topk(token_probs, 100, largest=False)

            # Calculate their sums
            sum_top_100 = top_probs.sum().item()
            sum_lowest_100 = lowest_probs.sum().item()

            # Log the results to wandb
            wandb.log({
            "sum_top_100_probs": sum_top_100,
            "sum_lowest_100_probs": sum_lowest_100,},step=self.global_step,)

            top_probs, top_indices = torch.topk(token_probs, 10000, largest=True)
            log_data = {
            f"token_probabilities/{i}": prob.item()
            for i, prob in zip(top_indices.tolist(), top_probs.tolist())}

            # Log to wandb
            wandb.log(log_data, step=self.global_step)

            # Initialize gradient weights as the smoothed probabilities
            grad_weights = token_probs.view(-1, 1)  # Reshape to match embedding dimensions
            grad_weights = grad_weights.expand(-1, grad.size(1))

            # Ensure target tokens' gradients are not scaled
            target_tokens = labels.reshape(-1)  # Flatten labels (target tokens)
            target_mask = torch.zeros_like(grad, dtype=torch.bool)
            target_mask[target_tokens] = True

            # Create a mask for non-target tokens
            non_target_mask = ~target_mask

            # Apply weights only to non-target tokens
            grad[non_target_mask] *= grad_weights[non_target_mask]

            
            #grad[target_mask] *= 1.0  # No scaling for target tokens, or set it to a factor > 1

            return grad

        self.lm_head.weight.register_hook(gradient_weighting_hook)
        return outputs

    @classmethod
    def from_config(cls, config, vocab_size, num_tokens_to_update, torch_dtype=None, attn_implementation=None):
        model = AutoModelForCausalLM.from_config(
            config=config,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
        )
        model.num_tokens_to_update = num_tokens_to_update
        model.token_frequencies = torch.zeros(vocab_size, dtype=torch.float32, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        return cls(model.config, vocab_size, num_tokens_to_update)
    
def initialized_model_proposed_method(attn_implementation, torch_dtype, tokenizer, num_tokens_update):
    """Initialize a LigerKernel-based model from scratch with frequency-based selective updates."""

    config_dict = {
        "_name_or_path": "JW17/SmolLM-14m-v0.1",
        "architectures": ["LlamaForCausalLM"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 0,
        "eos_token_id": 0,
        "hidden_act": "gelu",
        "hidden_size": 128,
        "initializer_range": 0.02,
        "intermediate_size": 512,
        "is_llama_config": True,
        "max_position_embeddings": 2048,
        "model_type": "llama",
        "num_attention_heads": 4,
        "num_hidden_layers": 6,
        "num_key_value_heads": 4,
        "pretraining_tp": 1,
        "rms_norm_eps": 1e-05,
        "rope_interleaved": False,
        "rope_scaling": None,
        "rope_theta": 100000,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "use_cache": True,
        "vocab_size": tokenizer.vocab_size,
        "flash_attn": True
    }

    config = LlamaConfig.from_dict(
        config_dict=config_dict
    )

    model = OutputEmbeddingSelectiveUpdate.from_config(config, config.vocab_size, num_tokens_update,torch_dtype=torch_dtype,
        attn_implementation=attn_implementation)

    return model

