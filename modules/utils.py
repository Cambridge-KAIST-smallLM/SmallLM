
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
    def __init__(self, config, vocab_size, training_args, max_steps=100000):
        super().__init__(config)

        # Store token frequencies on CPU to save GPU memory
        self.token_frequencies = torch.zeros(vocab_size, dtype=torch.float32, device="cpu")
        self.global_step = 0  # Initialize global_step
        self.max_steps = max_steps  # Total training steps

        # Extract warmup_steps from training_args
        self.warmup_steps = training_args.warmup_steps or int(0.1 * max_steps)

    def set_step(self, step):
        """
        Update the current global step in the model.
        """
        self.global_step = step

    def calculate_batch_token_frequencies(self, input_ids):
        """
        Calculate token frequencies for the current batch.
        """
        batch_tokens = input_ids.detach().to("cpu").reshape(-1)
        batch_token_counts = torch.bincount(batch_tokens, minlength=self.token_frequencies.size(0)).float()
        return batch_token_counts

    def gradient_weighting_schedule(self):
        """
        Exponential schedule: Gradually increase masking after the warm-up phase.
        """

        # Progress after warm-up
        progress = (self.global_step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        #k = 3   Exponential steepness
        progress_tensor = torch.tensor(progress, dtype=torch.float32, device="cuda" if torch.cuda.is_available() else "cpu")
        scaling_factor = torch.clamp(progress_tensor, min=0.0, max=1.0)
        #scaling_factor = 1 - torch.exp(-k * progress_tensor)
        return scaling_factor

    def forward(self, input_ids, *args, **kwargs):
        labels = input_ids[:, 1:].contiguous()  # Next-token labels
        #input_ids = input_ids[:, :-1].contiguous()  # Shift input_ids left
        #labels = input_ids.clone()

        # Update token frequencies for the current batch
        with torch.no_grad():
            batch_token_frequencies = self.calculate_batch_token_frequencies(input_ids)
            self.token_frequencies.add_(batch_token_frequencies)

        # Increment step counter
        self.global_step += 1

        # Hook to modify gradients
        if self.global_step >= self.warmup_steps and not hasattr(self.lm_head.weight, "_hook_registered"):
            def gradient_weighting_hook(grad):

            # Transfer token frequencies to GPU
                token_probs = self.token_frequencies.to(grad.device, dtype=grad.dtype, non_blocking=True)
                #token_probs = token_probs ** 0.75
                token_probs /= token_probs.sum()

            # Compute scaling factor
                scaling_factor = self.gradient_weighting_schedule()

            # Identify non-target tokens
                target_tokens = labels.reshape(-1)
                target_mask = torch.zeros_like(grad, dtype=torch.bool)
                target_mask[target_tokens] = True
                target_mask[-1, :] = False
                non_target_mask = ~target_mask

            # Mask non-target gradients
                masked_grad = grad.clone()
                masked_grad[non_target_mask] *= token_probs.unsqueeze(1).expand_as(grad)[non_target_mask]

            # Blend gradients
                grad = (1 - scaling_factor) * grad + scaling_factor * masked_grad
                return grad

        # Register hook
            self.lm_head.weight.register_hook(lambda grad: gradient_weighting_hook(grad.detach()))
            self.lm_head.weight._hook_registered = True

        # Forward pass
        outputs = super().forward(input_ids=input_ids, *args, **kwargs)
        return outputs

    @classmethod
    def from_config(cls, config, vocab_size, training_args, torch_dtype=None, attn_implementation=None, max_steps=100000):
        model = AutoModelForCausalLM.from_config(
            config=config,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
        )
        return cls(model.config, vocab_size, training_args, max_steps=max_steps)



    
def initialized_model_proposed_method(attn_implementation, torch_dtype, tokenizer, training_args):
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

    model = OutputEmbeddingSelectiveUpdate.from_config(config, config.vocab_size, training_args=training_args, max_steps=training_args.max_steps ,torch_dtype=torch_dtype,
        attn_implementation=attn_implementation)

    return model

