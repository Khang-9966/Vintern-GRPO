# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import os
import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Union

import torch
import torch.utils.data
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url
from collections import defaultdict

import copy


if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class MultimodalGRPOTrainer(Trainer):
    def __init__(self,**model_init_kwargs):
        self._metrics = defaultdict(list)
        super().__init__(**model_init_kwargs)
        
    def _prepare_deepspeed_orig(self, model):
        # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)

        # If ZeRO-3 is used, we shard both the active and reference model.
        # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
        if config_kwargs['zero_optimization']['stage'] != 3:
            config_kwargs['zero_optimization']['stage'] = 0
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        return model

    def _prepare_deepspeed(self, model):
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = deepspeed_plugin.deepspeed_config
        if config_kwargs['zero_optimization']['stage'] == 3:
            print('Enable DPOTrainer._prepare_deepspeed')
            return self._prepare_deepspeed_orig(model)

        print('Disable DPOTrainer._prepare_deepspeed')
        for param in model.parameters():
            param.requires_grad = False

        model.eval()
        model = model.to(self.accelerator.device)
        return model

    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(self, model, input_ids, attention_mask, pixel_values, position_ids, image_flags):
        logits = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, position_ids=position_ids, image_flags=image_flags ).logits  # (B, L, V)
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)


    # Trainer "prepares" the inputs before calling `compute_loss`. It converts to tensor and move to device.
    # Since we preprocess the data in `compute_loss`, we need to override this method to skip this step.
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        return inputs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        ASSISTENT_TOKEN_ID = 77091
        start_turn_indices = (inputs["input_ids"] == ASSISTENT_TOKEN_ID).nonzero(as_tuple=True)[1]
        # Lấy index lớn nhất
        end_ques_index = start_turn_indices.max().item() + 2

        temp_inputs = {
            "input_ids" : inputs["input_ids"][:,:end_ques_index],
            "labels" : inputs["labels"][:,:end_ques_index],
            "attention_mask" : inputs["attention_mask"][:,:end_ques_index],
            "position_ids" : inputs["position_ids"][:,:end_ques_index],
            "pixel_values" : inputs["pixel_values"],
            "image_flags" : inputs["image_flags"]
        }
     
        generation_config = GenerationConfig(
            max_new_tokens=300,
            do_sample=True,  
            temperature=1., # HACK
            num_return_sequences=1,
            repetition_penalty=1.,
            num_beams = 1,
            length_penalty=1.
        )
        generation_config = dict(max_new_tokens= 300, do_sample=True, num_beams = 1, temperature=1.0 , repetition_penalty=1.0, length_penalty=1.)

        prompt_ids, prompt_mask = temp_inputs["input_ids"], temp_inputs["attention_mask"]

        ####################### Generate completions
        with unwrap_model_for_generation(model, self.accelerator ) as unwrapped_model:
            # print(unwrapped_model.language_model.base_model.model.model.layers[0].mlp.gate_proj.weight)
            # print(unwrapped_model.language_model.base_model.model.model.embed_tokens.weight)
            # print(unwrapped_model.language_model.base_model.model.lm_head.weight)
            all_completions = []
            
            for i in range(self.args.num_generations):  # -1 because we already have one generation
                completion = unwrapped_model.generate(**temp_inputs, **generation_config)
                completion = torch.cat((temp_inputs["input_ids"],completion),dim=1)
                all_completions.append(completion)

            
            # print(all_completions)
            # print("="*100)
            ############################# Stack all completions and pad if needed
            max_length = max(completion.size(1) for completion in all_completions)
            padded_completions = []
            pad_token_id = 151643
            for completion in all_completions:
                if completion.size(1) < max_length:
                    padding = torch.full((completion.size(0), max_length - completion.size(1)), 
                                    pad_token_id, 
                                    dtype=completion.dtype,
                                    device=completion.device)
                    padded_completion = torch.cat([completion, padding], dim=1)
                else:
                    padded_completion = completion
                padded_completions.append(padded_completion)
            
            # Stack all padded completions
            prompt_completion_ids = torch.cat(padded_completions, dim=0)
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]
            prompt_mask = prompt_mask.repeat_interleave(self.args.num_generations, dim=0)

        #################### Mask everything after the first EOS token
        eos_token_id = 151645
        is_eos = completion_ids == eos_token_id
        device = self.accelerator.device
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        # print(is_eos, completion_mask)
        
        #################### Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B*G, P+C)
        pixel_values = temp_inputs["pixel_values"].repeat_interleave(self.args.num_generations, dim=0)
        image_flags = temp_inputs["image_flags"].repeat_interleave(self.args.num_generations, dim=0)
        
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)

        per_token_logps = self._get_per_token_logps(model, prompt_completion_ids, attention_mask, pixel_values, position_ids, image_flags)
        # print("per_token_logps", per_token_logps)

        #################### Get rid of the prompt (-1 because of the shift done in get_per_token_logps)
        per_token_logps = per_token_logps[:, prompt_length - 1 :]

        with torch.inference_mode():
            # if self.ref_model is not None:
            #     ref_per_token_logps = self._get_per_token_logps(self.ref_model, prompt_completion_ids, attention_mask, pixel_values, position_ids, image_flags)
            # else:
            # with self.accelerator.unwrap_model(model):
            #     with model.language_model.disable_adapter():
            unwrap_model = self.accelerator.unwrap_model(model)
            with unwrap_model.language_model.disable_adapter():
                ref_per_token_logps = self._get_per_token_logps(unwrap_model, prompt_completion_ids, attention_mask, pixel_values, position_ids, image_flags)
            
        ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1 :]
        
        ##################### Compute the KL divergence between the model and the reference model
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
        # print("per_token_kl", per_token_kl)
        ##################### Decode the generated completions
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        # if is_conversational(inputs[0]):
        #     completions = [[{"role": "assistant", "content": completion}] for completion in completions]

        ##################### Compute the rewards
        prompts = self.processing_class.batch_decode(inputs["input_ids"], skip_special_tokens=True)
        prompts = [prompt for prompt in prompts for _ in range(self.args.num_generations)]

        rewards_per_func = torch.zeros(len(prompts), len(self.args.reward_funcs), device=device)
        for i, reward_func in enumerate(self.args.reward_funcs):
            output_reward_func = reward_func(prompts=prompts, completions=completions)
            rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)
        print("========================> Rewards_per_func",rewards_per_func)
        print("***"*100)
        ###################### Sum the rewards from all reward functions
        rewards = rewards_per_func.sum(dim=1)

        ###################### Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.args.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.args.num_generations).std(dim=1)

        ###################### Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.args.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.args.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        ###################### x - x.detach() allows for preserving gradients from x
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.args.beta * per_token_kl)
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        ###################### Log the metrics
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.args.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())

        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())

        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        return loss


    
    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()

