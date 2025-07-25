# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
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
import importlib
import math
import re
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from ..utils import PeftConfig, PeftType, transpose


def is_bnb_available():
    return importlib.util.find_spec("bitsandbytes") is not None


if is_bnb_available():
    import bitsandbytes as bnb


@dataclass
class DoraConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`~peft.Lora`].

    Args:
        r (`int`): Lora attention dimension
        target_modules (`Union[List[str],str]`): The names of the modules to apply Lora to.
        lora_alpha (`float`): The alpha parameter for Lora scaling.
        lora_dropout (`float`): The dropout probability for Lora layers.
        merge_weights (`bool`):
            Whether to merge the weights of the Lora layers with the base transformer model in `eval` mode.
        fan_in_fan_out (`bool`): Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        enable_lora ( `List[bool]`): Used with `lora.MergedLinear`.
        bias (`str`): Bias type for Lora. Can be 'none', 'all' or 'lora_only'
        modules_to_save (`List[str]`):List of modules apart from LoRA layers to be set as trainable
            and saved in the final checkpoint.
    """

    r: int = field(default=8, metadata={"help": "Lora attention dimension"})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lora."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    lora_alpha: int = field(default=None, metadata={"help": "Lora alpha"})
    lora_dropout: float = field(default=None, metadata={"help": "Lora dropout"})
    dora_simple: bool = field(
        default=True, metadata={"help": "Whether to apply simple dora ver to save up GPU memory"}
    )
    Wdecompose_target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to only tune the magnitude part"
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    merge_weights: bool = field(
        default=False, metadata={"help": "Merge weights of the original model and the Lora model"}
    )
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    enable_lora: Optional[List[bool]] = field(default=None, metadata={"help": "Used with `lora.MergedLinear`."})
    bias: str = field(default="none", metadata={"help": "Bias type for Lora. Can be 'none', 'all' or 'lora_only'"})
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )

    def __post_init__(self):
        self.peft_type = PeftType.DORA


class DoraModel(torch.nn.Module):
    """
    Creates Low Rank Adapter (Lora) model from a pretrained transformers model.

    Args:
        model ([`transformers.PreTrainedModel`]): The model to be adapted.
        config ([`LoraConfig`]): The configuration of the Lora model.

    Returns:
        `torch.nn.Module`: The Lora model.

    Example::

        >>> from transformers import AutoModelForSeq2SeqLM, LoraConfig >>> from peft import LoraModel, LoraConfig >>>
        config = LoraConfig(
            peft_type="LORA", task_type="SEQ_2_SEQ_LM", r=8, lora_alpha=32, target_modules=["q", "v"],
            lora_dropout=0.01, )
        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base") >>> lora_model = LoraModel(config, model)

    **Attributes**:
        - **model** ([`transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`LoraConfig`]): The configuration of the Lora model.
    """

    def __init__(self, config, model):
        super().__init__()
        self.peft_config = config
        self.model = model
        self._find_and_replace()
        mark_only_lora_as_trainable(self.model, self.peft_config.bias)
        self.forward = self.model.forward

    def _find_and_replace(self):
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        if loaded_in_8bit and not is_bnb_available():
            raise ImportError(
                "To use Lora with 8-bit quantization, please install the `bitsandbytes` package. "
                "You can install it with `pip install bitsandbytes`."
            )
        is_target_modules_in_base_model = False
        is_hf_device_map_available = hasattr(self.model, "hf_device_map")
        kwargs = {
            "r": self.peft_config.r,
            "lora_alpha": self.peft_config.lora_alpha,
            "lora_dropout": self.peft_config.lora_dropout,
            "fan_in_fan_out": self.peft_config.fan_in_fan_out,
            "merge_weights": (self.peft_config.merge_weights or self.peft_config.inference_mode)
            and not is_hf_device_map_available,
            "dora_simple": self.peft_config.dora_simple
        }
        key_list = [key for key, _ in self.model.named_modules()]
        for key in key_list:
            if isinstance(self.peft_config.target_modules, str):
                target_module_found = re.fullmatch(self.peft_config.target_modules, key)
            else:
                target_module_found = any(key.endswith(target_key) for target_key in self.peft_config.target_modules)

            if isinstance(self.peft_config.Wdecompose_target_modules, str):
                wdecompose_target_module_found = re.fullmatch(self.peft_config.Wdecompose_target_modules, key)
            elif self.peft_config.Wdecompose_target_modules == None:
                wdecompose_target_module_found = False
            else:
                wdecompose_target_module_found = any(key.endswith(target_key) for target_key in self.peft_config.Wdecompose_target_modules)


            if target_module_found:
                if not is_target_modules_in_base_model:
                    is_target_modules_in_base_model = True
                parent, target, target_name = self._get_submodules(key)
                bias = target.bias is not None
                if loaded_in_8bit and isinstance(target, bnb.nn.Linear8bitLt):
                    kwargs.update(
                        {
                            "has_fp16_weights": target.state.has_fp16_weights,
                            "memory_efficient_backward": target.state.memory_efficient_backward,
                            "threshold": target.state.threshold,
                            "index": target.index,
                        }
                    )
                    if self.peft_config.enable_lora is None:
                        new_module = Linear8bitLt(target.in_features, target.out_features, bias=bias, **kwargs)
                    else:
                        raise NotImplementedError
                elif isinstance(target, Conv1D):
                    new_module = Linear(target.nx, target.nf, bias=bias, **kwargs)
                elif isinstance(target, torch.nn.Linear) and self.peft_config.enable_lora is None:
                    new_module = Linear(target.in_features, target.out_features, bias=bias, **kwargs)
                elif self.peft_config.enable_lora is not None:
                    raise NotImplementedError
                self._replace_module(parent, target_name, new_module, target)

            elif wdecompose_target_module_found:
                if not is_target_modules_in_base_model:
                    is_target_modules_in_base_model = True
                parent, target, target_name = self._get_submodules(key)
                bias = target.bias is not None
                if loaded_in_8bit and isinstance(target, bnb.nn.Linear8bitLt):
                    kwargs.update(
                        {
                            "has_fp16_weights": target.state.has_fp16_weights,
                            "memory_efficient_backward": target.state.memory_efficient_backward,
                            "threshold": target.state.threshold,
                            "index": target.index,
                        }
                    )
                    if self.peft_config.enable_lora is None:
                        new_module = Linear8bitLt(target.in_features, target.out_features, bias=bias, **kwargs)
                    else:
                        raise NotImplementedError

                elif isinstance(target, torch.nn.Linear) and self.peft_config.enable_lora is None:
                    new_module = Linear(target.in_features, target.out_features, bias=bias, Wdecompose= True, **kwargs)
                elif self.peft_config.enable_lora is not None:
                    raise NotImplementedError
                self._replace_module(parent, target_name, new_module, target)

 
        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {self.peft_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def _get_submodules(self, key):
        parent = self.model.get_submodule(".".join(key.split(".")[:-1]))
        target_name = key.split(".")[-1]
        target = self.model.get_submodule(key)
        return parent, target, target_name

    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        new_module.weight = old_module.weight

        # 
        with torch.no_grad():
            try:
                magnitude = (torch.linalg.norm(new_module.weight.detach(),dim=1)).unsqueeze(1).detach()
                new_module.weight_m_wdecomp.weight.copy_(magnitude)
            except RuntimeError:
                magnitude = (torch.linalg.norm(new_module.weight.detach(),dim=0)).unsqueeze(1).detach()
                new_module.weight_m_wdecomp.weight.copy_(magnitude)
        

        if old_module.bias is not None:
            new_module.bias = old_module.bias
        if getattr(old_module, "state", None) is not None:
            new_module.state = old_module.state
            new_module.to(old_module.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if "lora_" in name or "weight_m_wdecomp" in name:
                module.to(old_module.weight.device)
  
    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    @property
    def modules_to_save(self):
        return None

    def get_peft_config_as_dict(self, inference: bool = False):
        config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(self.peft_config).items()}
        if inference:
            config["inference_mode"] = True
        return config

    def _set_adapter_layers(self, enabled=True):
        for module in self.model.modules():
            if isinstance(module, LoraLayer):
                module.disable_adapters = False if enabled else True

    def enable_adapter_layers(self):
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self):
        self._set_adapter_layers(enabled=False)


# Below code is based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# and modified to work with PyTorch FSDP


#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


# had to adapt it for `lora_only` to work
def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none") -> None:
    for n, p in model.named_parameters():
        if "lora_" not in n and "weight_m_wdecomp" not in n:
            p.requires_grad = False
        else:
            print(f"{n} is trainable")
    if bias == "none":
        return
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, LoraLayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


class LoraLayer:
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights
        self.disable_adapters = False


class Linear(nn.Linear, LoraLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        Wdecompose: bool = False,
        dora_simple: bool = True,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoraLayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)

        self.weight_m_wdecomp = nn.Linear(1,out_features,bias=False) # self.weight_m_wdecomp.weight # shape: out_features, 1

        self.fan_in_fan_out = fan_in_fan_out
        self.Wdecompose = Wdecompose # whether to tune only the magnitude component of Wdecompose or not
        self.dora_simple = dora_simple # whether to use dora simple to save up GPU memory
        if self.Wdecompose == False:
            if r > 0:
                self.lora_A = nn.Linear(in_features, r, bias=False)
                self.lora_B = nn.Linear(r, out_features, bias=False)
                self.scaling = self.lora_alpha / self.r
                # Freezing the pre-trained weight matrix

        self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        if self.Wdecompose == False:
            self.lora_A.train(mode)
            self.lora_B.train(mode)
        self.weight_m_wdecomp.train(mode)

        if not mode and self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.Wdecompose:
                norm_scale = ( self.weight_m_wdecomp.weight / (torch.linalg.norm(self.weight,dim=1)).unsqueeze(1) )
                weight = norm_scale * self.weight
                self.weight.data.copy_(weight.detach())
            else:
                if self.r > 0:
                    try:
                        new_weight_v = self.weight + transpose(self.lora_B.weight @ self.lora_A.weight, fan_in_fan_out=self.fan_in_fan_out) * self.scaling
                        weight = ( self.weight_m_wdecomp.weight / (torch.linalg.norm(new_weight_v,dim=1)).unsqueeze(1)) * new_weight_v
                        self.weight.data.copy_(weight.detach())
                    except RuntimeError:
                        new_weight_v = self.weight.T + transpose(self.lora_B.weight @ self.lora_A.weight, fan_in_fan_out=self.fan_in_fan_out) * self.scaling
                        weight = ( self.weight_m_wdecomp.weight / (torch.linalg.norm(new_weight_v,dim=1)).unsqueeze(1)) * new_weight_v
                        weight = weight.T
                    self.weight.data.copy_(weight.detach())
            self.merged = True


    def eval(self):
        nn.Linear.eval(self)
        if self.Wdecompose == False:
            self.lora_A.eval()
            self.lora_B.eval()
        self.weight_m_wdecomp.eval()


    def forward(self, x: torch.Tensor):
        previous_dtype = self.weight.dtype

        if self.disable_adapters:
            raise NotImplementedError
        
        elif self.Wdecompose and not self.merged:


            norm_scale = self.weight_m_wdecomp.weight.view(-1) / (torch.linalg.norm(self.weight,dim=1))

            org_result = (F.linear(x, transpose(self.weight, self.fan_in_fan_out)))

            result = org_result + (norm_scale-1) * (F.linear(self.lora_dropout(x), transpose(self.weight, self.fan_in_fan_out)))

            if not self.bias is None:
                    result += self.bias.view(1, -1).expand_as(result)

        elif self.r > 0 and not self.merged:
            lora_ab = self.lora_B.weight @ self.lora_A.weight
            if lora_ab.shape[0] == self.weight.shape[0]:
                new_weight_v = self.weight + lora_ab * self.scaling
                dim_switch_flag = False
            else:
                new_weight_v = self.weight + lora_ab.T * self.scaling
                dim_switch_flag = True

            if self.dora_simple:
                norm_scale = self.weight_m_wdecomp.weight.view(-1) / (torch.linalg.norm(new_weight_v,dim=int(not dim_switch_flag))).detach()  # dim is 1
            else:
                norm_scale = self.weight_m_wdecomp.weight.view(-1) / (torch.linalg.norm(new_weight_v,dim=int(not dim_switch_flag)))
            
            self.fan_in_fan_out = False if not dim_switch_flag else True
            org_result = (F.linear(x, transpose(self.weight, self.fan_in_fan_out)))

            dropout_x = self.lora_dropout(x)

            result = org_result + (norm_scale-1) * (F.linear(dropout_x, transpose(self.weight, self.fan_in_fan_out)))

            if not self.bias is None:
                result += self.bias.view(1, -1).expand_as(result)

            result += ( norm_scale * (self.lora_B(self.lora_A(dropout_x.to(self.lora_A.weight.dtype))))) * self.scaling
            
        else:
            if self.weight.shape[1] == x.shape[-1]:
                result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
            else:
                fan_in_fan_out = not self.fan_in_fan_out
                result = F.linear(x, transpose(self.weight, fan_in_fan_out), bias=self.bias)
        if result.dtype != previous_dtype:
            result = result.to(previous_dtype)

        return result


class MergedLinear(nn.Linear, LoraLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        enable_lora: List[bool] = [False],
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs,
    ):
        raise NotImplementedError

if is_bnb_available():

    class Linear8bitLt(bnb.nn.Linear8bitLt, LoraLayer):
        # Lora implemented in a dense layer
        def __init__(
            self,
            in_features,
            out_features,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            Wdecompose: bool = False,
            **kwargs,
        ):
            raise NotImplementedError

    class MergedLinear8bitLt(bnb.nn.Linear8bitLt, LoraLayer):
        # Lora implemented in a dense layer
        def __init__(
            self,
            in_features: int,
            out_features: int,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            enable_lora: List[bool] = [False],
            **kwargs,
        ):
            raise NotImplementedError