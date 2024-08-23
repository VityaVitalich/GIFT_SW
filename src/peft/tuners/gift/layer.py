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

import math
import warnings
from typing import Any, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils.other import transpose

from .config import GIFTConfig

class BitNoiseQuant(nn.Module):
    '''
    Noise maker class. Used in GIFT Layer for addition of random variable to non-salient weights
    '''
    def __init__(
        self,
        out_features: int,
        bit: int,
        noise_type: str
    )-> None:

        super(BitNoiseQuant, self).__init__()
        self.out_features = out_features

        self.bit = torch.tensor(bit)
        self.noise_type = noise_type

        alpha = torch.ones((out_features, 1))
        self.alpha_scale = nn.Parameter(alpha)

    def compute_alpha_scale(self, quant_weight: torch.Tensor) -> None:
        w = quant_weight.data
        device = quant_weight.device
        alpha = self.alpha_scale
        bit = self.bit

        if alpha.device != device:
            alpha = alpha.to(device)
        if bit.device != device:
            bit = bit.to(device)

        out_features = w.shape[0]
        
        alpha = self._get_row_scale(w, bit)
        alpha = alpha.to(w.dtype)
        self.alpha_scale.data = alpha.reshape((out_features, 1))

    def _get_row_scale(self, w: torch.Tensor, bit: torch.Tensor, maxshrink: Optional[int] = 0.8, grid: Optional[int] = 100, norm: Optional[int] = 2):
        qmax = 2 ** (bit.detach() - 1) - 1
        qmin = -(2 ** (bit.detach() - 1))
        tmp = torch.zeros(w.shape[0], device=w.device)
        best = torch.full([w.shape[0]], float('inf'), device=w.device)

        wmin = torch.minimum(w.min(1)[0], tmp)
        wmax = torch.maximum(w.max(1)[0], tmp)

        wmax = torch.maximum(torch.abs(wmin), wmax)
        tmp = wmin < 0
        if torch.any(tmp):
            wmin[tmp] = -wmax[tmp]

        tmp = (wmax == 0)
        wmax[tmp] = +1

        alpha = wmax

        for i in range(int(maxshrink * grid)):
            p = 1 - i / grid 
            wmax1 = p * wmax

            delta1 = wmax1 / qmax

            #quantization
            q = torch.clamp(torch.round(w / delta1.unsqueeze(1)), qmin, qmax)
            #dequantization
            q = q * delta1.unsqueeze(1)

            q -= w
            q.abs_()
            q.pow_(norm)
            err = torch.sum(q, 1)
            tmp = err < best

            if torch.any(tmp):
                best[tmp] = err[tmp]
                alpha[tmp] = wmax1[tmp]

        return alpha

    def _compute_quant_noise(self, w: torch.Tensor, bit: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        delta = alpha / (2**(bit - 1) - 1)
        if self.noise_type == 'normal':
            noise = torch.randn_like(w, requires_grad=False) / 2
        elif self.noise_type == 'uniform':
            noise = torch.rand_like(w, requires_grad=False) - 0.5
        else:
            raise AttributeError(f'Unrecognized type of noise {self.noise_type}')

        w_rand = noise * delta

        if self.alpha_scale.requires_grad:  
            c1 = w >= alpha
            c2 = w <= -alpha     
            w_clipped = torch.where(c2, -alpha, w + w_rand)
            w_out = torch.where(c1, alpha, w_clipped)
        else:
            w_out = w + w_rand
        
        return w_out

    def quant_noise(self, quant_weight: torch.Tensor) -> torch.Tensor:

        w = quant_weight
        device = quant_weight.device
        alpha = self.alpha_scale
        bit = self.bit

        if alpha.device != device:
            alpha = alpha.to(device)
        if bit.device != device:
            bit = bit.to(device)

        w_out = self._compute_quant_noise(w, bit, alpha)

        return w_out

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        w = self.quant_noise(weight)
        return w

class GIFTLayer(BaseTunerLayer):
    def __init__(
        self,
        target: torch.nn.Linear,
        adapter_name: str,
        gift_config: GIFTConfig,
        outlier_ids: torch.Tensor, 
    ) -> None:
        super(GIFTLayer, self).__init__()

        self.base_layer = target
        self.in_features = target.in_features
        self.out_features = target.out_features
        self.outlier_cols_num = gift_config.outlier_num
        self.dtype = target.weight.data.dtype
        self.outlier_ids = outlier_ids
        self.add_noise = gift_config.add_noise
        self.merged_adapters = []
        assert len(self.outlier_ids) > 0, "List of outliers could not be empty"
        assert self.outlier_cols_num > 0, 'Outlier column number must be positive'

        self.q_in_features = self.in_features - self.outlier_cols_num
        self.fp_in_features = self.outlier_cols_num
        
        gift_q_weight = torch.rand((self.out_features, self.q_in_features), 
                                dtype=self.dtype)
        gift_fp_weight = torch.rand((self.out_features, self.fp_in_features), 
                                dtype=self.dtype)

        self.gift_q_weight = nn.Parameter(gift_q_weight)
        self.gift_fp_weight = nn.Parameter(gift_fp_weight)

        self.from_fp_Linear(self.base_layer, outlier_ids)
        if self.add_noise:
            self.add_quant_bitnoise_to_weight(
                bit=gift_config.layer_bits,
                noise_type=gift_config.noise_type,
                compute_quant_scale=gift_config.compute_quant_scale

            )

    @torch.no_grad()
    def set_mask(self, outlier_ids: torch.tensor) -> None:
        self.mask = torch.ones(self.in_features, 
                               dtype=torch.bool)
        self.mask[outlier_ids] = False

        col_ids = torch.arange(self.in_features, 
                               dtype=torch.int32)
        self.col_perm = torch.cat([col_ids[self.mask], 
                                   col_ids[~self.mask]])

        self.inv_col_perm = torch.zeros(self.col_perm.numel(), 
                                        dtype=self.col_perm.dtype)
        self.inv_col_perm[self.col_perm] = torch.arange(self.col_perm.numel(),
                                                        dtype=self.col_perm.dtype)       

    def from_fp_Linear(
        self,
        linear: torch.nn.Linear,
        outlier_ids: torch.Tensor,
    ) -> None:

        self.set_mask(outlier_ids)
        weight = torch.clone(linear.weight.data)
        self.gift_q_weight.data = weight[:, self.mask]
        self.gift_fp_weight.data = weight[:, ~self.mask]

        self.bias = nn.Parameter(torch.clone(linear.bias.data)) if linear.bias is not None else None


    def add_quant_bitnoise_to_weight(
        self, 
        bit,
        noise_type,
        compute_quant_scale,
    ) -> None:

        w = self.gift_q_weight

        self.noisemaker = BitNoiseQuant(
            out_features=w.shape[0],
            bit=bit,
            noise_type=noise_type
        )

        if compute_quant_scale:
            self.noisemaker.compute_alpha_scale(w)


class Linear(nn.Module, GIFTLayer):
    def __init__(
        self,
        base_layer: torch.nn.Linear,
        adapter_name: str,
        gift_config: GIFTConfig,
        outlier_ids: torch.Tensor,
        **kwargs,
    ) -> None:
        super().__init__()
        GIFTLayer.__init__(self, base_layer, adapter_name, gift_config, outlier_ids)

        self._active_adapter = adapter_name

        self._buffers["mask"] = self.mask
        self._buffers["col_perm"] = self.col_perm
        self._buffers["inv_col_perm"] = self.inv_col_perm
    
    def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        if self.merged:
            warnings.warn(
                f"Already following adapters were merged {','.join(self.merged_adapters)}. "
                f"You are now additionally merging {','.join(self.active_adapters)}."
            )

        if adapter_names is None:
            adapter_names = self.active_adapters

        for active_adapter in adapter_names:
            base_layer = self.get_base_layer()
            if safe_merge:
                # Note that safe_merge will be slower than the normal merge
                # because of the copy operation.
                orig_weights = base_layer.weight.data.clone()
                orig_weights[:, ~self.mask] = self.gift_fp_weight.data

                if not torch.isfinite(orig_weights).all():
                    raise ValueError(
                        f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                    )

                base_layer.weight.data = orig_weights
            else:
                base_layer.weight.data[:, ~self.mask] = self.gift_fp_weight
            self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        raise NotImplementedError('Method does not support unmerging')

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        
        if self.training:
            if self.add_noise:
                quant_w_noised = self.noisemaker(self.gift_q_weight)
            else:
                quant_w_noised = self.gift_q_weight

            out_w = torch.hstack([quant_w_noised, 
                                    self.gift_fp_weight])               
        else:
            out_w = torch.hstack([self.gift_q_weight, 
                                    self.gift_fp_weight])

        out_w = out_w[:, self.inv_col_perm]

        return F.linear(input, out_w, self.bias)   

def dispatch_default(
    target: torch.nn.Module,
    adapter_name: str,
    gift_config: GIFTConfig,
    outlier_ids: List[int],
    **kwargs,
) -> Optional[torch.nn.Module]:
    new_module = None

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    if isinstance(target_base_layer, torch.nn.Linear):
        new_module = Linear(target, adapter_name, gift_config, outlier_ids=outlier_ids)


    return new_module
