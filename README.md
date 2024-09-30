<!---
Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

<h1 align="center"> <p>🤗 GIFT-SW with PEFT</p></h1>
<h3 align="center">
    <p>GIFT-SW with State-of-the-art Parameter-Efficient Fine-Tuning (PEFT) methods</p>
</h3>

This repository contains code for GIFT-SW method implemented with [PEFT library](https://huggingface.co/PEFT). It could be used in the same interface as usual PEFT methods and easily pluggable into any code.

PEFT is integrated with Transformers for easy model training and inference, Diffusers for conveniently managing different adapters, and Accelerate for distributed training and inference for really big models.

## Quickstart

Install PEFT directly from repository:

```bash
cd GIFT_SW/
pip install -e .
```

In case you have already installed PEFT, you will need to reinstall it:

```bash
cd GIFT_SW/
pip uninstall -y peft
pip install -e .
```

Get the activation scales for outlier computation from precomputed scales in [QUIK repository](https://github.com/IST-DASLab/QUIK/tree/master/experiments/act_scales) or by collecting them with script from [SmoothQuant](https://github.com/mit-han-lab/smoothquant)

Prepare a model for training with a GIFT-SW method by wrapping the base model and PEFT configuration with `get_peft_model`.

```python
from transformers import AutoModelForCausalLM
from peft import get_peft_config, get_peft_model, GIFTConfig, TaskType

model_name_or_path = "facebook/opt-1.3b"
tokenizer_name_or_path = "facebook/opt-1.3b"
path_to_act_scales = "./opt-1.3b.pt"

peft_config = GIFTConfig(
    outlier_num=64,
    target_modules=['q_proj', 'k_proj'],
    path_to_act_scales=path_to_act_scales,
)

model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
#"trainable params: 6,291,456 || all params: 1,517,182,976 || trainable%: 0.4147"
```

To save and later inference GIFT-SW model it is highly recommended to "merge_and_unload()" the model as GIFT-SW is not a regular adapter, but a learned subset of model weights. Further tuning of already trained GIFT-SW model is equivalent to merging the model and learning new one.


## How to get activations in SmoothQuant

To get the activation scales for your model you will need to get them with SmoothQuant method, it is simple and easy to use.

```bash
git clone https://github.com/mit-han-lab/smoothquant

# make sure the git-lfs is installed
# curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
# apt-get install git-lfs
# git lfs install

# clone the calibration data
git clone https://huggingface.co/datasets/mit-han-lab/pile-val-backup
# move to smoothquant and run the script

cd smoothquant
python examples/generate_act_scales.py \
    --model-name <your model name> \
    --output-path <save path .pt> \
    --num-samples 512 \ #number of calibration samples
    --seq-len 2048 \ #max sequence length
    --dataset-path ../pile-val-backup/val.jsonl.zst
```

