# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in this directory.
#
# VENDORED SUBSET for Exp_ZeroShot — upstream source:
#   https://github.com/baoguangsheng/fast-detect-gpt/blob/main/scripts/model.py
# Changes vs upstream:
#   - Added `llama3-8b` / `llama3-8b-instruct` to `model_fullnames` (upstream
#     has them only in `local_infer.distrib_params`, which would crash on
#     load_model lookup for the recommended llama-3 pair).
#   - Dropped the `__main__` demo block.
#   - `load_tokenizer` signature accepts an optional ignored second positional
#     arg so callers written against either API (single-arg upstream newer,
#     two-arg upstream older) work.
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def from_pretrained(cls, model_name, kwargs, cache_dir):
    # use local model if it exists
    local_path = os.path.join(cache_dir, 'local.' + model_name.replace("/", "_"))
    if os.path.exists(local_path):
        return cls.from_pretrained(local_path, **kwargs)
    return cls.from_pretrained(model_name, **kwargs, cache_dir=cache_dir)


# Upstream model registry + our additions (llama3-8b pair).
model_fullnames = {
    'gpt2': 'gpt2',
    'gpt2-xl': 'gpt2-xl',
    'opt-2.7b': 'facebook/opt-2.7b',
    'gpt-neo-2.7B': 'EleutherAI/gpt-neo-2.7B',
    'gpt-j-6B': 'EleutherAI/gpt-j-6B',
    'gpt-neox-20b': 'EleutherAI/gpt-neox-20b',
    'mgpt': 'sberbank-ai/mGPT',
    'pubmedgpt': 'stanford-crfm/pubmedgpt',
    'mt5-xl': 'google/mt5-xl',
    'llama-13b': 'huggyllama/llama-13b',
    'llama2-13b': 'TheBloke/Llama-2-13B-fp16',
    'bloom-7b1': 'bigscience/bloom-7b1',
    'opt-13b': 'facebook/opt-13b',
    'falcon-7b': 'tiiuae/falcon-7b',
    'falcon-7b-instruct': 'tiiuae/falcon-7b-instruct',
    # Added: paper README (Jan 2026) recommends llama3 pair for best accuracy.
    'llama3-8b': 'meta-llama/Meta-Llama-3-8B',
    'llama3-8b-instruct': 'meta-llama/Meta-Llama-3-8B-Instruct',
}

float16_models = [
    'gpt-neo-2.7B', 'gpt-j-6B', 'gpt-neox-20b', 'llama-13b', 'llama2-13b',
    'bloom-7b1', 'opt-13b', 'falcon-7b', 'falcon-7b-instruct',
    'llama3-8b', 'llama3-8b-instruct',
]


def get_model_fullname(model_name):
    return model_fullnames[model_name] if model_name in model_fullnames else model_name


def load_model(model_name, device, cache_dir):
    model_fullname = get_model_fullname(model_name)
    print(f'Loading model {model_fullname}...')
    model_kwargs = {}
    if model_name in float16_models:
        model_kwargs.update(dict(torch_dtype=torch.float16))
    if 'gpt-j' in model_name:
        model_kwargs.update(dict(revision='float16'))
    model = from_pretrained(AutoModelForCausalLM, model_fullname, model_kwargs, cache_dir)
    print('Moving model to GPU...', end='', flush=True)
    start = time.time()
    model.to(device)
    print(f'DONE ({time.time() - start:.2f}s)')
    return model


def load_tokenizer(model_name, cache_dir=None, *_ignored):
    """Accepts optional trailing args for upstream-API compatibility."""
    if cache_dir is None:
        cache_dir = "./cache"
    model_fullname = get_model_fullname(model_name)
    optional_tok_kwargs = {}
    if "facebook/opt-" in model_fullname:
        print("Using non-fast tokenizer for OPT")
        optional_tok_kwargs['fast'] = False
    optional_tok_kwargs['padding_side'] = 'right'
    base_tokenizer = from_pretrained(AutoTokenizer, model_fullname, optional_tok_kwargs, cache_dir=cache_dir)
    if base_tokenizer.pad_token_id is None:
        base_tokenizer.pad_token_id = base_tokenizer.eos_token_id
        if '13b' in model_fullname:
            base_tokenizer.pad_token_id = 0
    return base_tokenizer
