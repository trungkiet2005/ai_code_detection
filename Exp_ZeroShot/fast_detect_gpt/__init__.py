"""Vendored subset of Fast-DetectGPT (MIT, Bao Guangsheng 2023).

Source: https://github.com/baoguangsheng/fast-detect-gpt
Retained files: fast_detect_gpt.py (scoring kernels), model.py (load_{model,
tokenizer} + `model_fullnames` registry), local_infer.py (`FastDetectGPT`
class + pre-calibrated normal-distrib params).

All other files from upstream (data_builder, baselines, metrics, detect_gpt,
custom_datasets, *.sh) are intentionally dropped — we use our own
_zs_loaders + _zs_runner for dataset iteration and τ-calibration.

Exposes:
    from fast_detect_gpt import FastDetectGPT
    detector = FastDetectGPT(args_like_obj)
    crit, ntoken = detector.compute_crit(text)
    prob, crit, ntoken = detector.compute_prob(text)
"""
from .local_infer import FastDetectGPT, compute_prob_norm
from .fast_detect_gpt import (
    get_sampling_discrepancy,
    get_sampling_discrepancy_analytic,
    get_likelihood,
    get_samples,
)
from .model import load_model, load_tokenizer, model_fullnames, float16_models

__all__ = [
    "FastDetectGPT",
    "compute_prob_norm",
    "get_sampling_discrepancy",
    "get_sampling_discrepancy_analytic",
    "get_likelihood",
    "get_samples",
    "load_model",
    "load_tokenizer",
    "model_fullnames",
    "float16_models",
]
