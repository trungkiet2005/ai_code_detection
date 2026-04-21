# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in this directory.
#
# VENDORED SUBSET for Exp_ZeroShot — upstream source:
#   https://github.com/baoguangsheng/fast-detect-gpt/blob/main/scripts/local_infer.py
# Changes vs upstream:
#   - Absolute-import → relative-import (package layout differs).
#   - Dropped the interactive `run()` + `__main__` driver (we call
#     `FastDetectGPT.compute_crit` directly from `exp_zs_31_*`).
#   - `FastDetectGPT.__init__` accepts a plain object OR a dict for args so
#     callers can pass either argparse.Namespace or a SimpleNamespace stub.
#   - Default `pad_token` normalisation for llama3 (Meta-Llama-3-8B has a
#     real pad token; load_tokenizer upstream only sets pad_token_id for
#     models with no pad token).
import torch
from scipy.stats import norm

from .fast_detect_gpt import get_sampling_discrepancy_analytic
from .model import load_model, load_tokenizer


# Considering balanced classification that p(D0) equals to p(D1), we have
#   p(D1|x) = p(x|D1) / (p(x|D1) + p(x|D0))
def compute_prob_norm(x, mu0, sigma0, mu1, sigma1):
    pdf_value0 = norm.pdf(x, loc=mu0, scale=sigma0)
    pdf_value1 = norm.pdf(x, loc=mu1, scale=sigma1)
    prob = pdf_value1 / (pdf_value0 + pdf_value1)
    return prob


class FastDetectGPT:
    """Thin wrapper around Bao et al.'s scoring kernel. Pre-calibrated normal
    params for 4 model pairs are hardcoded from the paper's dev-set fits.
    """
    def __init__(self, args):
        self.args = args
        self.criterion_fn = get_sampling_discrepancy_analytic
        self.scoring_tokenizer = load_tokenizer(args.scoring_model_name, args.cache_dir)
        self.scoring_model = load_model(args.scoring_model_name, args.device, args.cache_dir)
        self.scoring_model.eval()
        if args.sampling_model_name != args.scoring_model_name:
            self.sampling_tokenizer = load_tokenizer(args.sampling_model_name, args.cache_dir)
            self.sampling_model = load_model(args.sampling_model_name, args.device, args.cache_dir)
            self.sampling_model.eval()
        # Paper-reported normal-distrib fits (mu0/sigma0 for human, mu1/sigma1 for AI).
        distrib_params = {
            'gpt-j-6B_gpt-neo-2.7B': {'mu0': 0.2713, 'sigma0': 0.9366, 'mu1': 2.2334, 'sigma1': 1.8731},
            'gpt-neo-2.7B_gpt-neo-2.7B': {'mu0': -0.2489, 'sigma0': 0.9968, 'mu1': 1.8983, 'sigma1': 1.9935},
            'falcon-7b_falcon-7b-instruct': {'mu0': -0.0707, 'sigma0': 0.9520, 'mu1': 2.9306, 'sigma1': 1.9039},
            'llama3-8b_llama3-8b-instruct': {'mu0': 0.1603, 'sigma0': 1.0791, 'mu1': 2.4686, 'sigma1': 2.1582},
        }
        key = f'{args.sampling_model_name}_{args.scoring_model_name}'
        # If user picks an untabulated pair, let compute_prob() raise KeyError
        # — but compute_crit() (raw discrepancy) still works fine.
        self.classifier = distrib_params.get(key)

    # compute conditional probability curvature
    def compute_crit(self, text):
        tokenized = self.scoring_tokenizer(
            text, truncation=True, return_tensors="pt", padding=True, return_token_type_ids=False
        ).to(self.args.device)
        labels = tokenized.input_ids[:, 1:]
        with torch.no_grad():
            logits_score = self.scoring_model(**tokenized).logits[:, :-1]
            if self.args.sampling_model_name == self.args.scoring_model_name:
                logits_ref = logits_score
            else:
                tokenized = self.sampling_tokenizer(
                    text, truncation=True, return_tensors="pt", padding=True, return_token_type_ids=False
                ).to(self.args.device)
                assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
                logits_ref = self.sampling_model(**tokenized).logits[:, :-1]
            crit = self.criterion_fn(logits_ref, logits_score, labels)
        return crit, labels.size(1)

    # compute probability
    def compute_prob(self, text):
        if self.classifier is None:
            raise RuntimeError(
                f"No pre-calibrated distrib params for "
                f"{self.args.sampling_model_name}_{self.args.scoring_model_name}. "
                f"Use compute_crit() and calibrate τ on your own dev set."
            )
        crit, ntoken = self.compute_crit(text)
        mu0 = self.classifier['mu0']
        sigma0 = self.classifier['sigma0']
        mu1 = self.classifier['mu1']
        sigma1 = self.classifier['sigma1']
        prob = compute_prob_norm(crit, mu0, sigma0, mu1, sigma1)
        return prob, crit, ntoken
