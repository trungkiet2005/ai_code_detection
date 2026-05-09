"""Generate 3 advanced inline method files (hier, focal, hier_ntk).

These extend exp_fs_inline_baseline.py with new loss functions:
  hier      -- HierTree family prior (Exp_13's secret sauce, proved +3.6pt)
  focal     -- Focal loss (fixes class-3 llama3.1 collapse F1=0.002)
  hier_ntk  -- HierTree + NTK alignment combo (likely strongest)

Pattern: read master baseline, inject new loss + replace dispatch, write.
"""
from __future__ import annotations

import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "exp_fs_inline_baseline.py")


HIER_LOSS_CODE = '''
# HierTree family map (codellama and nxcode are siblings; rest are singletons).
# Class indices come from sorted vocab: human=0, codellama=1, gpt=2, llama3.1=3,
# nxcode=4, qwen1.5=5. Verified by run logs: "Author vocab (5 generators):
# ['codellama', 'gpt', 'llama3.1', 'nxcode', 'qwen1.5']" with class 0 = human.
HIER_FAMILY = {0: 0, 1: 1, 2: 2, 3: 3, 4: 1, 5: 4}  # nxcode (4) shares family with codellama (1)
HIER_NUM_FAMILIES = 5


def hier_loss(outputs, labels, lambda_hier=0.4, margin=0.3, class_weights=None):
    """CE + family-aware pull/push.

    For each pair in batch:
      - same family -> pull (||z_i - z_j||^2)
      - different family -> push (max(0, m - ||z_i - z_j||)^2)
    Uses ntk_proj as the L2-normalised feature space.
    """
    ce = F.cross_entropy(outputs["logits"], labels, weight=class_weights)
    z = outputs["ntk_proj"]
    B = z.size(0)
    fam = torch.tensor([HIER_FAMILY.get(int(y), int(y)) for y in labels.cpu().tolist()],
                        device=z.device)
    same = (fam.unsqueeze(0) == fam.unsqueeze(1)).float()
    eye = torch.eye(B, device=z.device)
    same = same - eye  # exclude self-pairs
    diff = 1.0 - same - eye
    dist = torch.cdist(z, z, p=2)
    pull = (same * dist.pow(2)).sum() / same.sum().clamp(min=1.0)
    push = (diff * F.relu(margin - dist).pow(2)).sum() / diff.sum().clamp(min=1.0)
    hier_l = pull + push
    return {"total": ce + lambda_hier * hier_l, "ce": ce, "hier": hier_l, "pull": pull, "push": push}
'''

FOCAL_LOSS_CODE = '''
def focal_loss(outputs, labels, gamma=2.0, lambda_focal=1.0, class_weights=None):
    """Focal loss (Lin et al. ICCV'17): down-weight easy, up-weight hard.

    Loss = -alpha * (1-pt)^gamma * log(pt)
    Targets the persistent class-3 (llama3.1) collapse where F1 stays at
    ~0.002 because cross-entropy is dominated by easier classes. gamma=2.0
    is the standard, alpha is class_weights here.
    """
    logp = F.log_softmax(outputs["logits"], dim=-1)
    p = logp.exp()
    pt = p.gather(1, labels.unsqueeze(1)).squeeze(1)
    log_pt = logp.gather(1, labels.unsqueeze(1)).squeeze(1)
    weight = (1 - pt).clamp(min=1e-8) ** gamma
    if class_weights is not None:
        weight = weight * class_weights[labels]
    f = -(weight * log_pt).mean()
    return {"total": lambda_focal * f, "focal": f}
'''

HIER_NTK_LOSS_CODE = '''
HIER_FAMILY = {0: 0, 1: 1, 2: 2, 3: 3, 4: 1, 5: 4}
HIER_NUM_FAMILIES = 5


def hier_ntk_loss(outputs, labels, lambda_hier=0.4, lambda_ntk=0.4,
                   margin=0.3, class_weights=None):
    """CE + HierTree (family pull/push) + NTK target-kernel alignment.

    The 'best of both' candidate that most likely pushes past the
    NTKAlign 5% = 0.665 result by adding a hierarchy prior.
    """
    ce = F.cross_entropy(outputs["logits"], labels, weight=class_weights)
    z = outputs["ntk_proj"]
    B = z.size(0)

    # HierTree family pull/push
    fam = torch.tensor([HIER_FAMILY.get(int(y), int(y)) for y in labels.cpu().tolist()],
                        device=z.device)
    same = (fam.unsqueeze(0) == fam.unsqueeze(1)).float()
    eye = torch.eye(B, device=z.device)
    same = same - eye
    diff = 1.0 - same - eye
    dist = torch.cdist(z, z, p=2)
    pull = (same * dist.pow(2)).sum() / same.sum().clamp(min=1.0)
    push = (diff * F.relu(margin - dist).pow(2)).sum() / diff.sum().clamp(min=1.0)
    hier_l = pull + push

    # NTK target-kernel alignment
    K = z @ z.t()
    Y = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    H = torch.eye(B, device=z.device) - torch.full((B, B), 1.0 / B, device=z.device)
    align = ((H @ K @ H - H @ Y @ H) ** 2).mean()

    total = ce + lambda_hier * hier_l + lambda_ntk * align
    return {"total": total, "ce": ce, "hier": hier_l, "ntk_align": align}
'''


METHODS = [
    {
        "exp_id": "exp_fs_inline_hier",
        "method_name": "FS-HierTree",
        "loss_code": HIER_LOSS_CODE,
        "dispatch_branch": '''elif loss_kind == "hier":
        loss_fn = lambda o, l, class_weights=None: hier_loss(
            o, l, lambda_hier=hparams.get("lambda_hier", 0.4),
            margin=hparams.get("margin", 0.3), class_weights=class_weights)''',
        "method_key_value": "hier",
        "loss_kind": "hier",
        "freeze": False,
        "blurb": "HierTree family prior (codellama+nxcode siblings); Exp_13's secret sauce in 20%-data climb",
    },
    {
        "exp_id": "exp_fs_inline_focal",
        "method_name": "FS-Focal",
        "loss_code": FOCAL_LOSS_CODE,
        "dispatch_branch": '''elif loss_kind == "focal":
        loss_fn = lambda o, l, class_weights=None: focal_loss(
            o, l, gamma=hparams.get("gamma", 2.0),
            lambda_focal=1.0, class_weights=class_weights)''',
        "method_key_value": "focal",
        "loss_kind": "focal",
        "freeze": False,
        "blurb": "Focal loss (Lin et al. ICCV'17) -- target class-3 (llama3.1) collapse F1=0.002",
    },
    {
        "exp_id": "exp_fs_inline_hier_ntk",
        "method_name": "FS-Hier-NTK",
        "loss_code": HIER_NTK_LOSS_CODE,
        "dispatch_branch": '''elif loss_kind == "hier_ntk":
        loss_fn = lambda o, l, class_weights=None: hier_ntk_loss(
            o, l, lambda_hier=hparams.get("lambda_hier", 0.4),
            lambda_ntk=cfg.lambda_ntk, margin=hparams.get("margin", 0.3),
            class_weights=class_weights)''',
        "method_key_value": "hier_ntk",
        "loss_kind": "hier_ntk",
        "freeze": False,
        "blurb": "HierTree family prior + NTK target-kernel alignment combo (most likely to beat NTKAlign 5%=0.665)",
    },
]


def render(spec, src_text):
    new_doc = f'''"""\n{spec["exp_id"]}.py -- ONE-FILE self-contained inline for Kaggle T4.

Method: {spec["method_name"]} ({spec["blurb"]})

Paste into a Kaggle cell, run. No `git clone`, no other files needed.

Default sweep: K=128 + fraction in {{0.01, 0.05}} = 3 configs (~50 min on T4).
Override via FS_SWEEP_KS / FS_SWEEP_FRACS env vars.

Output: /kaggle/working/results/{spec["exp_id"]}_<label>_seed<S>.json
"""'''

    out = re.sub(r'^""".*?"""', new_doc, src_text, count=1, flags=re.DOTALL)

    # Inject the new loss function before the trainer section.
    inject_marker = "# =============================================================================\n# 6. Trainer"
    if inject_marker not in out:
        raise SystemExit("Could not find injection marker for loss code.")
    out = out.replace(inject_marker, spec["loss_code"].rstrip() + "\n\n\n" + inject_marker)

    # Add a new entry to METHODS dict.
    methods_marker = '"baseline":       ("FS-Baseline-CE",       "exp_fs_inline_baseline",       "ce",     False),'
    new_entry = (
        f'    {spec["method_key_value"]!r:<16}: '
        f'({spec["method_name"]!r}, {spec["exp_id"]!r}, {spec["loss_kind"]!r}, {spec["freeze"]}),'
    )
    out = out.replace(
        methods_marker,
        methods_marker + "\n" + new_entry,
    )

    # Inject dispatch elif in run_one().
    dispatch_anchor = '''    if loss_kind == "ce":
        loss_fn = lambda o, l, class_weights=None: cross_entropy_loss(o, l, class_weights)
    elif loss_kind == "ntk":'''
    out = out.replace(
        dispatch_anchor,
        f'''    if loss_kind == "ce":
        loss_fn = lambda o, l, class_weights=None: cross_entropy_loss(o, l, class_weights)
    {spec["dispatch_branch"]}
    elif loss_kind == "ntk":'''
    )

    # Hardcode METHOD_KEY (skip the env var read).
    out = out.replace(
        "method_key = 'baseline'",
        f"method_key = {spec['method_key_value']!r}",
    )

    # Update logger name to match new exp_id.
    out = re.sub(
        r"logger = logging\.getLogger\([^\)]+\)",
        f'logger = logging.getLogger({spec["exp_id"]!r})',
        out, count=1,
    )

    return out


def main():
    if not os.path.exists(SRC):
        raise SystemExit(f"missing source: {SRC}; run _generate_split_inline.py first")
    with open(SRC, "r", encoding="utf-8") as f:
        src_text = f.read()
    for spec in METHODS:
        out_path = os.path.join(HERE, f"{spec['exp_id']}.py")
        new_text = render(spec, src_text)
        with open(out_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(new_text)
        print(f"[gen] {out_path}  ({len(new_text)} bytes)")


if __name__ == "__main__":
    main()
