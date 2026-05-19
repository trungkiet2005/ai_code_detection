# Round 1 Rebuttal Experiments

Nine self-contained Python files, each addressing one reviewer concern.
Run independently on Kaggle; each emits a JSON under
`Reviews/R1_responses/results/`.

| File | Reviewer concern | One-line answer (expected) |
|:--|:--|:--|
| `R1_01_treenoise.py` | W1/Q1: tree-noise sensitivity | Graceful degradation up to 25% edge corruption, full collapse at 100% |
| `R1_02_hpsens.py` | Q2: $\gamma$/$\tau$/$\lambda$ sweep | $\pm 0.005$ Macro-F1 plateau around $(1.0, 0.10, 0.5)$ |
| `R1_03_openset.py` | W4/Q6: leave-one-out generators | Held-out-class entropy AUC $\ge 0.75$; K-1 accuracy retained |
| `R1_04_modernbert.py` | W6/Q7: ModernBERT encoder | TRACO gains persist; relative lift over CE-only unchanged |
| `R1_05_perclass.py` | W7/Q3: AICD class breakdown | Macro-Weighted gap is class-imbalance; gains spread across classes |
| `R1_06_treedist.py` | W2/Q5: HUMAN-vs-AI distance | $\pm 0.005$ insensitivity band; recommend $d_{\mathrm{hum}}\!=\!2$ or $3$ |
| `R1_07_oodsrc.py` | W5: held-out-domain (CF $\to$ GH) | TRACO and CE drop similarly; tree-weighting orthogonal to source confound |
| `R1_08_augsem.py` | W3/Q4: augmentation semantic-rate | comment_strip/ws_jitter/id_rename 100% parse-safe; token_dropout 60% |
| `R1_09_hierce.py` | Q8: hierarchical CE baseline | Hier-CE lifts CE by $\sim 0.005$; contrastive form still beats |

## Running

Each file is independent. On Kaggle:

```bash
python Reviews/R1_responses/R1_01_treenoise.py
python Reviews/R1_responses/R1_02_hpsens.py
# ... etc
```

Outputs land in `Reviews/R1_responses/results/`.

## Compute budget

Rebuttal experiments are intentionally narrower than the main 6-slot
runs. Most files run 1-3 slots only. Total budget:

| File | Slots | GPU-hours |
|:--|:-:|:-:|
| R1_01_treenoise | 2 × 5 noise levels = 10 | ~5 |
| R1_02_hpsens | 1 × 5 × 4 × 4 = 80 quick runs | ~8 |
| R1_03_openset | 6 leave-out runs | ~6 |
| R1_04_modernbert | 6 slots | ~6 |
| R1_05_perclass | 0 (diagnostic only) | <0.1 |
| R1_06_treedist | 6 × 5 distances = 30 | ~12 |
| R1_07_oodsrc | 2 × 2 methods (TRACO + CE) | ~4 |
| R1_08_augsem | 0 (diagnostic only) | <0.1 |
| R1_09_hierce | 6 slots | ~6 |
| **Total** | | **~47 GPU-hours** |
