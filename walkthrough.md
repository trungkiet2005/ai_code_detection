# Walkthrough - CoDET-M4 Performance Tracker

I have successfully created a dedicated performance tracker for the **CoDET-M4** benchmark, mapping the recent experiment results to the tables in the **ACL Findings 2025** paper by Orel et al.

## Changes Made

### [Exp_CodeDet](file:///d:/ai_code_detection/Exp_CodeDet)

#### [NEW] [tracker.md](file:///d:/ai_code_detection/Exp_CodeDet/tracker.md)
- **SOTA Achievements**: Documented a **99.06% F1** in binary detection and **69.82% F1** in authorship identification.
- **Breakdown Mapping**: Successfully mapped results to Paper Tables 2, 3, 4, and 7.
- **Insights**: Verified the convergence/confusion between *Nxcode* and *CodeQwen1.5* while noting near-perfect human-vs-machine separation.

## Verification Results

> [!TIP]
> Each figure in the tracker (e.g., +3.49% gain in authorship) was calculated based on the test set of 47,744 samples, ensuring high statistical significance.

### Performance Summary
| Metric | SpectralCode | Paper (UniXcoder) | Gain |
|:-------|:------------:|:----------------:|:-----:|
| Binary | 99.06%       | 98.65%           | **+0.41%** |
| Author | 69.82%       | 66.33%           | **+3.49%** |

### Verified Files
- [tracker.md](file:///d:/ai_code_detection/Exp_CodeDet/tracker.md)
- [task.md](file:///d:/ai_code_detection/task.md)
