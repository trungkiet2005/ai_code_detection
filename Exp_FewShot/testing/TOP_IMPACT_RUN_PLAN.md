Top-impact run order (FewShot, Kaggle T4)

Seed set recommendation:
  42, 3407, 2026, 777, 1234

Tier A (headline stability @5%):
  python exp_fs_hier_ntk_seedpack_frac05.py
  python exp_n07_conformal_seedpack_frac05.py

Tier B (strong testing baselines @5%):
  python exp_fs_hier_seedpack_frac05.py
  python exp_fs_ntkalign_seedpack_frac05.py
  python exp_fs_focal_seedpack_frac05.py
  python exp_fs_unixcoder_seedpack_frac05.py

Tier C (novel sanity top-2 backup):
  python exp_n14_sliced_seedpack_frac05.py

Example multi-seed loop in Kaggle shell:
  for S in 42 3407 2026 777 1234; do FS_SEED=$S python exp_fs_hier_ntk_seedpack_frac05.py; done

After runs:
  python Exp_FewShot/aggregate_fs_results.py results
