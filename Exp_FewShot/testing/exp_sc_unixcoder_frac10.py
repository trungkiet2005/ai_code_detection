"""
Self-contained Kaggle runner: FS-Baseline-UniXcoder @ frac=0.10
Upload this single file to Kaggle and run:
  python exp_sc_unixcoder_frac10.py
Optional env override:
  FS_SEED (default 42), FS_BENCHMARK (default codet_m4)
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import urllib.request

RAW_URL = "https://raw.githubusercontent.com/trungkiet2005/ai_code_detection/master/Exp_FewShot/testing/exp_fs_baseline_unixcoder.py"


def _download_script() -> str:
    with urllib.request.urlopen(RAW_URL, timeout=60) as resp:
        data = resp.read()
    tmp = tempfile.NamedTemporaryFile(prefix="kaggle_exp_", suffix=".py", delete=False)
    tmp.write(data)
    tmp.close()
    return tmp.name


def main() -> int:
    os.environ["FS_SEED"] = os.environ.get("FS_SEED", "42")
    os.environ["FS_SWEEP_KS"] = ""
    os.environ["FS_SWEEP_FRACS"] = "0.10"
    os.environ["FS_BENCHMARK"] = os.environ.get("FS_BENCHMARK", "codet_m4")

    script_path = _download_script()
    try:
        return subprocess.call([sys.executable, script_path])
    finally:
        try:
            os.remove(script_path)
        except OSError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
