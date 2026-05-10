"""
Self-contained Kaggle runner: aggregate results after runs.
Run in Kaggle working dir where /kaggle/working/results contains JSON outputs.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import urllib.request
import os

RAW_URL = "https://raw.githubusercontent.com/trungkiet2005/ai_code_detection/master/Exp_FewShot/aggregate_fs_results.py"


def _download_script() -> str:
    with urllib.request.urlopen(RAW_URL, timeout=60) as resp:
        data = resp.read()
    tmp = tempfile.NamedTemporaryFile(prefix="kaggle_agg_", suffix=".py", delete=False)
    tmp.write(data)
    tmp.close()
    return tmp.name


def main() -> int:
    script_path = _download_script()
    try:
        return subprocess.call([sys.executable, script_path, "/kaggle/working/results"])
    finally:
        try:
            os.remove(script_path)
        except OSError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
