import os
import subprocess
import sys


def main():
    os.environ['FS_SEED'] = os.environ.get('FS_SEED', '42')
    os.environ['FS_SWEEP_KS'] = ''
    os.environ['FS_SWEEP_FRACS'] = '0.20'
    os.environ['FS_BENCHMARK'] = os.environ.get('FS_BENCHMARK', 'codet_m4')

    cmd = [sys.executable, 'Exp_FewShot/testing/exp_fs_inline_hier_ntk.py']
    raise SystemExit(subprocess.call(cmd))


if __name__ == '__main__':
    main()
