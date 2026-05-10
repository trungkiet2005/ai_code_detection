import subprocess
import sys


def main():
    cmd = [sys.executable, 'Exp_FewShot/aggregate_fs_results.py']
    raise SystemExit(subprocess.call(cmd))


if __name__ == '__main__':
    main()
