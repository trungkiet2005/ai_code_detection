# kaggle_D1_runall.py — ONE-CELL runner for the Sink-Class Law apparatus (D1).
# =============================================================================
# Paste this whole file into a single Kaggle GPU cell (T4/P100) after attaching:
#   - dataset "chiboiz/ai-detection-encoders"  -> unixcoder-base backbone
#   - dataset "chiboiz/codetm4"                 -> CoDET-M4 parquet
#   - dataset "chiboiz/ai-code-detection"       -> AICD-Bench (Task 2)
#   (these are the exact slugs used by ext_gptsniffer.py)
# It runs, IN ORDER: SMOKE all 3 experiments on both benches (cheap sanity),
# then FULL runs, then prints a compact summary. Each experiment writes
# results/<expNN_name>_results.json. Open those JSONs (or run make_law_figures.py)
# to get the 3 paper figures. Honors the memory rule: smoke every bench first,
# open each JSON and check it is not {"error":...} before trusting it.
# =============================================================================
import os, sys, json, glob, subprocess, time

HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
RES  = os.path.join(HERE, "results"); os.makedirs(RES, exist_ok=True)
SCRIPTS = ["exp160_leaveonegen.py", "exp161_ksweep.py", "exp162_chernoff.py"]
BENCHES = ["codet_m4", "aicd_t2"]

def run(script, env_extra, label):
    env = dict(os.environ); env.update({k: str(v) for k, v in env_extra.items()})
    t0 = time.time()
    print(f"\n{'='*70}\n[RUN] {label}  ::  {script}  env={env_extra}\n{'='*70}", flush=True)
    p = subprocess.run([sys.executable, os.path.join(HERE, script)], env=env,
                       capture_output=True, text=True)
    sys.stdout.write(p.stdout[-4000:]);  sys.stderr.write(p.stderr[-2000:])
    print(f"[done] {label} in {time.time()-t0:.0f}s rc={p.returncode}", flush=True)
    return p.returncode == 0

def check_json(pattern):
    """Open matching JSONs; flag any {'error':...} or empty. A file existing != success."""
    ok = True
    for f in glob.glob(os.path.join(RES, pattern)):
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception as e:
            print(f"  [BAD] {os.path.basename(f)} unreadable: {e}"); ok = False; continue
        recs = d if isinstance(d, list) else [d]
        if any(isinstance(r, dict) and "error" in r for r in recs):
            print(f"  [ERROR-RECORD] {os.path.basename(f)}"); ok = False
        else:
            print(f"  [ok] {os.path.basename(f)}  ({len(recs)} records)")
    return ok

# ---- Phase 1: SMOKE (both benches, all 3 scripts) --------------------------
print("\n########## PHASE 1: SMOKE ##########")
smoke_ok = True
for b in BENCHES:
    for s in SCRIPTS:
        smoke_ok &= run(s, {"SMOKE": 1, "BENCH": b}, f"smoke:{b}")
print("\n[smoke JSON check]"); check_json("*_results.json")
if not smoke_ok:
    print("\n!!! SMOKE FAILED — fix before full runs. Stopping."); sys.exit(1)

# ---- Phase 2: FULL runs -----------------------------------------------------
print("\n########## PHASE 2: FULL ##########")
# E1 sink-class + E2 collapse: run per bench; E3 Chernoff: per bench.
for b in BENCHES:
    run("exp160_leaveonegen.py", {"BENCH": b, "FRAC": 0.20}, f"E1:{b}")
    run("exp161_ksweep.py",      {"BENCH": b},               f"E2:{b}")
    run("exp162_chernoff.py",    {"BENCH": b},               f"E3:{b}")

print("\n[full JSON check]"); check_json("*_results.json")

# ---- Phase 3: compact summary ----------------------------------------------
print("\n########## PHASE 3: SUMMARY ##########")
def show(pattern, keys):
    for f in sorted(glob.glob(os.path.join(RES, pattern))):
        d = json.load(open(f, encoding="utf-8")); recs = d if isinstance(d, list) else [d]
        print(f"\n# {os.path.basename(f)}")
        for r in recs[:20]:
            if isinstance(r, dict):
                print("  " + "  ".join(f"{k}={r.get(k)}" for k in keys if k in r))
show("*leaveonegen*", ["bench", "held_out", "sink_rate", "dominant_sink_rate", "dominant_sink_class"])
show("*ksweep*",      ["bench", "K", "macro", "acc"])
show("*chernoff*",    ["bench", "C_min_chernoff", "C_min_symkl", "C_min_auroc"])
print("\n[ALL DONE] Next: run  python make_law_figures.py  (or the image axis) to draw the 3 figures.")
