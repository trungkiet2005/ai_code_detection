import json
import hashlib
import time
from collections import Counter
from statistics import median
from urllib.parse import urlencode
from urllib.request import urlopen, Request
from urllib.error import HTTPError


def basic_code_stats(codes):
    if len(codes) == 0:
        return {
            "samples": 0,
            "chars_mean": 0.0,
            "chars_p50": 0.0,
            "chars_p95": 0.0,
            "lines_mean": 0.0,
            "lines_p50": 0.0,
            "lines_p95": 0.0,
            "non_ascii_ratio": 0.0,
            "empty_ratio": 0.0,
            "comment_like_ratio": 0.0,
            "tab_indent_ratio": 0.0,
            "near_duplicate_ratio": 0.0,
        }

    n = len(codes)
    char_lens = [len(c) for c in codes]
    line_lens = [c.count("\n") + 1 for c in codes]
    non_ascii = sum(any(ord(ch) > 127 for ch in c) for c in codes)
    empty = sum(len(c.strip()) == 0 for c in codes)

    # Lightweight style proxies
    comment_like = sum(1 for c in codes if ("#" in c or "//" in c or "/*" in c))
    tab_indent = sum(1 for c in codes if "\t" in c)

    # Near-duplicate proxy by short hash of normalized prefix
    fp = []
    for c in codes:
        x = " ".join(c.strip().split())[:512]
        fp.append(hashlib.md5(x.encode("utf-8", errors="ignore")).hexdigest())
    dup_ratio = 1.0 - (len(set(fp)) / n)

    return {
        "samples": len(codes),
        "chars_mean": float(sum(char_lens) / n),
        "chars_p50": float(median(char_lens)),
        "chars_p95": float(sorted(char_lens)[int(0.95 * (n - 1))]),
        "lines_mean": float(sum(line_lens) / n),
        "lines_p50": float(median(line_lens)),
        "lines_p95": float(sorted(line_lens)[int(0.95 * (n - 1))]),
        "non_ascii_ratio": float(non_ascii / n),
        "empty_ratio": float(empty / n),
        "comment_like_ratio": float(comment_like / n),
        "tab_indent_ratio": float(tab_indent / n),
        "near_duplicate_ratio": float(dup_ratio),
    }


def fetch_rows(dataset_id, config_name, split, sample_size, page_size=100):
    rows = []
    offset = 0
    while len(rows) < sample_size:
        length = min(page_size, sample_size - len(rows))
        query = urlencode(
            {
                "dataset": dataset_id,
                "config": config_name,
                "split": split,
                "offset": offset,
                "length": length,
            }
        )
        url = f"https://datasets-server.huggingface.co/rows?{query}"
        req = Request(url, headers={"User-Agent": "codeorigin-eda/1.0"})
        payload = None
        for attempt in range(6):
            try:
                with urlopen(req, timeout=60) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                break
            except HTTPError as e:
                if e.code == 429:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise
        if payload is None:
            break
        batch = [x["row"] for x in payload.get("rows", [])]
        if not batch:
            break
        rows.extend(batch)
        offset += len(batch)
        time.sleep(0.2)
        if len(batch) < length:
            break
    return rows


def run():
    report = {}

    # AICD-Bench EDA for T1
    aicd_splits = ["train", "validation", "test"]
    aicd = {}
    for split in aicd_splits:
        rows = fetch_rows("AICD-bench/AICD-Bench", "T1", split, sample_size=2000)
        codes = [r["code"] for r in rows]
        labels = [int(r["label"]) for r in rows]
        aicd[split] = {
            "label_counts": dict(Counter(labels)),
            "code_stats": basic_code_stats(codes),
        }
    report["aicd_t1_sample_2k"] = aicd

    # DroidCollection EDA
    droid_splits = ["train", "dev", "test"]
    droid = {}
    for split in droid_splits:
        rows = fetch_rows("project-droid/DroidCollection", "default", split, sample_size=2000)
        codes = [r["Code"] for r in rows]
        labels = [r.get("Label", "UNK") for r in rows]
        langs = [r.get("Language", "UNK") for r in rows]
        sources = [r.get("Source", "UNK") for r in rows]
        gen_modes = [r.get("Generation_Mode", "UNK") for r in rows]
        families = [r.get("Model_Family", "UNK") for r in rows]
        droid[split] = {
            "label_counts": dict(Counter(labels)),
            "top_languages": dict(Counter(langs).most_common(12)),
            "top_sources": dict(Counter(sources).most_common(12)),
            "top_generation_modes": dict(Counter(gen_modes).most_common(12)),
            "top_model_families": dict(Counter(families).most_common(12)),
            "code_stats": basic_code_stats(codes),
        }
    report["droid_sample_2k"] = droid

    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    run()
