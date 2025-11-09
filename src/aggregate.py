import json, glob, statistics as stats
from pathlib import Path

def _collect(run_dir):
    with open(Path(run_dir) / "final_metrics.json", "r", encoding="utf-8") as f:
        return json.load(f)

def aggregate(runs_glob="runs/seed*"):
    paths = sorted(glob.glob(runs_glob))
    records = [_collect(p) for p in paths]
    def m(key): 
        vals = [r[key] for r in records]
        return float(stats.mean(vals)), (float(stats.pstdev(vals)) if len(vals) > 1 else 0.0)

    keys = ["test_accuracy", "balanced_accuracy", "f1_score"]
    summary = {k: {"mean": m(k)[0], "std": m(k)[1]} for k in keys}
    print("Aggregated over", len(records), "runs:", summary)
    return summary

if __name__ == "__main__":
    aggregate()
