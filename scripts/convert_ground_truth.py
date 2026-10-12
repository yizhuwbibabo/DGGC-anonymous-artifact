"""Convert official SANTOS / TUS ground-truth pickles into the JSON format
expected by src/dggc/evaluate.py: {"query_file.csv": ["positive_1.csv", ...]}.

Sources:
  SANTOS small/large: github.com/northeastern-datalab/santos
      (groundtruth/santosUnionBenchmark.pickle, santosIntentColumnBenchmark.pickle)
  TUS small/large:    github.com/RJMillerLab/table-union-search-benchmark

NOTE: verify the pickle structure against the benchmark release you actually
use before relying on the conversion. When the official ground truth is used,
reported recalls must stay within the published ceilings (R@10 <= 0.75 on
SANTOS-Small, R@60 <= 0.341 on TUS-Small, R@60 <= 0.277 on TUS-Large).
"""
import argparse
import json
import os
import pickle


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pickle", required=True, help="Official benchmark ground-truth pickle")
    ap.add_argument("--output", required=True, help="Output JSON path for evaluate.py --gt_path")
    args = ap.parse_args()
    with open(args.pickle, "rb") as f:
        raw = pickle.load(f)
    gt = {}
    for q, vals in raw.items():
        key = os.path.basename(str(q).strip())
        if isinstance(vals, (list, tuple, set)):
            gt[key] = sorted(os.path.basename(str(v).strip()) for v in vals)
        else:
            gt[key] = []
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(gt, f, indent=1)
    print(f"Wrote {len(gt)} queries to {args.output}")


if __name__ == "__main__":
    main()
