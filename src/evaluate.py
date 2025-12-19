from __future__ import annotations
import json
import argparse
import pickle
import os

from rag import retrieve

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="outputs/index.pkl")
    parser.add_argument("--eval", default="data/eval.jsonl")
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()

    if not os.path.exists(args.index):
        raise FileNotFoundError(f"Missing index: {args.index} (run build_index.py first)")

    with open(args.index, "rb") as f:
        index = pickle.load(f)

    total = 0
    hit = 0
    with open(args.eval, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            q = row["query"]
            must = row["relevant_contains"]
            hits = retrieve(index, q, k=args.k)
            total += 1
            if any(must.lower() in p.lower() for (_, _, p) in hits):
                hit += 1

    recall = hit / max(total, 1)
    print(f"Recall@{args.k}: {recall:.3f} ({hit}/{total})")

if __name__ == "__main__":
    main()
