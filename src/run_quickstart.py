from __future__ import annotations

import argparse
import json
import os
import pickle

from io_utils import read_text, write_text
from rag import build_index, retrieve


def run_quickstart(
    docs: str, index_path: str, eval_path: str, query: str, k: int, summary_path: str
) -> None:
    os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)

    corpus = read_text(docs)
    index = build_index(corpus)
    with open(index_path, "wb") as f:
        pickle.dump(index, f)
    print(f"[1/3] Built index -> {index_path} (passages={len(index.passages)})")

    hits = retrieve(index, query, k=k)
    print(f'[2/3] Query: "{query}" (k={k})')
    for rank, (pid, score, passage) in enumerate(hits, start=1):
        print(f"  #{rank} score={score:.3f} passage_id={pid}")
        print(f"  {passage}")
        print("  " + "-" * 58)

    total = 0
    hit = 0
    recall = None
    if os.path.exists(eval_path):
        with open(eval_path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                q = row["query"]
                must = row["relevant_contains"]
                hits_q = retrieve(index, q, k=k)
                total += 1
                if any(must.lower() in p.lower() for (_, _, p) in hits_q):
                    hit += 1
        recall = hit / max(total, 1)
        print(f"[3/3] Recall@{k}: {recall:.3f} ({hit}/{total})")
    else:
        print(f"[3/3] Eval file missing, skipped: {eval_path}")

    lines = []
    lines.append("=== Final Results ===")
    lines.append(f"Index: {index_path} (passages={len(index.passages)})")
    lines.append(f"Query: {query}")
    if hits:
        pid, score, passage = hits[0]
        lines.append(f"Top-1: id={pid} score={score:.3f}")
        lines.append(f"Snippet: {passage[:200]}{'...' if len(passage) > 200 else ''}")
    if recall is not None:
        lines.append(f"Recall@{k}: {hit}/{total} = {recall:.3f}")

    summary_text = "\n".join(lines)
    print(summary_text)
    write_text(summary_path, summary_text)
    print(f"Summary written to {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run build -> query -> eval and print a summary.")
    parser.add_argument("--docs", default="data/docs.txt")
    parser.add_argument("--index", default="outputs/index.pkl")
    parser.add_argument("--eval", default="data/eval.jsonl")
    parser.add_argument("--q", default="What is reinforcement learning?")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--summary", default="outputs/quickstart_summary.txt")
    args = parser.parse_args()

    run_quickstart(args.docs, args.index, args.eval, args.q, args.k, args.summary)


if __name__ == "__main__":
    main()
