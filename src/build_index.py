# build_index.py
# This script builds an index from a text file and saves it to a pickle file.
from __future__ import annotations
import argparse
import pickle
import os

from io_utils import read_text, write_text
from rag import build_index

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", default="data/docs.txt")
    parser.add_argument("--out", default="outputs/index.pkl")
    args = parser.parse_args()

    os.makedirs("outputs", exist_ok=True)
    corpus = read_text(args.docs)
    index = build_index(corpus)

    with open(args.out, "wb") as f:
        pickle.dump(index, f)

    print(f"Saved index: {args.out}")
    print(f"Passages: {len(index.passages)}")

if __name__ == "__main__":
    main()
