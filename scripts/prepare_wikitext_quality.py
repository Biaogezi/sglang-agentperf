"""Create a reproducible paragraph-NLL regression corpus (not standard WikiText PPL)."""

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
SHA256 = "5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer", help="Local tokenizer; emit fixed 128-token windows")
    args = parser.parse_args()
    source = Path(args.parquet)
    if hashlib.sha256(source.read_bytes()).hexdigest() != SHA256:
        raise ValueError("Unexpected dataset hash; refusing to change the regression corpus")
    paragraphs = [
        (idx, text.strip())
        for idx, text in enumerate(pq.read_table(source)["text"].to_pylist())
        if 500 <= len(text.strip()) <= 6000
    ]
    tokenizer_hashes = {}
    if args.tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
        paragraphs = [
            (idx, tokenizer.encode(text, add_special_tokens=False)) for idx, text in paragraphs
        ]
        paragraphs = [(idx, tokens[:128]) for idx, tokens in paragraphs if len(tokens) >= 128]
        for filename in ("tokenizer.json", "tokenizer_config.json"):
            tokenizer_hashes[filename] = hashlib.sha256(
                (Path(args.tokenizer) / filename).read_bytes()
            ).hexdigest()
    chosen = [paragraphs[i * (len(paragraphs) - 1) // 63] for i in range(64)]
    field = "input_ids" if args.tokenizer else "text"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(
            json.dumps({"id": f"wikitext2_test_row_{idx}", field: text}) + "\n"
            for idx, text in chosen
        ),
        encoding="utf-8",
    )
    metadata = {
        "dataset": "Salesforce/wikitext",
        "revision": REVISION,
        "split": "wikitext-2-raw-v1/test",
        "source_sha256": SHA256,
        "license": "CC-BY-SA-3.0",
        "documents": len(chosen),
        "row_indices": [idx for idx, _ in chosen],
        "selection": "64 evenly spaced qualifying paragraphs; length 500..6000 characters",
        "token_window": 128 if args.tokenizer else None,
        "tokenizer_sha256": tokenizer_hashes,
        "metric_scope": "independent paragraph prompt NLL; not standard concatenated WikiText perplexity",
        "corpus_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
