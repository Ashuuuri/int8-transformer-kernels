#!/usr/bin/env python3
"""Download/prepare the real WikiText-2 corpus for validate_int8.py Gate 5.

Gate 5 measures real GPT-2 perplexity on real text held in
`testdata/real_corpus.txt` (the WikiText-2 *test* split, tokenized variant
`wikitext-2-v1`). That file lives under the gitignored `testdata/` tree, so a
fresh checkout must regenerate it with this script before running Gate 5:

    pip install --user datasets        # one-time, if missing
    python prepare_real_corpus.py

The reconstruction below is byte-exact with the original `wiki.test.tokens`:
the HF parquet "text" field keeps each content line's trailing " \\n" but
collapses blank lines to "" — the original blank line was a single space
followed by a newline, so we restore " \\n" for empty rows and pass content
rows through unchanged. (Verified md5 781c9418ab9414bd09e4aad825aae752.)

Gate 5 is a *relative* comparison (int8 vs fp16 on the same text), so even a
slightly different corpus would not change the pass/fail verdict — but the
byte-exact path keeps the documented absolute perplexity (fp16 31.946)
reproducible.
"""
import argparse
import os
import sys

OUT_PATH = os.path.join("testdata", "real_corpus.txt")
HF_REPO = "Salesforce/wikitext"
HF_CONFIG = "wikitext-2-v1"
SPLIT = "test"


def build_corpus() -> str:
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit(
            "error: the `datasets` package is required.\n"
            "  install it with:  pip install --user datasets\n"
            "  (Gate 5 already needs `transformers`; this adds the corpus loader.)"
        )
    ds = load_dataset(HF_REPO, HF_CONFIG, split=SPLIT)
    # Blank lines come back as "" from the parquet; the original tokens file
    # wrote them as " \n". Content rows already carry their trailing "\n".
    return "".join(" \n" if row == "" else row for row in ds["text"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true",
                    help="overwrite testdata/real_corpus.txt if it already exists")
    args = ap.parse_args()

    if os.path.exists(OUT_PATH) and not args.force:
        sz = os.path.getsize(OUT_PATH)
        print(f"[skip] {OUT_PATH} already exists ({sz} bytes). Use --force to rebuild.")
        return

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    text = build_corpus()
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[ok] wrote {OUT_PATH}  ({len(text.encode('utf-8'))} bytes, "
          f"{HF_REPO}/{HF_CONFIG} {SPLIT} split)")


if __name__ == "__main__":
    main()
