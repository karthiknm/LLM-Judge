#!/usr/bin/env python
"""Generate CNN/DailyMail summaries with multiple summarization models."""

import hashlib
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import torch
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


MODELS: List[str] = [
    "facebook/bart-large-cnn",
    "google/pegasus-cnn_dailymail",
    "Sachin21112004/distilbart-news-summarizer",
]

OUTPUT_PATH = Path("data/cnn_summaries.jsonl")
SAMPLES_PER_MODEL = 1000
SEED = 42
MIN_ARTICLE_WORDS = 120
MAX_INPUT_TOKENS = 1024
MIN_LENGTH = 40
MAX_LENGTH = 130
NUM_BEAMS = 4
LENGTH_PENALTY = 2.0
NO_REPEAT_NGRAM_SIZE = 3


def stable_article_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("abisee/cnn_dailymail", "1.0.0", split="train")
    indices = list(range(len(ds)))
    random.shuffle(indices)

    generation_params: Dict[str, object] = {
        "min_length": MIN_LENGTH,
        "max_length": MAX_LENGTH,
        "do_sample": False,
        "num_beams": NUM_BEAMS,
        "length_penalty": LENGTH_PENALTY,
        "no_repeat_ngram_size": NO_REPEAT_NGRAM_SIZE,
        "early_stopping": True,
    }

    with OUTPUT_PATH.open("w", encoding="utf-8") as fout:
        for model_name in MODELS:
            print(f"\nLoading model: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
            model.eval()

            written = 0
            seen_hashes = set()
            start_t = time.time()

            for idx in indices:
                if written >= SAMPLES_PER_MODEL:
                    break

                row = ds[idx]
                article = row["article"].strip()
                if len(article.split()) < MIN_ARTICLE_WORDS:
                    continue

                art_hash = stable_article_id(article)
                if art_hash in seen_hashes:
                    continue
                seen_hashes.add(art_hash)

                try:
                    inputs = tokenizer(
                        article,
                        return_tensors="pt",
                        truncation=True,
                        max_length=MAX_INPUT_TOKENS,
                    ).to(device)

                    with torch.no_grad():
                        summary_ids = model.generate(
                            **inputs,
                            **generation_params,
                        )

                    summary_text = tokenizer.decode(
                        summary_ids[0], skip_special_tokens=True
                    ).strip()
                    if not summary_text:
                        continue

                    record = {
                        "id": f"cnn_dm_train_{idx}_{written}_{model_name.replace('/', '_')}",
                        "dataset": "cnn_dailymail",
                        "dataset_config": "1.0.0",
                        "split": "train",
                        "source_index": idx,
                        "model": model_name,
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "original": article,
                        "summary": summary_text,
                        "source_tokens": int(inputs["input_ids"].shape[-1]),
                        "summary_tokens": int(summary_ids.shape[-1]),
                    }
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    written += 1

                    if written % 50 == 0:
                        elapsed = time.time() - start_t
                        print(
                            f"[{model_name}] generated {written}/{SAMPLES_PER_MODEL} "
                            f"(elapsed {elapsed:.1f}s)"
                        )
                except Exception as exc:
                    print(f"[WARN] model={model_name} idx={idx} failed: {exc}")
                    continue

            print(f"Completed {model_name}: {written} records written.")

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\nDone. Wrote JSONL to: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
