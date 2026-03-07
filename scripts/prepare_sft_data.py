#!/usr/bin/env python
"""Prepare cleaned SFT data from gpt5.jsonl for Qwen judge distillation."""

import json
import os
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = Path(os.getenv("SFT_INPUT_PATH", str(PROJECT_ROOT / "data" / "gpt5.jsonl")))
TRAIN_OUT = Path(os.getenv("SFT_TRAIN_PATH", str(PROJECT_ROOT / "data" / "sft_train.jsonl")))
VAL_OUT = Path(os.getenv("SFT_VAL_PATH", str(PROJECT_ROOT / "data" / "sft_val.jsonl")))
BAD_OUT = Path(os.getenv("SFT_BAD_PATH", str(PROJECT_ROOT / "data" / "sft_bad_rows.jsonl")))

VAL_RATIO = 0.1
SEED = 42
REQUIRED_KEYS = ("coherence", "consistency", "fluency", "relevance")


def parse_response_json(raw: str) -> dict | None:
    if not raw:
        return None
    raw = raw.strip()
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    for key in REQUIRED_KEYS:
        if key not in obj or not isinstance(obj[key], dict):
            return None
        score = obj[key].get("score")
        expl = obj[key].get("explanation")
        if not isinstance(score, int) or not (1 <= score <= 5):
            return None
        if not isinstance(expl, str) or not expl.strip():
            return None
    return obj


def to_chat_record(input_prompt: str, response_json: dict, rec_id: str) -> dict:
    return {
        "id": rec_id,
        "messages": [
            {"role": "user", "content": input_prompt},
            {
                "role": "assistant",
                "content": json.dumps(response_json, ensure_ascii=False),
            },
        ],
    }


def main() -> None:
    random.seed(SEED)
    TRAIN_OUT.parent.mkdir(parents=True, exist_ok=True)

    good_rows = []
    bad_rows = []

    with INPUT_PATH.open("r", encoding="utf-8") as fin:
        for line_num, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                bad_rows.append({"line_num": line_num, "reason": "invalid_jsonl_line"})
                continue

            rec_id = row.get("id")
            input_prompt = row.get("input_prompt", "")
            raw_resp = row.get("raw_response_text", "")

            if not rec_id or not input_prompt:
                bad_rows.append(
                    {
                        "line_num": line_num,
                        "id": rec_id,
                        "reason": "missing_id_or_prompt",
                    }
                )
                continue

            parsed = parse_response_json(raw_resp)
            if parsed is None:
                bad_rows.append(
                    {"line_num": line_num, "id": rec_id, "reason": "truncated_or_invalid_output"}
                )
                continue

            good_rows.append(to_chat_record(input_prompt, parsed, rec_id))

    random.shuffle(good_rows)
    n_val = int(len(good_rows) * VAL_RATIO)
    val_rows = good_rows[:n_val]
    train_rows = good_rows[n_val:]

    with TRAIN_OUT.open("w", encoding="utf-8") as ftrain:
        for row in train_rows:
            ftrain.write(json.dumps(row, ensure_ascii=False) + "\n")

    with VAL_OUT.open("w", encoding="utf-8") as fval:
        for row in val_rows:
            fval.write(json.dumps(row, ensure_ascii=False) + "\n")

    with BAD_OUT.open("w", encoding="utf-8") as fbad:
        for row in bad_rows:
            fbad.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Input rows scanned: {len(good_rows) + len(bad_rows)}")
    print(f"Valid rows: {len(good_rows)}")
    print(f"Filtered bad/truncated rows: {len(bad_rows)}")
    print(f"Train rows: {len(train_rows)} -> {TRAIN_OUT}")
    print(f"Val rows: {len(val_rows)} -> {VAL_OUT}")
    print(f"Bad rows log: {BAD_OUT}")


if __name__ == "__main__":
    main()
