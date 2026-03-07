#!/usr/bin/env python
"""Run inference with Qwen2.5-3B LoRA judge and save predictions to JSONL."""

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_ADAPTER_PATH = PROJECT_ROOT / "artifacts" / "qwen25_3b_lora_judge" / "final"
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "sft_val.jsonl"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "qwen25_3b_lora_test_preds.jsonl"
PROMPT_PATH = PROJECT_ROOT / "scripts" / "prompt.txt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--adapter-path", default=str(DEFAULT_ADAPTER_PATH))
    p.add_argument("--input-path", default=str(DEFAULT_INPUT_PATH))
    p.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    p.add_argument("--source-rows-path", default=str(PROJECT_ROOT / "data" / "train_data.json"))
    p.add_argument("--max-new-tokens", type=int, default=320)
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def build_input(base_prompt: str, article: str, summary: str) -> str:
    return (
        f"{base_prompt}\n\n"
        "This is the complete source article for which you will be evaluating:\n"
        f"{article}\n\n"
        "This is the LLM generated summary you are about to evaluate:\n"
        f"{summary}\n\n"
        "Return ONLY the final JSON object."
    )


def load_rows(path: Path) -> list[dict]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    return json.loads(path.read_text(encoding="utf-8"))


def extract_json(raw: str) -> dict | None:
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = raw[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def get_prompt_for_row(row: dict, base_prompt: str) -> str:
    if isinstance(row.get("messages"), list) and row["messages"]:
        for m in row["messages"]:
            if m.get("role") == "user":
                return m.get("content", "")

    article = row.get("original", "")
    summary = row.get("summary", "")
    if article and summary:
        return build_input(base_prompt, article, summary)

    raise ValueError("Row missing both `messages` and `original`+`summary`.")


def extract_article_summary_from_prompt(prompt: str) -> tuple[str | None, str | None]:
    a_marker = "This is the complete source article for which you will be evaluating:\n"
    s_marker = "\n\nThis is the LLM generated summary you are about to evaluate:\n"
    end_marker = "\n\nReturn ONLY the final JSON object."
    a_start = prompt.find(a_marker)
    s_start = prompt.find(s_marker)
    if a_start == -1 or s_start == -1:
        return None, None
    article = prompt[a_start + len(a_marker) : s_start]
    rem = prompt[s_start + len(s_marker) :]
    end_pos = rem.rfind(end_marker)
    summary = rem[:end_pos] if end_pos != -1 else rem
    return article.strip() or None, summary.strip() or None


def main() -> None:
    args = parse_args()
    base_prompt = PROMPT_PATH.read_text(encoding="utf-8")

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    adapter_path = Path(args.adapter_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = load_rows(input_path)
    source_map: dict[str, dict] = {}
    source_rows_path = Path(args.source_rows_path)
    if source_rows_path.exists():
        source_rows = load_rows(source_rows_path)
        source_map = {str(r.get("id")): r for r in source_rows if r.get("id")}

    if args.limit > 0:
        rows = rows[: args.limit]

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model.eval()

    total = len(rows)
    with output_path.open("w", encoding="utf-8") as fout:
        for i, row in enumerate(rows, start=1):
            prompt = get_prompt_for_row(row, base_prompt)
            messages = [{"role": "user", "content": prompt}]
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            new_tokens = outputs[0][inputs["input_ids"].shape[-1] :]
            pred_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            pred_json = extract_json(pred_text)

            out = {
                "idx": i - 1,
                "id": row.get("id", f"row_{i-1}"),
                "dataset": row.get("dataset"),
                "source_index": row.get("source_index"),
                "prediction_text": pred_text,
                "prediction_json": pred_json,
                "parse_ok": pred_json is not None,
            }
            rec_id = str(out["id"])
            source_row = source_map.get(rec_id)
            if source_row:
                out["dataset"] = out["dataset"] or source_row.get("dataset")
                out["source_index"] = out["source_index"] or source_row.get("source_index")
                out["original"] = source_row.get("original")
                out["summary"] = source_row.get("summary")
            else:
                article, summary = extract_article_summary_from_prompt(prompt)
                out["original"] = row.get("original") or article
                out["summary"] = row.get("summary") or summary

            if isinstance(row.get("messages"), list):
                for m in row["messages"]:
                    if m.get("role") == "assistant":
                        out["reference_text"] = m.get("content", "")
                        break
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")

            if i % 25 == 0 or i == total:
                print(f"[{i}/{total}] scored")

    print(f"Done. Wrote predictions to: {output_path}")


if __name__ == "__main__":
    main()
