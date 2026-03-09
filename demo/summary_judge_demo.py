"""
Summary Judge Demo — Gradio Web Interface
==========================================
使用前请先填写下方路径配置：
  - BASE_MODEL  : Hugging Face model name 
  - ADAPTER_PATH: your local fine-tuned LoRA adapter path

运行方式：
    pip install torch transformers peft bitsandbytes gradio
    python summary_judge_demo.py
"""

import json
import re

import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


# ============================================================
# ✏️  FILL your path 
# ============================================================
BASE_MODEL   = "Qwen/Qwen2.5-3B-Instruct"   # model name
ADAPTER_PATH = ""                             # ← fill your own adpater path============


def load_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()
    return tokenizer, model


# ============================================================
# Prompt helpers
# ============================================================
SYSTEM_PROMPT = (
    "You are a summarization judge. "
    "Evaluate the summary against the source document on four dimensions: "
    "relevance, coherence, fluency, and consistency. "
    "Return ONLY a single valid JSON object with NO extra text before or after. "
    "The JSON must have exactly these keys: "
    "relevance, coherence, fluency, consistency, explanations. "
    "The first four must be integers 1-5. "
    "'explanations' must be a JSON object with exactly these string keys: "
    "relevance, coherence, fluency, consistency. "
    'Example format:\n'
    '{"relevance": 3, "coherence": 4, "fluency": 5, "consistency": 3, '
    '"explanations": {"relevance": "...", "coherence": "...", "fluency": "...", "consistency": "..."}}'
)


def make_user_prompt(doc, summary):
    return (
        f"Document:\n{doc}\n\n"
        f"Summary:\n{summary}\n\n"
        "Evaluate the summary and return JSON only."
    )


def safe_parse_json(text):
    if text is None:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r'\{.*\}', text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return None


def extract_scores_and_reasoning_regex(text):
    out = {
        "pred_relevance": None,
        "pred_coherence": None,
        "pred_fluency": None,
        "pred_consistency": None,
        "pred_relevance_expl": None,
        "pred_coherence_expl": None,
        "pred_fluency_expl": None,
        "pred_consistency_expl": None,
    }

    if text is None:
        return out

    score_patterns = {
        "pred_relevance":    r'"relevance"\s*:\s*([1-5])',
        "pred_coherence":    r'"coherence"\s*:\s*([1-5])',
        "pred_fluency":      r'"fluency"\s*:\s*([1-5])',
        "pred_consistency":  r'"consistency"\s*:\s*([1-5])',
    }
    for k, p in score_patterns.items():
        m = re.search(p, text)
        if m:
            out[k] = int(m.group(1))

    expl_block = None
    for pattern in [
        r'"?explanations"?\s*:?\s*\{(.*?)\}\s*$',
        r'"explanations"\s*:\s*\{(.*)',
        r'explanations\s*\{(.*)',
    ]:
        m_block = re.search(pattern, text, flags=re.DOTALL)
        if m_block:
            expl_block = m_block.group(1)
            break

    if expl_block is not None:
        expl_patterns = {
            "relevance":   r'"relevance"\s*:\s*"((?:[^"\\]|\\.)*)"',
            "coherence":   r'"coherence"\s*:\s*"((?:[^"\\]|\\.)*)"',
            "fluency":     r'"fluency"\s*:\s*"((?:[^"\\]|\\.)*)"',
            "consistency": r'"consistency"\s*:\s*"((?:[^"\\]|\\.)*)"',
        }
        for k, p in expl_patterns.items():
            m = re.search(p, expl_block, flags=re.DOTALL)
            if m:
                val = m.group(1)
                val = val.replace('\\"', '"').replace("\\n", "\n").replace("\\t", "\t")
                out[f"pred_{k}_expl"] = val.strip()

    return out


def judge_one_from_prompt(tokenizer, model, user_prompt, max_new_tokens=386):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    gen = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()
    return gen


# ============================================================
# Gradio UI
# ============================================================
def build_demo(tokenizer, model):
    def demo_fn(source_text, summary_text):
        p = make_user_prompt(source_text, summary_text)
        pred_text = judge_one_from_prompt(tokenizer, model, p, max_new_tokens=256)
        parsed = extract_scores_and_reasoning_regex(pred_text)
        return (
            parsed["pred_coherence"],
            parsed["pred_consistency"],
            parsed["pred_fluency"],
            parsed["pred_relevance"],
            parsed["pred_coherence_expl"],
            parsed["pred_consistency_expl"],
            parsed["pred_fluency_expl"],
            parsed["pred_relevance_expl"],
            pred_text,
        )

    with gr.Blocks() as demo:
        gr.Markdown("# Summary Judge Demo")

        with gr.Row():
            source_input  = gr.Textbox(label="Source Article",    lines=16)
            summary_input = gr.Textbox(label="Candidate Summary", lines=16)

        run_btn = gr.Button("Evaluate")

        with gr.Row():
            coherence_out    = gr.Number(label="Coherence")
            consistency_out  = gr.Number(label="Consistency")
            fluency_out      = gr.Number(label="Fluency")
            relevance_out    = gr.Number(label="Relevance")

        coherence_reason    = gr.Textbox(label="Coherence Reasoning",    lines=4)
        consistency_reason  = gr.Textbox(label="Consistency Reasoning",  lines=4)
        fluency_reason      = gr.Textbox(label="Fluency Reasoning",      lines=4)
        relevance_reason    = gr.Textbox(label="Relevance Reasoning",    lines=4)
        raw_out             = gr.Textbox(label="Raw Output",             lines=10)

        run_btn.click(
            fn=demo_fn,
            inputs=[source_input, summary_input],
            outputs=[
                coherence_out, consistency_out, fluency_out, relevance_out,
                coherence_reason, consistency_reason, fluency_reason, relevance_reason,
                raw_out,
            ],
        )

    return demo


if __name__ == "__main__":
    if not ADAPTER_PATH:
        raise ValueError("please fill adapter path ")

    print("Loading model...")
    tokenizer, model = load_model()
    print("Model loaded. Launching Gradio...")

    demo = build_demo(tokenizer, model)
    demo.launch(share=True)
