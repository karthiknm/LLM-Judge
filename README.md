# LLM-Judge for Summarization

## Goal
Build and evaluate a student LLM judge for summarization quality via teacher distillation, under a tight timeline and low API budget.

## Core Idea
1. Generate synthetic summary data from news documents using local summarizers.
2. Use a strong teacher LLM API to produce rubric scores + concise reasoning.
3. Fine-tune a smaller open model (LoRA/SFT) on teacher outputs.
4. Evaluate only on human-annotated benchmarks (no benchmark training).

## Dataset Plan
- Synthetic training pool: `3,000` source documents total
  - `1,800` from CNN/DailyMail
  - `1,200` from XSum
- Benchmark test sets (held out, test-only):
  - SummEval (human alignment)
  - FRANK (factuality transfer)

## Synthetic Summary Generation
Use local/open summarizers for diversity (no API cost), e.g.:
- `facebook/bart-large-cnn`
- `google/pegasus-cnn_dailymail` or `google/pegasus-xsum`
- `sshleifer/distilbart-cnn-12-6` (or equivalent distilBART)

Assign one summarizer per source (randomized mix) to avoid style collapse.

## Teacher Annotation (API)
For each `(source, generated_summary)` pair, request strict JSON with:
- `coherence` (1-5)
- `consistency` (1-5)
- `fluency` (1-5)
- `relevance` (1-5)
- `overall` (1-5)
- `strict_factuality` (0/1)
- `confidence` (0-1)
- `rationale` (concise evidence-grounded explanation)

Start with 3k annotations; expand later only if needed.

## Student Fine-Tuning
- Base model: small open LLM (roughly 1B-8B)
- Method: LoRA SFT (e.g., Unsloth)
- Objective: next-token prediction on structured assistant output (JSON-style)
- Key ablation:
  - score-only targets
  - rationale + score targets

## Evaluation
1. **SummEval**: Spearman/Kendall correlation with human scores.
2. **FRANK**: factuality classification metrics (F1/accuracy).


## Expected Outcome
A practical, budget-aware LLM-judge pipeline showing whether a distilled student can approximate teacher/human judgment for summarization quality and factuality.
