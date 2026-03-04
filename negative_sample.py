import pandas as pd
from datasets import load_dataset

import re
import json
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd

# =========================
# Patterns / vocab
# =========================
CONNECTIVES = [
    "however", "therefore", "because", "although", "meanwhile", "moreover",
    "instead", "consequently", "despite", "thus"
]
PRONOUNS = ["he", "she", "they", "it", "this", "that"]
NUM_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
CAP_ENTITY_RE = re.compile(r"\b(?:[A-Z][a-z]+)(?:\s+[A-Z][a-z]+){0,3}\b")  # crude entity-ish
WORD_RE = re.compile(r"\b[A-Za-z]{4,}\b")

COMMON_TYPO_MAP = {
    "the": "teh", "and": "adn", "with": "wiht", "from": "form",
    "their": "thier", "because": "becuase", "government": "goverment",
    "people": "peopel", "report": "reprot", "increase": "increaes",
    "decrease": "decreaes", "public": "pubic",
}
import re
import random
from typing import List, Set, Optional, Tuple

CAP_ENTITY_RE = re.compile(r"\b(?:[A-Z][a-z]+)(?:\s+[A-Z][a-z]+){0,4}\b")

# filter obvious non-names / sentence starters
ENTITY_BLOCKLIST = {
    "The", "A", "An", "In", "On", "At", "As", "After", "Before", "Meanwhile",
    "However", "Therefore", "Because", "Although", "This", "That", "It", "They",
    "He", "She", "His", "Her", "Their", "Its", "Mr", "Mrs", "Ms", "Dr", "Sir",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December",
}
def clean_text(s: str) -> str:
    if not s:
        return s

    s = re.sub(r"\s+", " ", s).strip()

    # remove duplicated immediate tokens: "According According"
    s = re.sub(r"\b(\w+)\s+\1\b", r"\1", s)

    # fix spacing before punctuation
    s = re.sub(r"\s+([.,!?;:])", r"\1", s)

    # ensure comma after sentence-initial connectives if missing
    s = re.sub(r"^(However|Therefore|Because)\b(?!,)", r"\1,", s)

    # remove stray space before quotes
    s = re.sub(r'\s+"', '"', s)

    # collapse again
    s = re.sub(r"\s+", " ", s).strip()
    return s
def extract_entities(text: str) -> List[str]:
    """Heuristic: capitalized spans; returns unique in-order."""
    if not text:
        return []
    ents = []
    seen = set()
    for m in CAP_ENTITY_RE.finditer(text):
        span = m.group(0).strip()
        if span in ENTITY_BLOCKLIST:
            continue
        # avoid sentence-start singletons like "The", "This"
        if len(span.split()) == 1 and span in ENTITY_BLOCKLIST:
            continue
        if span not in seen:
            ents.append(span)
            seen.add(span)
    return ents

PERSON_NAME_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}\b")  # e.g., Sara Mellado

def pick_person_span(text: str, rng: random.Random):
    ms = list(PERSON_NAME_RE.finditer(text))
    if not ms:
        return None
    m = rng.choice(ms)
    return m.start(), m.end(), m.group(0)

def build_person_pool_from_summaries(summaries):
    pool = set()
    for s in summaries:
        for m in PERSON_NAME_RE.finditer(str(s)):
            pool.add(m.group(0))
    return sorted(pool)


def build_entity_pool(texts: List[str], min_len: int = 2) -> List[str]:
    pool: Set[str] = set()
    for t in texts:
        for e in extract_entities(t):
            if len(e) >= min_len:
                pool.add(e)
    return sorted(pool)

POLARITY_FLIPS = [
    ("increased", "decreased"),
    ("increase", "decrease"),
    ("decreased", "increased"),
    ("won", "lost"),
    ("win", "lose"),
    ("supports", "opposes"),
    ("support", "oppose"),
    ("approved", "rejected"),
    ("approve", "reject"),
    ("is", "is not"),
    ("was", "was not"),
    ("will", "will not"),
    ("can", "cannot"),
]

RELEVANCE_IRRELEVANT_SENTS = [
    "Officials declined to comment.",
    "The situation remains unclear.",
    "No additional information was released.",
    "The report did not provide further details."
]

CONSISTENCY_FALLBACK_FACTS = [
    "The incident happened in 2018.",
    "The announcement was made in London.",
    "The figure was reported as 70%.",
    "The decision was backed by the government."
]

# =========================
# Helpers
# =========================
def sent_split(text: str) -> List[str]:
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sents if s]

def pick_span(pattern: re.Pattern, text: str, rng: random.Random) -> Optional[Tuple[int, int, str]]:
    matches = list(pattern.finditer(text))
    if not matches:
        return None
    m = rng.choice(matches)
    return m.start(), m.end(), m.group(0)

def safe_replace_at(text: str, start: int, end: int, repl: str) -> str:
    return text[:start] + repl + text[end:]

def random_typo(word: str, rng: random.Random) -> str:
    lw = word.lower()
    if lw in COMMON_TYPO_MAP and rng.random() < 0.85:
        repl = COMMON_TYPO_MAP[lw]
        return repl if word.islower() else repl.capitalize()
    if len(word) >= 5:
        i = rng.randint(1, len(word) - 2)
        w = list(word)
        w[i], w[i + 1] = w[i + 1], w[i]
        return "".join(w)
    return word

def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

# =========================
# Logs
# =========================
@dataclass
class EditLog:
    rubric_target: str
    type: str
    before: str
    after: str
    note: str
    severity: str  # mild/medium/strong

# =========================
# Corruptions with "must-hit" fallback
# Each returns (text, logs)
# =========================
def corrupt_coherence(summary: str, rng: random.Random, severity: str) -> Tuple[str, List[EditLog]]:
    s = summary
    logs: List[EditLog] = []

    # (A) sentence-initial connective insert (always with comma)
    ins = rng.choice(["However,", "Therefore,", "Because,"])
    s = normalize_ws(ins + " " + s)
    logs.append(EditLog("coherence", "connective_insert", "", ins, "inserted connective causing jump", severity))

    # (B) swap adjacent sentences if there are >=2 (keeps punctuation)
    sents = sent_split(s)
    if len(sents) >= 3 and severity == "strong":
        i = rng.randrange(1, len(sents) - 1)  # avoid swapping the inserted connective sentence boundary too aggressively
        before = sents[i] + " " + sents[i + 1]
        sents[i], sents[i + 1] = sents[i + 1], sents[i]
        after = sents[i] + " " + sents[i + 1]
        s = " ".join(sents)
        logs.append(EditLog("coherence", "sentence_swap", before, after, "swapped adjacent sentences", severity))

    s = clean_text(s)
    return s, logs


    # s = summary
    # logs: List[EditLog] = []

    # # edits count by severity
    # target_edits = {"mild": 1, "medium": 2, "strong": 3}[severity]
    # done = 0

    # # (A) year/number shifts
    # while done < target_edits and rng.random() < 0.8:
    #     span = pick_span(YEAR_RE, s, rng) or pick_span(NUM_RE, s, rng)
    #     if not span:
    #         break
    #     a, b, before = span
    #     if YEAR_RE.fullmatch(before):
    #         year = int(before)
    #         delta = {"mild": rng.choice([-1, 1]),
    #                  "medium": rng.choice([-2, -1, 1, 2]),
    #                  "strong": rng.choice([-5, -3, -2, 2, 3, 5])}[severity]
    #         after = str(max(1900, min(2099, year + delta)))
    #         s = safe_replace_at(s, a, b, after)
    #         logs.append(EditLog("consistency", "year_shift", before, after, "shifted year", severity))
    #         done += 1
    #     else:
    #         val = float(before)
    #         factor = {"mild": rng.choice([0.8, 1.2]),
    #                   "medium": rng.choice([0.5, 0.7, 1.5, 2.0]),
    #                   "strong": rng.choice([0.25, 0.5, 2.0, 3.0, 5.0])}[severity]
    #         after = str(int(val * factor)) if val.is_integer() else str(round(val * factor, 2))
    #         s = safe_replace_at(s, a, b, after)
    #         logs.append(EditLog("consistency", "number_scale", before, after, "scaled number", severity))
    #         done += 1

    # # (B) entity swap
    #    # (B) entity/name swap using global pool first; fallback to doc/entities in text
    # while done < target_edits and rng.random() < 0.95:
    #     ent_span = pick_span(CAP_ENTITY_RE, s, rng)
    #     if not ent_span:
    #         break
    #     a, b, before = ent_span

    #     candidates = []

    #     # 1) global pool candidates (best)
    #     if entity_pool:
    #         candidates = [x for x in entity_pool if x != before]

    #     # 2) doc candidates
    #     if not candidates and doc:
    #         candidates = [m.group(0) for m in CAP_ENTITY_RE.finditer(doc)]
    #         candidates = [c for c in candidates if c != before]

    #     # 3) within-summary candidates
    #     if not candidates:
    #         candidates = [m.group(0) for m in CAP_ENTITY_RE.finditer(s)]
    #         candidates = [c for c in candidates if c != before]

    #     # 4) hard fallback: common plausible names
    #     if not candidates:
    #         candidates = ["John Smith", "Michael Brown", "Sarah Johnson", "David Miller", "Emily Davis"]

    #     after = rng.choice(candidates)

    #     # apply replace
    #     s = safe_replace_at(s, a, b, after)
    #     logs.append(EditLog("consistency", "entity_swap_pool", before, after, "replaced entity/name from pool", severity))
    #     done += 1

    # # (C) polarity flip
    # while done < target_edits and rng.random() < 0.9:
    #     before, after = rng.choice(POLARITY_FLIPS)
    #     pat = re.compile(rf"\b{re.escape(before)}\b", flags=re.IGNORECASE)
    #     m = pat.search(s)
    #     if not m:
    #         break
    #     repl = after if m.group(0).islower() else after.capitalize()
    #     s = safe_replace_at(s, m.start(), m.end(), repl)
    #     logs.append(EditLog("consistency", "polarity_flip", m.group(0), repl, "flipped polarity/relationship", severity))
    #     done += 1

    # # (D) fallback: must create at least 1 factual wrong-looking insert
    # if done == 0:
    #     ins = rng.choice(CONSISTENCY_FALLBACK_FACTS)
    #     before = ""
    #     after = ins
    #     # insert after first sentence if possible
    #     sents = sent_split(s)
    #     if len(sents) >= 1:
    #         s = normalize_ws(sents[0] + " " + ins + " " + " ".join(sents[1:]))
    #     else:
    #         s = normalize_ws(s + " " + ins)
    #     logs.append(EditLog("consistency", "fallback_insert_fact", before, after, "inserted fabricated fact", severity))

#     # return normalize_ws(s), logs
# def corrupt_consistency(summary: str, rng: random.Random, severity: str, person_pool=None):
#     s = summary
#     logs: List[EditLog] = []

#     # (1) 优先：换一个人名（2-3 token 的人名）
#     span = pick_person_span(s, rng)
#     if span:
#         a, b, before = span
#         candidates = [x for x in (person_pool or []) if x != before]
#         if not candidates:
#             candidates = ["John Smith", "Emily Davis", "Michael Brown", "Sarah Johnson"]
#         after = rng.choice(candidates)
#         s = safe_replace_at(s, a, b, after)
#         logs.append(EditLog("consistency", "person_name_swap", before, after, "replaced person name", severity))
#         s = clean_text(s)
#         return s, logs

#     # (2) 如果 summary 没有人名：只追加一次 attribution，而且避免 "According According"
#     # 若已经有 "According to" 就不追加
#     if re.search(r"\bAccording to\b", s):
#         # 改一个数字/年份（如果能找到）
#         span2 = pick_span(YEAR_RE, s, rng) or pick_span(NUM_RE, s, rng)
#         if span2:
#             a, b, before = span2
#             if YEAR_RE.fullmatch(before):
#                 year = int(before)
#                 after = str(max(1900, min(2099, year + rng.choice([2, 3, -2]))))
#                 s = safe_replace_at(s, a, b, after)
#                 logs.append(EditLog("consistency", "year_shift", before, after, "shifted year", severity))
#             else:
#                 val = float(before)
#                 after = str(int(val * 2)) if val.is_integer() else str(round(val * 2, 2))
#                 s = safe_replace_at(s, a, b, after)
#                 logs.append(EditLog("consistency", "number_scale", before, after, "scaled number", severity))
#         s = clean_text(s)
#         return s, logs

#     # 追加 attribution（只一次）
#     name = rng.choice((person_pool or []) or ["John Smith", "Emily Davis", "Michael Brown", "Sarah Johnson"])
#     s = normalize_ws(s + f' According to {name}, this was confirmed.')
#     logs.append(EditLog("consistency", "fallback_insert_person", "", name, "inserted fabricated person attribution", severity))
#     s = clean_text(s)
#     return s, logs
def corrupt_consistency(summary: str, rng: random.Random, severity: str, person_pool=None):
    s = summary
    logs: List[EditLog] = []

    # 1) Prefer swapping a real person name if present
    span = pick_person_span(s, rng)
    if span:
        a, b, before = span
        candidates = [x for x in (person_pool or []) if x != before]
        if not candidates:
            candidates = ["John Smith", "Emily Davis", "Michael Brown", "Sarah Johnson"]
        after = rng.choice(candidates)
        s = safe_replace_at(s, a, b, after)
        logs.append(EditLog("consistency", "person_name_swap", before, after, "replaced person name", severity))
        return clean_text(normalize_ws(s)), logs

    # 2) If no person name: add exactly one well-formed attribution sentence.
    # Avoid adding if one already exists.
    if re.search(r"\bAccording to\b", s):
        return clean_text(normalize_ws(s)), logs

    name = rng.choice((person_pool or []) or ["John Smith", "Emily Davis", "Michael Brown", "Sarah Johnson"])
    addition = f'According to {name}, this was confirmed.'
    s = normalize_ws(s + " " + addition)
    logs.append(EditLog("consistency", "fallback_insert_person", "", addition, "inserted fabricated attribution", severity))
    return clean_text(s), logs
def corrupt_fluency(summary: str, rng: random.Random, severity: str) -> Tuple[str, List[EditLog]]:
    s = summary
    logs: List[EditLog] = []
    target_edits = {"mild": 2, "medium": 4, "strong": 6}[severity]  # fluency needs multiple small errors
    done = 0

    words = s.split()
    if not words:
        return s, [EditLog("fluency", "empty", "", "", "empty summary", severity)]

    # (A) typos
    while done < target_edits and rng.random() < 0.95:
        # pick a word-ish token
        m = list(WORD_RE.finditer(s))
        if not m:
            break
        w = rng.choice(m)
        before = w.group(0)
        after = random_typo(before, rng)
        s = safe_replace_at(s, w.start(), w.end(), after)
        logs.append(EditLog("fluency", "typo", before, after, "introduced typo", severity))
        done += 1

    # (B) duplicate word
    if done < target_edits:
        toks = s.split()
        if len(toks) >= 6:
            idx = rng.randrange(1, len(toks) - 1)
            before = toks[idx]
            toks.insert(idx, toks[idx])
            after = f"{before} {before}"
            s = " ".join(toks)
            logs.append(EditLog("fluency", "dup_word", before, after, "duplicated a word", severity))
            done += 1

    # (C) punctuation damage
    if done < target_edits:
        m = re.search(r"[.,;:]", s)
        if m:
            before = s[m.start():m.end()]
            s = safe_replace_at(s, m.start(), m.end(), "")
            logs.append(EditLog("fluency", "punct_drop", before, "", "dropped punctuation", severity))
            done += 1

    # fallback: ensure at least 2 fluency edits
    if len(logs) < 2:
        s = normalize_ws(s + " .")
        logs.append(EditLog("fluency", "fallback_punct", "", ".", "forced punctuation oddity", severity))

    return normalize_ws(s), logs


# def corrupt_relevance(summary: str, doc: Optional[str], rng: random.Random, severity: str) -> Tuple[str, List[EditLog]]:
#     s = summary
#     logs: List[EditLog] = []
#     target_edits = {"mild": 1, "medium": 2, "strong": 3}[severity]
#     done = 0

#     # (A) delete lead (stronger => delete more)
#     toks = s.split()
#     if toks and rng.random() < 0.85 and done < target_edits:
#         if len(toks) >= 12:
#             if severity == "mild":
#                 k = max(3, len(toks) // 8)
#             elif severity == "medium":
#                 k = max(5, len(toks) // 4)
#             else:
#                 k = max(8, len(toks) * 2 // 5)
#             before = " ".join(toks[:k])
#             s = " ".join(toks[k:])
#             logs.append(EditLog("relevance", "delete_lead", before, "", "deleted leading content", severity))
#             done += 1

#     # (B) add irrelevant sentence (stronger: add 2)
#     while done < target_edits and rng.random() < 0.95:
#         irr = None
#         if doc:
#             ds = sent_split(doc)
#             if ds:
#                 irr = rng.choice(ds)
#                 irr_toks = irr.split()
#                 if len(irr_toks) > 18:
#                     irr = " ".join(irr_toks[:rng.randint(10, 16)]) + "."
#         if not irr:
#             irr = rng.choice(RELEVANCE_IRRELEVANT_SENTS)

#         s = normalize_ws(s + " " + irr)
#         logs.append(EditLog("relevance", "add_irrelevant", "", irr, "added irrelevant/low-signal sentence", severity))
#         done += 1
#         if severity == "mild":
#             break

#     # fallback
#     if not logs:
#         irr = rng.choice(RELEVANCE_IRRELEVANT_SENTS)
#         s = normalize_ws(s + " " + irr)
#         logs.append(EditLog("relevance", "fallback_add_irrelevant", "", irr, "forced relevance drop", severity))

#     return normalize_ws(s), logs
# def corrupt_relevance(summary: str, doc: Optional[str], rng: random.Random, severity: str) -> Tuple[str, List[EditLog]]:
#     s = summary
#     logs: List[EditLog] = []

#     # 强制：strong 只做 2 个 relevance 操作：delete_lead + add_irrelevant(最多1句)
#     # 但 add_irrelevant 只加固定句，避免 "class." 这种碎片
#     toks = s.split()

#     # (A) delete lead (only if long enough)
#     if len(toks) >= 12:
#         k = max(8, len(toks) * 2 // 5) if severity == "strong" else max(5, len(toks) // 4)
#         before = " ".join(toks[:k])
#         s = " ".join(toks[k:])
#         logs.append(EditLog("relevance", "delete_lead", before, "", "deleted leading content", severity))

#     # (B) add at most ONE irrelevant sentence (fixed set only)
#     irr = rng.choice(RELEVANCE_IRRELEVANT_SENTS)
#     s = normalize_ws(s + " " + irr)
#     logs.append(EditLog("relevance", "add_irrelevant", "", irr, "added irrelevant/low-signal sentence", severity))

#     s = clean_text(s)
#     return s, logs
def corrupt_relevance(summary: str, doc: Optional[str], rng: random.Random, severity: str) -> Tuple[str, List[EditLog]]:
    s = summary
    logs: List[EditLog] = []

    toks = s.split()
    if len(toks) >= 12:
        k = max(8, len(toks) * 2 // 5)  # strong
        before = " ".join(toks[:k])
        s = " ".join(toks[k:])
        logs.append(EditLog("relevance", "delete_lead", before, "", "deleted leading content", severity))

    irr = rng.choice(RELEVANCE_IRRELEVANT_SENTS)
    s = normalize_ws(s + " " + irr)
    logs.append(EditLog("relevance", "add_irrelevant", "", irr, "added irrelevant/low-signal sentence", severity))

    return clean_text(s), logs


# =========================
# Orchestration
# =========================
# def corrupt_one(summary, doc, rng, severity, rubric_plan=None, entity_pool=None)-> Tuple[str, List[EditLog], List[str]]:
#     """
#     severity: mild / medium / strong
#     rubric_plan: which rubrics to apply, else auto based on severity
#     returns: corrupted_summary, logs, rubrics_applied
#     """
#     if rubric_plan is None:
#         # Strong: hit 3-4 rubrics; Medium: 2 rubrics; Mild: 1 rubric
#         if severity == "strong":
#             k = rng.choice([3, 4])
#         elif severity == "medium":
#             k = 2
#         else:
#             k = 1
#         rubrics = ["consistency", "coherence", "fluency", "relevance"]
#         rng.shuffle(rubrics)
#         rubric_plan = rubrics[:k]

#     s = summary
#     logs: List[EditLog] = []
#     applied: List[str] = []

#     for r in rubric_plan:
#         if r == "consistency":
#             s, l = corrupt_consistency(s, doc, rng, severity, entity_pool=entity_pool)
#         elif r == "coherence":
#             s, l = corrupt_coherence(s, rng, severity)
#         elif r == "fluency":
#             s, l = corrupt_fluency(s, rng, severity)
#         elif r == "relevance":
#             s, l = corrupt_relevance(s, doc, rng, severity)
#         else:
#             continue
#         logs.extend(l)
#         applied.append(r)

#     # validation: must have at least 1 edit log, and corrupted != original (after normalization)
#     if not logs or normalize_ws(s) == normalize_ws(summary):
#         # force minimal fluency + relevance
#         s2, l2 = corrupt_fluency(s, rng, severity)
#         s3, l3 = corrupt_relevance(s2, doc, rng, severity)
#         s = s3
#         logs.extend(l2 + l3)
#         applied.extend(["fluency", "relevance"])

#     # never empty
#     if not s.strip():
#         s = summary
#         logs.append(EditLog("relevance", "fallback_restore", "", "", "corruption emptied summary; restored original", severity))

#     return normalize_ws(s), logs, applied

def corrupt_one(summary: str, doc: str, rng: random.Random, severity: str, person_pool=None):
    s = summary
    all_logs = []
    applied = []

    # 1) relevance delete_lead (will delete the start)
    s, l = corrupt_relevance(s, doc, rng, severity)
    all_logs += l; applied.append("relevance")

    # 2) consistency name swap (after deletion so you still change remaining names)
    s, l = corrupt_consistency(s, rng, severity, person_pool=person_pool)
    all_logs += l; applied.append("consistency")

    # 3) optional: add fluency + coherence to make it even worse
    s, l = corrupt_fluency(s, rng, severity)
    all_logs += l; applied.append("fluency")

    s, l = corrupt_coherence(s, rng, severity)
    all_logs += l; applied.append("coherence")

    return normalize_ws(s), all_logs, applied


def assign_severities(
    n: int,
    rng: random.Random,
    target_counts: Dict[str, int]
) -> List[str]:
    """
    Creates a length-n list of severities with exact counts if possible.
    If n != sum(counts), scales down/up proportionally but keeps exact integers.
    """
    total = sum(target_counts.values())
    if total == 0:
        return ["mild"] * n

    if n == total:
        sev = (["strong"] * target_counts["strong"] +
               ["medium"] * target_counts["medium"] +
               ["mild"] * target_counts["mild"])
        rng.shuffle(sev)
        return sev

    # scale counts to n (largest remainder)
    scaled = {k: target_counts[k] * n / total for k in target_counts}
    base = {k: int(scaled[k]) for k in scaled}
    remainder = n - sum(base.values())
    # distribute remainder
    order = sorted(target_counts.keys(), key=lambda k: (scaled[k] - base[k]), reverse=True)
    for i in range(remainder):
        base[order[i % len(order)]] += 1

    sev = (["strong"] * base.get("strong", 0) +
           ["medium"] * base.get("medium", 0) +
           ["mild"] * base.get("mild", 0))
    rng.shuffle(sev)
    return sev

import json
import random
from typing import Optional, Any, Dict, List
import pandas as pd

def generate_corrupted_dataset(
    input_path: str,
    output_path: str,
    log_path: str,
    *,
    id_col: str = "id",
    doc_col: Optional[str] = "document",
    summary_col: str = "summary",
    seed: int = 7,
) -> None:
    """
    Corrupt EVERY row with severity='strong'.
    Keeps original columns and adds: severity, corrupted_summary, corruption_json.
    Writes JSONL log with severity='strong' for each example.
    """

    # load
    if input_path.endswith(".csv"):
        df = pd.read_csv(input_path)
        fmt = "csv"
    elif input_path.endswith(".jsonl"):
        df = pd.read_json(input_path, lines=True)
        fmt = "jsonl"
    else:
        raise ValueError("input_path must end with .csv or .jsonl")

    if id_col not in df.columns:
        df[id_col] = [f"row_{i}" for i in range(len(df))]

    if summary_col not in df.columns:
        raise ValueError(f"Missing required column: {summary_col}")
    # build entity pool from ALL summaries (or documents too)
    all_summaries = [str(x) for x in df[summary_col].fillna("").tolist()]
    entity_pool = build_entity_pool(all_summaries)

# optionally expand pool with documents
# if has_doc: entity_pool = sorted(set(entity_pool) | set(build_entity_pool(df[doc_col].fillna("").tolist())))

    has_doc = (doc_col is not None) and (doc_col in df.columns)

    corrupted_summaries: List[str] = []
    corruption_jsons: List[str] = []
    severities_out: List[str] = []
    logs_out: List[Dict[str, Any]] = []
    person_pool = build_person_pool_from_summaries(df[summary_col].fillna("").tolist())
    for i, row in df.iterrows():
        rid = row[id_col]
        summ = str(row[summary_col]) if pd.notna(row[summary_col]) else ""
        doc = str(row[doc_col]) if has_doc and pd.notna(row[doc_col]) else None

        severity = "strong"
        severities_out.append(severity)

        # deterministic per-row RNG
        row_seed = (seed * 1_000_003) ^ (hash(str(rid)) & 0xFFFFFFFF)
        row_rng = random.Random(row_seed)

        # IMPORTANT: this calls your existing corrupt_one(...) from earlier
        # corr, edit_logs, rubrics_applied = corrupt_one(
        # summ, doc, row_rng, severity=severity, rubric_plan=None, entity_pool=entity_pool
        # )
    
# inside your per-row loop:
        severity = "strong"
        corr, edit_logs, rubrics_applied = corrupt_one(
            summ, doc if has_doc else None, row_rng, severity=severity, person_pool=person_pool
        )

        corrupted_summaries.append(corr)
        corruption_jsons.append(json.dumps([asdict(e) for e in edit_logs], ensure_ascii=False))

        logs_out.append({
            id_col: rid,
            "severity": severity,
            "rubrics_applied": rubrics_applied,
            "orig_summary": summ,
            "corrupted_summary": corr,
            "edits": [asdict(e) for e in edit_logs],
        })

    df_out = df.copy()
    df_out["severity"] = severities_out
    df_out["corrupted_summary"] = corrupted_summaries
    df_out["corruption_json"] = corruption_jsons

    if output_path.endswith(".csv"):
        df_out.to_csv(output_path, index=False)
    elif output_path.endswith(".jsonl"):
        df_out.to_json(output_path, orient="records", lines=True, force_ascii=False)
    else:
        raise ValueError("output_path must end with .csv or .jsonl")

    with open(log_path, "w", encoding="utf-8") as f:
        for rec in logs_out:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Read: {len(df)} rows ({fmt})")
    print(f"Wrote corrupted dataset (ALL strong): {output_path}")
    print(f"Wrote edit logs (JSONL): {log_path}")



# =========================
# Example run
# =========================



import pandas as pd
from datasets import load_dataset
data_name = "cnn_dm_test_first100.csv"

def build_cnn_dm_first100_csv(out_csv: str, split: str = "test") -> str:
    ds = load_dataset("cnn_dailymail", "3.0.0", split=split).select(range(100))
    pd.DataFrame({
        "id": [f"cnn_dm_{split}_{i}" for i in range(100)],
        "document": ds["article"],
        "summary": ds["highlights"],   # gold
    }).to_csv(out_csv, index=False)
    return out_csv

if __name__ == "__main__":
    in_csv = build_cnn_dm_first100_csv(data_name, split="test")

    generate_corrupted_dataset(
        input_path=in_csv,
        output_path="currupt_data.csv",
        log_path="currupt_data_logs.jsonl",
        id_col="id",
        doc_col="document",     # set to None if you want summary-only, no article usage at all
        summary_col="summary",
        seed=7
        
    )
