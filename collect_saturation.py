"""
collect_saturation.py — Data collection for the Saturation Study

Pre-registered design: saturation_prereg_v1_0.md
OSF project: (to be created before main collection)

Usage:
    python collect_saturation.py --test             # Single trial on M3 NUM — RUN FIRST
    python collect_saturation.py --sanity           # 5 items × 8 models × 2 conditions = 80 trials
    python collect_saturation.py                    # 524 items × 8 models × 2 conditions = 8,384 trials
    python collect_saturation.py --model M3 --cond NUM  # Single cell (for testing)

Output:
    test_run.parquet        (if --test)
    sanity_run.parquet      (if --sanity)
    raw_responses.parquet   (main collection)

IMPORTANT: Run --test first to verify infrastructure on a single trial. Do not skip.

CHANGES IN v1.0 (relative to draft v0.4):
- Inference uses raw __call__ with Jinja2-rendered chat templates from gguf metadata
  (not create_chat_completion, which returned logprobs=None on llama-cpp-python 0.3.16).
- M8 max_tokens raised to 1024 based on pre-collection verification trials.
- M1 unchanged on continuation prompt path.
- Parser-input clarification (M1 first-line, M8 post-think) folded into process_trial
  as the locked behavior. Locked regex patterns themselves are unchanged.

Author: JP Cacioli
Date: 15 April 2026
"""

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from llama_cpp import Llama
from jinja2 import Template, Environment, BaseLoader

# ============================================================================
# CONFIGURATION — LOCKED BY PRE-REG v1.0
# ============================================================================

MODELS_DIR = Path(r"C:\sdt_calibration\models")
OUTPUT_DIR = Path(r"C:\sdt_calibration\koriat_project_b")

# Model table — order matters for model_idx (used in order seed formula)
MODELS = [
    {"id": "M1", "file": "Meta-Llama-3-8B.Q5_K_M.gguf",              "family": "llama3_base",  "params_b": 8, "cond_first": "NUM"},
    {"id": "M2", "file": "Meta-Llama-3-8B-Instruct-Q5_K_M.gguf",     "family": "llama3",       "params_b": 8, "cond_first": "CAT"},
    {"id": "M3", "file": "Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf",   "family": "llama3",       "params_b": 8, "cond_first": "NUM"},
    {"id": "M4", "file": "Mistral-7B-Instruct-v0.3-Q5_K_M.gguf",     "family": "mistral_v03",  "params_b": 7, "cond_first": "CAT"},
    {"id": "M5", "file": "Qwen2.5-3B-Instruct-Q5_K_M.gguf",          "family": "qwen_chatml",  "params_b": 3, "cond_first": "NUM"},
    {"id": "M6", "file": "Qwen2.5-7B-Instruct-Q5_K_M.gguf",          "family": "qwen_chatml",  "params_b": 7, "cond_first": "CAT"},
    {"id": "M7", "file": "gemma-2-9b-it-Q5_K_M.gguf",                "family": "gemma2",       "params_b": 9, "cond_first": "NUM"},
    {"id": "M8", "file": "DeepSeek-R1-Distill-Llama-8B-Q5_K_M.gguf", "family": "deepseek_r1",  "params_b": 8, "cond_first": "CAT"},
]

# Inference parameters — locked
INFERENCE_PARAMS = {
    "temperature": 0.0,
    "top_p": 0.95,
    "max_tokens": 256,          # default for M1-M7; M8 uses M8_MAX_TOKENS below
    "repeat_penalty": 1.0,
    "seed": 42,
    "logprobs": 5,              # top-k logprobs from raw __call__ path
}

LLAMA_INIT_PARAMS = {
    "n_ctx": 4096,
    "n_gpu_layers": -1,
    "logits_all": True,
    "verbose": False,
    "seed": 42,
}

# M8 max_tokens — LOCKED at 1024 in v1.0 based on pre-collection verification trials
# (see pre-reg §3.7). DeepSeek-R1-Distill emits reasoning traces of 350-710 tokens
# on representative TriviaQA items at temperature 0; 256 was insufficient (verification
# trial 1 truncated mid-reasoning); 1024 produced finish_reason=stop on all four
# follow-up verification trials.
M8_MAX_TOKENS = 1024

# Sample specification — locked
MAIN_N_ITEMS = 524
SAMPLE_SEED = 42
SANITY_INDICES = [0, 100, 500, 1000, 2000]

CONDITIONS = ["NUM", "CAT"]

# ============================================================================
# PROMPTS — LOCKED BY PRE-REG v0.3 §3.3
# ============================================================================

SYSTEM_NUM = (
    "You are answering trivia questions. After your answer, "
    "state your confidence as a percentage from 0 to 100."
)

SYSTEM_CAT = (
    'Classify your confidence into one of the following classes based on '
    'how likely the answer above is to be correct (NO REASONING OR EXPLANATION):\n'
    '- "No chance" (0.0–0.1)\n'
    '- "Really unlikely" (0.1–0.2)\n'
    '- "Chances are slight" (0.2–0.3)\n'
    '- "Unlikely" (0.3–0.4)\n'
    '- "Less than even" (0.4–0.5)\n'
    '- "Better than even" (0.5–0.6)\n'
    '- "Likely" (0.6–0.7)\n'
    '- "Very good chance" (0.7–0.8)\n'
    '- "Highly likely" (0.8–0.9)\n'
    '- "Almost certain" (0.9–1.0)'
)

# Base model (M1) continuation prompt — no system message, no chat template
BASE_PROMPT_TEMPLATE = "Q: {question}\nA:"

# ============================================================================
# CATEGORICAL CLASS MIDPOINTS — LOCKED
# ============================================================================

CAT_CLASS_MIDPOINTS = {
    "no chance":          0.05,
    "really unlikely":    0.15,
    "chances are slight": 0.25,
    "unlikely":           0.35,
    "less than even":     0.45,
    "better than even":   0.55,
    "likely":             0.65,
    "very good chance":   0.75,
    "highly likely":      0.85,
    "almost certain":     0.95,
}

# ============================================================================
# HEDGE LEXICONS (OPTION A — category-pure) — LOCKED
# ============================================================================

HEDGE_EPISTEMIC = {"might", "may", "possibly", "perhaps", "probably", "likely", "seems", "appears"}
HEDGE_SELF      = {"i think", "i believe", "i'd say", "i'd guess", "in my view"}
HEDGE_UNCERT    = {"not sure", "uncertain", "unsure", "unclear", "hard to say", "i'm not certain"}


# ============================================================================
# PARSE REGEX — LOCKED AT PRE-REG TIME
# ============================================================================
# After this file is finalised, these patterns MUST NOT be tuned on observed data.
# Any inadequacy gets reported as a limitation, not fixed post-hoc.

# NUM: capture any number 0-100 that looks like a confidence expression
# Priority: "Confidence: X", "X% confident", "X%", bare number in trailing position
NUM_PATTERNS = [
    re.compile(r"confidence[:\s]*(?:is\s+|of\s+|level\s*[:\s]*)?(\d{1,3})\s*(?:%|percent|/100|out of 100)?", re.IGNORECASE),
    re.compile(r"(\d{1,3})\s*%\s*(?:confident|confidence|sure|certain)", re.IGNORECASE),
    re.compile(r"(?:i am|i'm)\s+(\d{1,3})\s*%", re.IGNORECASE),
    re.compile(r"(\d{1,3})\s*percent", re.IGNORECASE),
    re.compile(r"(\d{1,3})\s*%"),
    re.compile(r"\b(\d{1,3})\b\s*$"),  # bare trailing number — lowest priority
]

# CAT: case-insensitive match against any of the 10 class strings
CAT_PATTERN = re.compile(
    r'"?(no chance|really unlikely|chances are slight|unlikely|less than even|'
    r'better than even|likely|very good chance|highly likely|almost certain)"?',
    re.IGNORECASE
)

# Answer extraction — liberal, catches "Answer: X", "A: X", or first line
ANSWER_PATTERNS = [
    re.compile(r"(?:answer|a)\s*[:\-]\s*(.+?)(?:\n|$|confidence|conf\.)", re.IGNORECASE | re.DOTALL),
    re.compile(r"^(.+?)(?:\n|$)", re.DOTALL),  # first line fallback
]

# Degenerate loop detection — repeating substring covering >50% of response
def is_degenerate_loop(text: str, min_repeat_len: int = 10) -> bool:
    if len(text) < min_repeat_len * 4:
        return False
    for window in range(min_repeat_len, min(100, len(text) // 3)):
        for start in range(0, len(text) - window * 3):
            chunk = text[start:start + window]
            repeats = text.count(chunk)
            if repeats >= 3 and repeats * window > len(text) * 0.5:
                return True
    return False


# ============================================================================
# TRIAL RECORD SCHEMA
# ============================================================================

@dataclass
class TrialRecord:
    # Trial identification
    model_id: str
    condition: str
    item_index: int
    presentation_order_index: int
    triviaqa_question_id: str
    inference_seed: int
    order_seed: int

    # Item content
    question: str
    gold_answer_value: str
    gold_aliases: list

    # Model output
    raw_response: str
    response_length_chars: int
    response_length_tokens: int
    finish_reason: str  # 'stop', 'length', or other — v1.1 diagnostic field
    inference_time_seconds: float
    parse_status: str   # success / no_answer_field / no_confidence_field / degenerate_loop / inference_error / max_tokens_reached

    # M8-specific (0/False for M1-M7)
    thought_block_token_count: int = 0
    thought_block_present: bool = False

    # Parsed measurements
    parsed_answer: Optional[str] = None
    parsed_confidence: Optional[float] = None
    parsed_confidence_class: Optional[str] = None
    parsed_confidence_raw_string: Optional[str] = None
    confidence_position_relative_to_answer: str = "missing"  # before / after / interleaved / missing
    multiple_numeric_candidates_present: bool = False
    correct: Optional[bool] = None

    # Logprob measurements (excluding <think> tokens for M8)
    mean_logprob: Optional[float] = None
    sum_logprob: Optional[float] = None
    min_logprob: Optional[float] = None
    length_normalised_logprob: Optional[float] = None

    # Hedge marker counts
    hedge_epistemic_count: int = 0
    hedge_self_count: int = 0
    hedge_uncertainty_count: int = 0


# ============================================================================
# TRIVIAQA LOADING AND SAMPLING
# ============================================================================

def load_triviaqa_sample(sanity: bool) -> list:
    """
    Load TriviaQA rc.nocontext validation split and return the item list.

    For sanity runs: fixed indices [0, 100, 500, 1000, 2000].
    For main collection: 524 items sampled with seed=42.
    """
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
    assert len(ds) == 17944, f"Expected 17944 validation items, got {len(ds)}"

    if sanity:
        indices = SANITY_INDICES
    else:
        rng = np.random.default_rng(seed=SAMPLE_SEED)
        indices = rng.choice(17944, size=MAIN_N_ITEMS, replace=False).tolist()
        indices.sort()  # canonical order before per-cell shuffling

    items = []
    for item_idx, ds_idx in enumerate(indices):
        row = ds[int(ds_idx)]
        items.append({
            "item_index": item_idx,             # position in the fixed sample (0-523 or 0-4)
            "ds_index": int(ds_idx),            # position in TriviaQA validation split
            "question_id": row["question_id"],
            "question": row["question"],
            "answer_value": row["answer"]["value"],
            "aliases": list(row["answer"]["aliases"]) + [row["answer"]["value"]],
        })
    return items


def get_item_order_for_cell(model_idx: int, cond_idx: int, n_items: int) -> list:
    """
    Compute per-cell presentation order via the pre-registered seed formula.
    Returns a permutation of item indices [0..n_items-1].
    """
    order_seed = 42 + (model_idx * 100) + cond_idx
    rng = np.random.default_rng(seed=order_seed)
    order = rng.permutation(n_items).tolist()
    return order, order_seed


# ============================================================================
# CHAT TEMPLATE APPLICATION (gguf metadata + Jinja2)
# ============================================================================
#
# llama-cpp-python 0.3.16's create_chat_completion returns logprobs=None on the
# chat-completion path (verified empirically — see pre-reg §3.7). To get logprobs,
# we must use the raw __call__ interface. To still apply model-appropriate chat
# templates, we read the Jinja2 chat_template string from each gguf's metadata
# (tokenizer.chat_template) and render it ourselves before passing the resulting
# string to llm(prompt=...).
#
# This is conceptually identical to what create_chat_completion would do internally,
# except that the raw __call__ path returns populated logprobs while the chat path
# does not. The model receives the same final token sequence either way.

def get_bos_token(llm: Llama) -> str:
    """
    Get the BOS token string from gguf metadata. Some chat templates reference
    `bos_token` in their Jinja2 source (Llama 3, Mistral, Gemma 2 all do).
    """
    try:
        bos_id = int(llm.metadata.get("tokenizer.ggml.bos_token_id", 0))
        return llm.detokenize([bos_id]).decode("utf-8", errors="replace")
    except Exception:
        return ""


def render_chat_template(llm: Llama, model_family: str, condition: str,
                         question: str) -> str:
    """
    Render the Jinja2 chat template from this gguf's metadata into a final prompt
    string. Handles per-family special cases for system-prompt placement.

    NOT used for M1 (llama3_base) — M1 uses build_base_prompt() instead.
    """
    template_str = llm.metadata.get("tokenizer.chat_template")
    if not template_str:
        raise RuntimeError(f"No chat_template in gguf metadata for {model_family}")

    bos_token = get_bos_token(llm)
    system_prompt = SYSTEM_NUM if condition == "NUM" else SYSTEM_CAT

    # Gemma 2 and Mistral v0.3 both refuse a system role in their chat templates:
    # - Gemma 2: raise_exception('System role not supported')
    # - Mistral v0.3: raise_exception('Conversation roles must alternate user/assistant/...')
    # Both are handled by folding the system prompt into the user turn. This is the
    # same principle and produces an equivalent prompt under both templates.
    if model_family in ("gemma2", "mistral_v03"):
        messages = [{"role": "user", "content": f"{system_prompt}\n\n{question}"}]
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

    # Some templates require add_generation_prompt to append the assistant turn marker
    template = Template(template_str)
    rendered = template.render(
        messages=messages,
        bos_token=bos_token,
        add_generation_prompt=True,
    )
    return rendered


def build_base_prompt(question: str) -> str:
    """Continuation-style prompt for M1 base model. No chat template."""
    return BASE_PROMPT_TEMPLATE.format(question=question)


# ============================================================================
# INFERENCE
# ============================================================================

def run_inference(llm: Llama, model_family: str, condition: str, question: str,
                  max_tokens: int) -> dict:
    """
    Run a single inference call. Returns dict with raw_response, tokens, logprobs,
    finish_reason, wall time.

    Uses raw llm(prompt=...) for ALL models (including instruct models), with chat
    templates applied via render_chat_template() for instruct models. This is required
    because llama-cpp-python 0.3.16's create_chat_completion returns logprobs=None.
    """
    start = time.perf_counter()
    try:
        if model_family == "llama3_base":
            # M1: continuation prompt, no chat template
            prompt = build_base_prompt(question)
        else:
            # M2-M8: render gguf chat template via Jinja2
            prompt = render_chat_template(llm, model_family, condition, question)

        result = llm(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=INFERENCE_PARAMS["temperature"],
            top_p=INFERENCE_PARAMS["top_p"],
            repeat_penalty=INFERENCE_PARAMS["repeat_penalty"],
            seed=INFERENCE_PARAMS["seed"],
            logprobs=INFERENCE_PARAMS["logprobs"],   # top-k logprobs
        )

        choice = result["choices"][0]
        raw_response = choice["text"]
        finish_reason = choice.get("finish_reason", "")

        # Logprobs structure for raw __call__ in llama-cpp-python 0.3.16:
        # {'tokens': [str], 'text_offset': [int], 'token_logprobs': [float], 'top_logprobs': [dict]}
        logprobs_obj = choice.get("logprobs") or {}
        tokens = logprobs_obj.get("tokens") or []
        token_logprobs = logprobs_obj.get("token_logprobs") or []

        elapsed = time.perf_counter() - start
        return {
            "raw_response": raw_response,
            "tokens": tokens,
            "token_logprobs": token_logprobs,
            "finish_reason": finish_reason,
            "inference_time_seconds": elapsed,
            "error": None,
        }
    except Exception as e:
        elapsed = time.perf_counter() - start
        return {
            "raw_response": "",
            "tokens": [],
            "token_logprobs": [],
            "finish_reason": "error",
            "inference_time_seconds": elapsed,
            "error": str(e),
        }


# ============================================================================
# M8 THINK-BLOCK HANDLING
# ============================================================================

def extract_think_block(raw_response: str) -> tuple:
    """
    For M8 only. Extract <think>...</think> block.
    Returns (think_text, post_think_text, present).
    If no closing tag: the whole response is treated as still-thinking, post_think_text = "".
    If no opening tag: no think block, post_think_text = raw_response.
    """
    open_tag = raw_response.find("<think>")
    close_tag = raw_response.find("</think>")

    if open_tag == -1:
        return "", raw_response, False
    if close_tag == -1:
        # opened but not closed — reasoning ran off the end
        return raw_response[open_tag + len("<think>"):], "", False
    think_text = raw_response[open_tag + len("<think>"):close_tag]
    post_think = raw_response[close_tag + len("</think>"):]
    return think_text, post_think, True


def count_tokens_in_think_block(tokens: list, raw_response: str) -> int:
    """
    Rough count of how many response tokens fall inside <think>...</think>.
    Simple approach: count tokens up to and including </think> marker.
    For a more precise count, reconcile token offsets with character positions.
    """
    if "<think>" not in raw_response or "</think>" not in raw_response:
        return 0
    # Reconstruct character offsets for each token
    cumulative = ""
    start_idx = None
    end_idx = None
    for i, tok in enumerate(tokens):
        cumulative += tok
        if start_idx is None and "<think>" in cumulative:
            start_idx = i
        if end_idx is None and "</think>" in cumulative:
            end_idx = i
            break
    if start_idx is None or end_idx is None:
        return 0
    return max(0, end_idx - start_idx)


# ============================================================================
# PARSING
# ============================================================================

def normalise(text: str) -> str:
    """Normalisation for accuracy matching — per pre-reg §3.2."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"^(the|a|an)\s+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_answer(text: str) -> Optional[str]:
    """Parse the answer field from `text`. Caller chooses what `text` is."""
    for pat in ANSWER_PATTERNS:
        m = pat.search(text)
        if m:
            candidate = m.group(1).strip()
            if candidate and len(candidate) < 500:  # sanity bound
                return candidate
    return None


def parse_numeric_confidence(text: str) -> tuple:
    """
    Returns (value_in_unit_interval, raw_string, multiple_candidates).
    value in [0, 1] or None; raw_string is the substring captured; multiple_candidates
    is True if 2+ distinct numeric candidates were found anywhere in `text`.

    Two-pass design:
      Pass 1 — exhaustively collect all valid numeric candidates from all patterns
               into a set, for the multiple_candidates flag.
      Pass 2 — pick the canonical value from the first pattern that yields a match
               (priority order: explicit confidence markers first, bare trailing
               number last). raw_string corresponds to the chosen value, not to
               the first scanned hit.
    """
    # Pass 1: exhaustive accumulation for the multiple_candidates flag
    all_numbers = set()
    for pat in NUM_PATTERNS:
        for m in pat.finditer(text):
            try:
                val = int(m.group(1))
            except (ValueError, IndexError):
                continue
            if 0 <= val <= 100:
                all_numbers.add(val)

    if not all_numbers:
        return None, None, False

    # Pass 2: priority-ordered selection. raw_string corresponds to the chosen value.
    chosen_value = None
    chosen_raw = None
    for pat in NUM_PATTERNS:
        m = pat.search(text)
        if m:
            try:
                val = int(m.group(1))
            except (ValueError, IndexError):
                continue
            if 0 <= val <= 100:
                chosen_value = val
                chosen_raw = m.group(0)
                break

    if chosen_value is None:
        return None, None, len(all_numbers) > 1
    return chosen_value / 100.0, chosen_raw, len(all_numbers) > 1


def parse_categorical_confidence(text: str) -> tuple:
    """
    Returns (value_in_unit_interval, class_string, raw_string).
    Caller chooses what `text` is.
    """
    m = CAT_PATTERN.search(text)
    if not m:
        return None, None, None
    class_str = m.group(1).lower().strip()
    midpoint = CAT_CLASS_MIDPOINTS.get(class_str)
    return midpoint, class_str, m.group(0)


def compute_confidence_position(text: str, answer_str: Optional[str],
                                 confidence_raw: Optional[str]) -> str:
    """
    Return one of: before / after / interleaved / missing.
    Caller chooses what `text` is — must be the same string that was passed to
    parse_answer and the relevant confidence parser, so positions are consistent.
    """
    if confidence_raw is None or answer_str is None:
        return "missing"
    ans_pos = text.find(answer_str)
    conf_pos = text.find(confidence_raw)
    if ans_pos == -1 or conf_pos == -1:
        return "missing"
    if conf_pos < ans_pos:
        return "before"
    if conf_pos > ans_pos + len(answer_str):
        return "after"
    return "interleaved"


def score_correct(parsed_answer: Optional[str], aliases: list) -> Optional[bool]:
    if parsed_answer is None:
        return None
    norm_answer = normalise(parsed_answer)
    for alias in aliases:
        norm_alias = normalise(alias)
        if norm_alias and norm_alias in norm_answer:
            return True
    return False


def count_hedges(text: str) -> tuple:
    """Case-insensitive whole-word / whole-phrase counts."""
    text_lower = text.lower()
    # Word-boundary for single-word lexicons
    epist = sum(len(re.findall(rf"\b{re.escape(w)}\b", text_lower)) for w in HEDGE_EPISTEMIC)
    # Phrase matching for multi-word lexicons
    self_ct = sum(text_lower.count(phrase) for phrase in HEDGE_SELF)
    uncert = sum(text_lower.count(phrase) for phrase in HEDGE_UNCERT)
    return epist, self_ct, uncert


def compute_logprob_stats(token_logprobs: list, exclude_count: int = 0) -> dict:
    """
    Compute summary statistics over token logprobs.
    For M8: exclude_count = number of tokens inside <think> block.
    """
    usable = [lp for lp in token_logprobs[exclude_count:] if lp is not None]
    if not usable:
        return {"mean": None, "sum": None, "min": None, "length_normalised": None}
    total = float(sum(usable))
    return {
        "mean": total / len(usable),
        "sum": total,
        "min": float(min(usable)),
        "length_normalised": total / len(usable),
    }


# ============================================================================
# TRIAL ASSEMBLY
# ============================================================================

def process_trial(model_spec: dict, condition: str, item: dict,
                  item_idx_in_sample: int, presentation_order_index: int,
                  order_seed: int, inference_result: dict) -> TrialRecord:
    """
    Assemble a TrialRecord from an inference result. Does all parsing, scoring,
    logprob computation, hedge counting, and M8 think-block extraction.
    """
    raw_response = inference_result["raw_response"]
    tokens = inference_result["tokens"]
    token_logprobs = inference_result["token_logprobs"]
    error = inference_result["error"]
    finish_reason = inference_result["finish_reason"]

    is_m1 = model_spec["id"] == "M1"
    is_m8 = model_spec["id"] == "M8"

    # M8 think-block handling
    think_token_count = 0
    think_present = False
    post_think = ""
    if is_m8:
        _, post_think, think_present = extract_think_block(raw_response)
        think_token_count = count_tokens_in_think_block(tokens, raw_response)

    # ------------------------------------------------------------------
    # Determine PARSE INPUT — which substring the locked patterns operate on.
    # The locked regex patterns are NOT modified. Only the input string changes,
    # for two structurally distinct cases documented in pre-reg v0.4 §3.5 and §10.
    #
    # Case M1 (continuation prompt): the response starts AT the answer because
    # the prompt ends with "A:" and the model continues from there. The locked
    # ANSWER_PATTERNS are designed for instruct-model chat replies and would
    # otherwise match later hallucinated "Q: ... A: ..." continuation pairs.
    # Fix: truncate raw_response at the first "\nQ:" marker (catching runaway
    # continuation), then use the remaining substring for both answer and
    # confidence parsing. Answer is taken from the first line of that substring.
    #
    # Case M8 (reasoning model with </think> closure): when </think> is emitted,
    # the answer and confidence live in the post-</think> text. Parsing the full
    # raw_response would let ANSWER_PATTERNS' first-line fallback match the
    # literal string "<think>". Fix: when think_present is True, parse from
    # post_think. When </think> is absent (runaway reasoning hit max_tokens),
    # fall back to raw_response, which is the case already contemplated in §3.3.
    #
    # All other models: parse_input is raw_response. No change from v0.3.
    # ------------------------------------------------------------------
    if is_m1:
        # Truncate at the first hallucinated follow-up Q marker
        m1_truncated = raw_response.split("\nQ:", 1)[0]
        parse_input = m1_truncated
        # For M1 specifically, the answer is the first line of the truncated block
        m1_first_line = parse_input.split("\n", 1)[0].strip()
    elif is_m8 and think_present:
        # Strip leading whitespace from post_think so the first-line fallback
        # in ANSWER_PATTERNS doesn't trip on a leading "\n\n".
        parse_input = post_think.lstrip()
    elif is_m8 and not think_present and "<think>" in raw_response:
        # Runaway reasoning case: the model opened <think> but never closed it
        # before max_tokens. Strip the literal opening tag so parsed_answer
        # doesn't contain "<think>" as its first characters. The trial will
        # still be flagged as max_tokens_reached downstream.
        parse_input = raw_response.replace("<think>", "", 1).lstrip()
    else:
        parse_input = raw_response

    # ------------------------------------------------------------------
    # Determine parse status
    #
    # NOTE: degenerate_loop and max_tokens_reached checks run on PARSE_INPUT,
    # not raw_response. For M1, parse_input has already been truncated at the
    # first "\nQ:" hallucinated continuation marker, so a clean first-line
    # answer is not wrongly flagged as degenerate because of the continuation
    # tail. For M8 with closed think block, parse_input is the post-</think>
    # substring, so degenerate reasoning inside the think block does not
    # contaminate the status of the post-think answer. This ordering follows
    # pre-reg §3.5's intent that model-specific parse-input logic precedes
    # general parse-status detection.
    # ------------------------------------------------------------------
    if error is not None:
        parse_status = "inference_error"
    elif not raw_response.strip():
        parse_status = "inference_error"
    elif is_degenerate_loop(parse_input):
        parse_status = "degenerate_loop"
    elif finish_reason == "length":
        parse_status = "max_tokens_reached"
    else:
        parse_status = "pending"  # fill in after parsing

    # ------------------------------------------------------------------
    # Parse answer (uses parse_input, with M1 first-line override)
    # ------------------------------------------------------------------
    parsed_answer = None
    if parse_status in ("pending", "max_tokens_reached"):
        if is_m1:
            # M1 answer is the first line of the truncated continuation,
            # if non-empty. The ANSWER_PATTERNS still serve as a fallback
            # in case the first line itself contains an explicit "A:" marker
            # that the patterns can clean up.
            if m1_first_line:
                parsed_answer = m1_first_line
            else:
                parsed_answer = parse_answer(parse_input)
        else:
            parsed_answer = parse_answer(parse_input)

    # ------------------------------------------------------------------
    # Parse confidence (condition-specific, runs on parse_input)
    # ------------------------------------------------------------------
    parsed_confidence = None
    parsed_confidence_class = None
    parsed_confidence_raw = None
    multiple_candidates = False
    if parse_status in ("pending", "max_tokens_reached"):
        if condition == "NUM":
            parsed_confidence, parsed_confidence_raw, multiple_candidates = parse_numeric_confidence(parse_input)
        else:  # CAT
            parsed_confidence, parsed_confidence_class, parsed_confidence_raw = parse_categorical_confidence(parse_input)

    # ------------------------------------------------------------------
    # Resolve final parse_status
    # ------------------------------------------------------------------
    if parse_status == "pending":
        if parsed_answer is None and parsed_confidence is None:
            parse_status = "no_answer_field"
        elif parsed_answer is None:
            parse_status = "no_answer_field"
        elif parsed_confidence is None:
            parse_status = "no_confidence_field"
        else:
            parse_status = "success"
    # If max_tokens_reached: keep that status regardless of parse outcome, but still try to extract

    # ------------------------------------------------------------------
    # Confidence position (computed on the same parse_input as the parsers,
    # so positions are internally consistent)
    # ------------------------------------------------------------------
    conf_position = compute_confidence_position(parse_input, parsed_answer, parsed_confidence_raw)

    # Accuracy
    correct = score_correct(parsed_answer, item["aliases"]) if parsed_answer else None

    # Logprobs (exclude think-block tokens for M8)
    lp_stats = compute_logprob_stats(token_logprobs, exclude_count=think_token_count if is_m8 else 0)

    # Hedge counts (on full raw_response, including think block — this is a design choice,
    # documented here: we count hedges anywhere in the response)
    epist, self_ct, uncert = count_hedges(raw_response)

    return TrialRecord(
        model_id=model_spec["id"],
        condition=condition,
        item_index=item_idx_in_sample,
        presentation_order_index=presentation_order_index,
        triviaqa_question_id=item["question_id"],
        inference_seed=42,
        order_seed=order_seed,
        question=item["question"],
        gold_answer_value=item["answer_value"],
        gold_aliases=item["aliases"],
        raw_response=raw_response,
        response_length_chars=len(raw_response),
        response_length_tokens=len(tokens),
        finish_reason=finish_reason or "",
        inference_time_seconds=inference_result["inference_time_seconds"],
        parse_status=parse_status,
        thought_block_token_count=think_token_count,
        thought_block_present=think_present,
        parsed_answer=parsed_answer,
        parsed_confidence=parsed_confidence,
        parsed_confidence_class=parsed_confidence_class,
        parsed_confidence_raw_string=parsed_confidence_raw,
        confidence_position_relative_to_answer=conf_position,
        multiple_numeric_candidates_present=multiple_candidates,
        correct=correct,
        mean_logprob=lp_stats["mean"],
        sum_logprob=lp_stats["sum"],
        min_logprob=lp_stats["min"],
        length_normalised_logprob=lp_stats["length_normalised"],
        hedge_epistemic_count=epist,
        hedge_self_count=self_ct,
        hedge_uncertainty_count=uncert,
    )


# ============================================================================
# INCREMENTAL PARQUET WRITER (patched 15 April 2026 — explicit schema)
# ============================================================================
# Bugfix: the original writer inferred its schema from the first flushed batch
# via pa.Table.from_pandas(). When a batch contained columns that were entirely
# None (e.g. M1 NUM where parsed_confidence is always None because M1 is the
# base model), pyarrow inferred those columns as `null` type, and subsequent
# batches with actual double/string values mismatched the locked schema and
# raised ValueError at write_table().
#
# Fix: declare an explicit pyarrow schema matching the TrialRecord dataclass
# field-for-field and pass it to both pa.Table.from_pandas() and the
# ParquetWriter constructor. Schema no longer depends on batch contents.
#
# Pre-registration deviation disclosure: this patch is a bugfix to the writer
# only. No collection logic, seed, prompt, parser, record, hypothesis,
# threshold, decision rule, or analysis procedure is affected. To be disclosed
# in Methods as a post-registration bugfix to the ParquetWriter class.
# ============================================================================

TRIAL_RECORD_SCHEMA = pa.schema([
    # Trial identification
    ("model_id", pa.large_string()),
    ("condition", pa.large_string()),
    ("item_index", pa.int64()),
    ("presentation_order_index", pa.int64()),
    ("triviaqa_question_id", pa.large_string()),
    ("inference_seed", pa.int64()),
    ("order_seed", pa.int64()),
    # Item content
    ("question", pa.large_string()),
    ("gold_answer_value", pa.large_string()),
    ("gold_aliases", pa.list_(pa.string())),
    # Model output
    ("raw_response", pa.large_string()),
    ("response_length_chars", pa.int64()),
    ("response_length_tokens", pa.int64()),
    ("finish_reason", pa.large_string()),
    ("inference_time_seconds", pa.float64()),
    ("parse_status", pa.large_string()),
    # M8-specific
    ("thought_block_token_count", pa.int64()),
    ("thought_block_present", pa.bool_()),
    # Parsed measurements (all nullable)
    ("parsed_answer", pa.large_string()),
    ("parsed_confidence", pa.float64()),
    ("parsed_confidence_class", pa.large_string()),
    ("parsed_confidence_raw_string", pa.large_string()),
    ("confidence_position_relative_to_answer", pa.large_string()),
    ("multiple_numeric_candidates_present", pa.bool_()),
    ("correct", pa.bool_()),
    # Logprob measurements (all nullable)
    ("mean_logprob", pa.float64()),
    ("sum_logprob", pa.float64()),
    ("min_logprob", pa.float64()),
    ("length_normalised_logprob", pa.float64()),
    # Hedge counts
    ("hedge_epistemic_count", pa.int64()),
    ("hedge_self_count", pa.int64()),
    ("hedge_uncertainty_count", pa.int64()),
])


class ParquetWriter:
    """Append-only parquet writer that flushes every flush_every trials.

    Uses an explicit schema (TRIAL_RECORD_SCHEMA) so batches with all-None
    nullable columns do not corrupt schema inference.
    """

    def __init__(self, path: Path, flush_every: int = 50):
        self.path = path
        self.flush_every = flush_every
        self.buffer: list = []
        self.rows_written = 0
        self.writer = None

    def add(self, record: TrialRecord):
        self.buffer.append(asdict(record))
        if len(self.buffer) >= self.flush_every:
            self.flush()

    def flush(self):
        if not self.buffer:
            return
        df = pd.DataFrame(self.buffer)
        # Cast to explicit schema; raises loudly if a field type drifts.
        table = pa.Table.from_pandas(
            df,
            schema=TRIAL_RECORD_SCHEMA,
            preserve_index=False,
        )
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.path, TRIAL_RECORD_SCHEMA)
        self.writer.write_table(table)
        self.rows_written += len(self.buffer)
        self.buffer = []

    def close(self):
        self.flush()
        if self.writer is not None:
            self.writer.close()


# ============================================================================
# MAIN COLLECTION LOOP
# ============================================================================

def collect(sanity: bool, test_only: bool = False,
            only_model: Optional[str] = None, only_cond: Optional[str] = None):
    """
    Run the collection. sanity=True uses 5 fixed items; test_only=True uses 1 item
    on M3 NUM for infrastructure verification.
    """
    # Determine output file
    if test_only:
        output_path = OUTPUT_DIR / "test_run.parquet"
        items = [ {"item_index": 0, "ds_index": 0, **_loader_single(0)} ]
    elif sanity:
        output_path = OUTPUT_DIR / "sanity_run.parquet"
        items = load_triviaqa_sample(sanity=True)
    else:
        output_path = OUTPUT_DIR / "raw_responses.parquet"
        items = load_triviaqa_sample(sanity=False)

    n_items = len(items)
    print(f"[init] Output: {output_path}")
    print(f"[init] Items: {n_items}")
    print(f"[init] Models: {len(MODELS)}")
    print(f"[init] Conditions: {CONDITIONS}")

    writer = ParquetWriter(output_path, flush_every=50)

    try:
        for model_idx, model_spec in enumerate(MODELS):
            if only_model and model_spec["id"] != only_model:
                continue

            # Test mode: only M3
            if test_only and model_spec["id"] != "M3":
                continue

            model_path = MODELS_DIR / model_spec["file"]
            if not model_path.exists():
                print(f"[error] Model not found: {model_path}")
                sys.exit(1)

            # Determine max_tokens for this model — M8 is locked at 1024 per pre-reg v1.0 §3.4
            if model_spec["id"] == "M8":
                max_tokens = M8_MAX_TOKENS
            else:
                max_tokens = INFERENCE_PARAMS["max_tokens"]

            print(f"[model] Loading {model_spec['id']} ({model_spec['file']})...")
            llm = Llama(model_path=str(model_path), **LLAMA_INIT_PARAMS)
            print(f"[model] Loaded {model_spec['id']}. max_tokens={max_tokens}")

            # Condition ordering — counterbalanced per pre-reg §3.3
            if model_spec["cond_first"] == "NUM":
                cond_sequence = ["NUM", "CAT"]
            else:
                cond_sequence = ["CAT", "NUM"]

            for cond_idx_in_sequence, condition in enumerate(cond_sequence):
                if only_cond and condition != only_cond:
                    continue
                if test_only and condition != "NUM":
                    continue

                # cond_idx for seed formula: NUM=0, CAT=1 (fixed, independent of sequence order)
                cond_idx_for_seed = 0 if condition == "NUM" else 1
                order, order_seed = get_item_order_for_cell(model_idx, cond_idx_for_seed, n_items)

                print(f"[cell] {model_spec['id']} × {condition}: order_seed={order_seed}")

                for pres_idx, item_pos in enumerate(order):
                    if test_only and pres_idx > 0:
                        break
                    item = items[item_pos]

                    inference_result = run_inference(
                        llm=llm,
                        model_family=model_spec["family"],
                        condition=condition,
                        question=item["question"],
                        max_tokens=max_tokens,
                    )

                    record = process_trial(
                        model_spec=model_spec,
                        condition=condition,
                        item=item,
                        item_idx_in_sample=item["item_index"],
                        presentation_order_index=pres_idx,
                        order_seed=order_seed,
                        inference_result=inference_result,
                    )
                    writer.add(record)

                    if pres_idx % 10 == 0:
                        print(f"  [{model_spec['id']}/{condition}] {pres_idx+1}/{n_items} "
                              f"status={record.parse_status} conf={record.parsed_confidence} "
                              f"correct={record.correct}")

            # Unload model before loading next one
            del llm
            print(f"[model] Unloaded {model_spec['id']}")

    finally:
        writer.close()
        print(f"[done] Wrote {writer.rows_written} rows to {output_path}")


def _loader_single(ds_index: int) -> dict:
    """Helper for --test mode: load a single TriviaQA item."""
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
    row = ds[ds_index]
    return {
        "question_id": row["question_id"],
        "question": row["question"],
        "answer_value": row["answer"]["value"],
        "aliases": list(row["answer"]["aliases"]) + [row["answer"]["value"]],
    }


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Saturation study data collection")
    parser.add_argument("--sanity", action="store_true",
                        help="Run sanity collection (5 items, fixed indices)")
    parser.add_argument("--test", action="store_true",
                        help="Single-trial infrastructure test (M3 NUM, 1 item). "
                             "Run this FIRST before --sanity.")
    parser.add_argument("--model", type=str, default=None,
                        help="Restrict to one model ID (e.g., M3)")
    parser.add_argument("--cond", type=str, default=None, choices=["NUM", "CAT"],
                        help="Restrict to one condition")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.test:
        print("[main] Infrastructure test mode — 1 trial on M3 NUM")
        collect(sanity=False, test_only=True)
    else:
        collect(sanity=args.sanity, test_only=False,
                only_model=args.model, only_cond=args.cond)


if __name__ == "__main__":
    main()


# ============================================================================
# KNOWN GAPS FOR TOMORROW'S CHAT TO RESOLVE
# ============================================================================
#
# This is a STRUCTURAL SKELETON. Tomorrow's session needs to verify and fix:
#
# 1. LOGPROB EXTRACTION. The llama-cpp-python API for logprobs differs between
#    completion (__call__) and create_chat_completion. The code paths here are
#    placeholders — verify the exact return structure on your installed version
#    and adjust the token/logprob extraction accordingly. Test this on the
#    single --test trial before running sanity.
#
# 2. GEMMA 2 SYSTEM HANDLING. I've used "fold system into user turn" but
#    double-check whether llama-cpp-python's Gemma 2 chat template handler
#    does this automatically. If it does, remove the special case in build_messages.
#
# 3. DEEPSEEK R1 DISTILL CHAT TEMPLATE. Verify that llama-cpp-python applies
#    the correct DeepSeek R1 chat template. The <think> block should be emitted
#    by the model automatically in its response; if it isn't, inspect the
#    chat template the gguf file ships with.
#
# 4. M1 BASE LOGPROBS. The base model uses llm(prompt=...) not chat completion.
#    The logprobs structure is the openai-style {tokens, token_logprobs}.
#    Verify this works on M1 specifically — base models sometimes have quirky
#    tokenizer behaviour.
#
# 5. DEGENERATE LOOP DETECTOR. The current implementation is O(n²) in response
#    length. For 991-char responses it's fine; if you see collection slowing
#    dramatically, optimise or raise min_repeat_len.
#
# 6. THINK-BLOCK TOKEN COUNT. count_tokens_in_think_block uses a cumulative
#    character-based reconstruction. This will miscount if tokens span the
#    <think> or </think> markers. Verify on a real M8 trial and adjust.
#
# 7. PARSING REGEX ROBUSTNESS. The NUM patterns are ordered by specificity.
#    On the --test trial, print the raw_response and parsed_confidence side
#    by side to verify the parser picks up the model's actual confidence
#    expression. If it fails, the regex is LOCKED — report as limitation,
#    do not tune.
#
# 8. CRASH RECOVERY. The incremental parquet writer flushes every 50 trials,
#    but there's no resume-from-checkpoint logic. If collection crashes mid-run,
#    the partial parquet is readable but you'll need to figure out which cells
#    were completed and restart from the next cell. Add a manifest file that
#    tracks completed (model, condition) pairs if you want true resume.
#
# 9. SEED BEHAVIOUR. llama-cpp-python's seed parameter interacts with
#    temperature=0 — at temp=0 the output should be deterministic regardless
#    of seed, but verify on M5 (smallest model, fastest to test) before trusting.
