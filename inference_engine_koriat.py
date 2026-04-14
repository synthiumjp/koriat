"""
inference_engine_koriat.py — Koriat cue-utilisation project (lean v5)

Extends the SDT Calibration 4.1 inference infrastructure
(inference_engine.py, Cacioli 2026, arXiv:2603.25112) with the two additional
model templates required by the Koriat project:

  - Qwen 2.5 Instruct (ChatML format, used for 3B and 7B)
  - DeepSeek R1 Distill Llama 8B (Llama 3 chat template per DeepSeek's
    official model card)

And adds a Koriat-specific prompt formatter that builds the pre-registered
system + user prompt structure for C1 (baseline) and C2 (length-controlled)
conditions.

Pre-reg alignment:
  - System/user structure: pre-reg §"Conditions" (lines 145-152)
  - Exact system prompt wording: pre-reg §"Manipulated variables" (lines 418-425)
  - Cues computed on parsed ANSWER field only: pre-reg §"Indices" (line 488-495)
  - Inference params: n_ctx=4096, temperature=0.0, top_p=0.95, seed=42,
    max_tokens=256 (C1) / 64 (C2), repeat_penalty=1.0 uniform

Gemma 2 template note: Gemma 2's native chat template does not support a
separate system role. The system text is prepended to the user turn.
This is standard Gemma 2 prompting convention and is consistent with how
the M1 SDT project handled Gemma 2. Documented as a Methods footnote,
not as a pre-reg deviation.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Model registry for Koriat project
# ---------------------------------------------------------------------------

KORIAT_MODEL_CONFIGS = {
    "llama31_instruct": {
        "filename": "Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf",
        "stop_tokens": ["<|eot_id|>", "<|end_of_text|>"],
        "model_type": "llama31_instruct",
        "prereg_name": "Llama 3.1 8B Instruct",
        "short_name": "Llama-3.1-8B-Instruct",
    },
    "mistral_instruct": {
        "filename": "Mistral-7B-Instruct-v0.3-Q5_K_M.gguf",
        "stop_tokens": ["</s>"],
        "model_type": "mistral_instruct",
        "prereg_name": "Mistral 7B Instruct v0.3",
        "short_name": "Mistral-7B-Instruct-v0.3",
    },
    "qwen25_3b_instruct": {
        "filename": "Qwen2.5-3B-Instruct-Q5_K_M.gguf",
        "stop_tokens": ["<|im_end|>", "<|endoftext|>"],
        "model_type": "qwen25_instruct",
        "prereg_name": "Qwen 2.5 3B Instruct",
        "short_name": "Qwen2.5-3B-Instruct",
    },
    "qwen25_7b_instruct": {
        "filename": "Qwen2.5-7B-Instruct-Q5_K_M.gguf",
        "stop_tokens": ["<|im_end|>", "<|endoftext|>"],
        "model_type": "qwen25_instruct",
        "prereg_name": "Qwen 2.5 7B Instruct",
        "short_name": "Qwen2.5-7B-Instruct",
    },
    "gemma2_instruct": {
        "filename": "gemma-2-9b-it-Q5_K_M.gguf",
        "stop_tokens": ["<end_of_turn>"],
        "model_type": "gemma2_instruct",
        "prereg_name": "Gemma 2 9B IT",
        "short_name": "gemma-2-9b-it",
    },
    "deepseek_r1_distill_llama": {
        "filename": "DeepSeek-R1-Distill-Llama-8B-Q5_K_M.gguf",
        # DeepSeek R1 Distill Llama uses the Llama 3 chat template per
        # DeepSeek's model card. R1 also uses <think>...</think> tags but
        # those are content, not structural tokens. Stop tokens are Llama 3's.
        "stop_tokens": ["<|eot_id|>", "<|end_of_text|>"],
        "model_type": "deepseek_r1_distill_llama",
        "prereg_name": "DeepSeek R1 Distill Llama 8B",
        "short_name": "DeepSeek-R1-Distill-Llama-8B",
    },
}

# Order of models for the Koriat collection run (per pre-reg protocol).
KORIAT_MODEL_ORDER = [
    "llama31_instruct",
    "mistral_instruct",
    "qwen25_3b_instruct",
    "qwen25_7b_instruct",
    "gemma2_instruct",
    "deepseek_r1_distill_llama",
]


# ---------------------------------------------------------------------------
# Pre-registered prompt strings (LOCKED 14 April 2026)
# ---------------------------------------------------------------------------
# Source: pre-reg §"Manipulated variables" (lines 418-425), §"Conditions"
# (lines 145-152), §"Indices" (lines 488-495).
# Final review and sign-off on these exact strings: JP Cacioli, 14 April 2026.
# Any change to these strings after this point is a pre-reg deviation.

SYSTEM_C1 = (
    "Answer the following question accurately and concisely. "
    "Provide a confidence rating from 0 to 100 for your answer."
)

SYSTEM_C2 = (
    "Answer the following question accurately and concisely. "
    "Answer in EXACTLY one sentence, no more than 25 words. "
    "Provide a confidence rating from 0 to 100 for your answer."
)

USER_TEMPLATE = (
    "{question}\n"
    "\n"
    "Respond in exactly this format:\n"
    "ANSWER: [your answer]\n"
    "CONFIDENCE: [integer 0-100]"
)


def get_system_prompt(condition: int) -> str:
    if condition == 1:
        return SYSTEM_C1
    if condition == 2:
        return SYSTEM_C2
    raise ValueError(f"Condition must be 1 or 2, got {condition}")


def get_user_prompt(question: str) -> str:
    return USER_TEMPLATE.format(question=question)


# ---------------------------------------------------------------------------
# Chat template formatters for all six Koriat models
# ---------------------------------------------------------------------------
def format_prompt_koriat(model_type: str, system_text: str, user_text: str) -> str:
    """
    Build a model-appropriate chat-formatted prompt string for llama-cpp-python.

    Each branch implements the model family's native chat template. Output
    is a raw string ready for llm(prompt, ...) — we do NOT use
    create_chat_completion because llama-cpp-python's built-in template
    handling has inconsistent behavior across model versions and we prefer
    explicit control over the exact token sequence.

    For Gemma 2, which does not support a separate system role, the system
    text is folded into the user turn per standard Gemma 2 convention.
    """
    if model_type == "llama31_instruct" or model_type == "deepseek_r1_distill_llama":
        # Llama 3 / 3.1 chat template (DeepSeek R1 Distill Llama uses the same)
        # llama-cpp-python adds <|begin_of_text|> automatically.
        return (
            "<|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_text}<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_text}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    elif model_type == "mistral_instruct":
        # Mistral v0.3 instruct template.
        # Mistral v0.3 supports system prompts via folding into [INST] per
        # standard practice (the template does not have a dedicated system role).
        return f"[INST] {system_text}\n\n{user_text} [/INST]"
    elif model_type == "qwen25_instruct":
        # Qwen 2.5 uses ChatML format.
        return (
            "<|im_start|>system\n"
            f"{system_text}<|im_end|>\n"
            "<|im_start|>user\n"
            f"{user_text}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    elif model_type == "gemma2_instruct":
        # Gemma 2 does not support a separate system role; fold system text
        # into the user turn. Blank line between system and user text for
        # visual/semantic separation within the single user turn.
        return (
            "<start_of_turn>user\n"
            f"{system_text}\n\n"
            f"{user_text}<end_of_turn>\n"
            "<start_of_turn>model\n"
        )
    else:
        raise ValueError(f"Unknown Koriat model type: {model_type}")


# ---------------------------------------------------------------------------
# Koriat inference engine
# ---------------------------------------------------------------------------

# Pre-reg locked parameters
KORIAT_N_CTX = 4096
KORIAT_TEMPERATURE = 0.0
KORIAT_TOP_P = 0.95
KORIAT_MAX_TOKENS_C1 = 256
KORIAT_MAX_TOKENS_C2 = 64
KORIAT_SEED = 42
KORIAT_REPEAT_PENALTY = 1.0  # uniform across models per pre-reg
KORIAT_LOGPROBS_TOP_K = 5


class KoriatInferenceEngine:
    """
    Koriat inference engine. One instance per model. Pre-reg-locked parameters.

    Usage:
        engine = KoriatInferenceEngine(
            "gemma2_instruct",
            models_dir=r"C:\\sdt_calibration\\models"
        )
        result = engine.generate(question="What is the capital of France?",
                                  condition=1)
        # result dict has: raw_output, first_token_top1_prob,
        #                  answer_mean_log_prob, generation_time_s
        engine.unload()
    """

    def __init__(self, model_key: str, models_dir: str):
        from llama_cpp import Llama

        if model_key not in KORIAT_MODEL_CONFIGS:
            raise ValueError(
                f"Unknown model key: {model_key}. "
                f"Valid keys: {list(KORIAT_MODEL_CONFIGS.keys())}"
            )

        config = KORIAT_MODEL_CONFIGS[model_key]
        model_path = str(Path(models_dir) / config["filename"])

        self.model_key = model_key
        self.model_type = config["model_type"]
        self.stop_tokens = config["stop_tokens"]
        self.prereg_name = config["prereg_name"]
        self.short_name = config["short_name"]

        print(f"  Loading {config['short_name']}...")
        t0 = time.perf_counter()
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=-1,
            n_ctx=KORIAT_N_CTX,
            logits_all=True,   # required for logprobs= in llm() calls
            verbose=False,
            seed=KORIAT_SEED,
        )
        print(f"  Loaded in {time.perf_counter() - t0:.1f}s")

    def generate(self, question: str, condition: int) -> dict:
        """
        Generate one response for one (question, condition) pair.
        Returns dict with all fields needed by the collection pipeline.
        """
        if condition not in (1, 2):
            raise ValueError(f"Condition must be 1 or 2, got {condition}")

        system_text = get_system_prompt(condition)
        user_text = get_user_prompt(question)
        prompt = format_prompt_koriat(self.model_type, system_text, user_text)

        max_tokens = (
            KORIAT_MAX_TOKENS_C1 if condition == 1 else KORIAT_MAX_TOKENS_C2
        )

        t0 = time.perf_counter()
        response = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=KORIAT_TEMPERATURE,
            top_p=KORIAT_TOP_P,
            repeat_penalty=KORIAT_REPEAT_PENALTY,
            stop=self.stop_tokens,
            seed=KORIAT_SEED,
            logprobs=KORIAT_LOGPROBS_TOP_K,
            echo=False,
        )
        gen_time = time.perf_counter() - t0

        raw_output = response["choices"][0]["text"]
        first_prob = _get_first_token_top1_prob(response)
        mean_lp = _compute_answer_mean_log_prob(response)

        return {
            "prompt": prompt,
            "system_text": system_text,
            "user_text": user_text,
            "raw_output": raw_output,
            "first_token_top1_prob": first_prob,
            "answer_mean_log_prob": mean_lp,
            "generation_time_s": gen_time,
        }

    def unload(self):
        if hasattr(self, "llm") and self.llm is not None:
            del self.llm
            self.llm = None
            import gc
            gc.collect()


# ---------------------------------------------------------------------------
# Logprob extraction helpers (ported from collect_data.py, unchanged logic)
# ---------------------------------------------------------------------------
def _compute_answer_mean_log_prob(response) -> Optional[float]:
    logprobs_data = response["choices"][0].get("logprobs")
    if not logprobs_data:
        return None
    token_logprobs = logprobs_data.get("token_logprobs", [])
    if not token_logprobs:
        return None
    valid = [lp for lp in token_logprobs if lp is not None]
    if not valid:
        return None
    return round(sum(valid) / len(valid), 6)


def _get_first_token_top1_prob(response) -> Optional[float]:
    logprobs_data = response["choices"][0].get("logprobs")
    if not logprobs_data:
        return None
    top_logprobs = logprobs_data.get("top_logprobs", [])
    if not top_logprobs or not top_logprobs[0]:
        return None
    max_logprob = max(top_logprobs[0].values())
    return round(math.exp(max_logprob), 6)
