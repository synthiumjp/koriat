# Verbal Confidence Saturation in 3–9B Open-Weight Instruction-Tuned LLMs

**A Pre-Registered Psychometric Validity Screen**

**Paper:** [arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX) (update when posted)

**Author:** JP Cacioli
**ORCID:** [0009-0000-7054-2014](https://orcid.org/0009-0000-7054-2014)
**Programme:** Classical Minds, Modern Machines

## What this study found

Seven instruction-tuned open-weight models (3–9B parameters, four families) were tested on 524 TriviaQA items under numeric and categorical confidence elicitation. A psychometric validity screen was applied to each model–format cell. Every instruct model was classified Invalid on numeric confidence. The mean ceiling rate was 91.7%. Categorical elicitation did not rescue validity. It disrupted task performance, dropping accuracy below 5% in six of seven models. Token-level logprobability did not usefully predict verbalised confidence.

H1 (saturation prevalence): confirmed. H2 (validity screening): confirmed, 7/7 vs predicted ≥4. H4 (format rescue): not confirmed. H5 (logprob independence): confirmed.

## Pre-registration

Registered on OSF. Immutable as of 15 April 2026.

- **OSF project:** https://osf.io/xgt73
- **OSF registration:** https://osf.io/azbvx/overview
- **Pre-registration document:** [`saturation_prereg_v1_2.md`](saturation_prereg_v1_2.md)

One deviation is disclosed: a bugfix to the ParquetWriter schema inference. No collection logic, seeds, prompts, or records were affected.

## Design

- **Models (8):** Meta-Llama-3-8B (base, exploratory), Meta-Llama-3-8B-Instruct, Meta-Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3, Qwen2.5-3B-Instruct, Qwen2.5-7B-Instruct, Gemma-2-9b-it, DeepSeek-R1-Distill-Llama-8B
- **Conditions (2):** numeric (NUM, 0–100) and categorical (CAT, 10-class)
- **Items:** 524 TriviaQA rc.nocontext validation items, `seed=42`
- **Total trials:** 8,384
- **Inference:** llama-cpp-python 0.3.16, Vulkan backend, greedy decoding, `logprobs=5`
- **Hardware:** AMD RX 7900 GRE 16 GB, Q5_K_M quantisation

## Repository layout

```
saturation_prereg_v1_2.md       Locked pre-registration document (v1.2)
collect_saturation.py           Collection script (patched post-registration, deviation disclosed)
analyze_saturation.py           Analysis pipeline (H1, H2, H4, H5, A1–A8, E1–E10)
generate_figures.py             Figure generation (3 publication figures)
inspect_sanity.py               Sanity inspection utility
m8_pilot_verification.json      M8 pilot parse-rate verification
requirements.txt                Frozen Python environment
figures/                        Publication figures (PDF + PNG)
analysis_outputs/               CSV outputs from the analysis pipeline
LICENSE                         MIT
CITATION.cff                    Citation metadata
```

Data artifacts (`raw_responses.parquet`, `sanity_run.parquet`) are on the OSF project page.

## Prompts

**NUM system prompt:**
```
You are answering trivia questions. After your answer, state your confidence as a percentage from 0 to 100.
```

**CAT system prompt:**
```
Classify your confidence into one of the following classes based on how likely
the answer above is to be correct (NO REASONING OR EXPLANATION):
- "No chance" (0.0–0.1)
- "Really unlikely" (0.1–0.2)
- "Chances are slight" (0.2–0.3)
- "Unlikely" (0.3–0.4)
- "Less than even" (0.4–0.5)
- "Better than even" (0.5–0.6)
- "Likely" (0.6–0.7)
- "Very good chance" (0.7–0.8)
- "Highly likely" (0.8–0.9)
- "Almost certain" (0.9–1.0)
```

## Reproducibility

All randomness is seeded. Item draw: `seed=42`. Per-cell order: `42 + (model_idx * 100) + cond_idx`. Ridge CV and bootstrap: `seed=42`. Inference is greedy (`temperature=0.0`). Outputs are bit-identical on the same hardware and library versions.

To reproduce:

```bash
pip install -r requirements.txt
python collect_saturation.py        # ~8 hours on RX 7900 GRE
python analyze_saturation.py        # full pipeline
python generate_figures.py          # 3 figures
```

## Related papers

This study is part of the Classical Minds, Modern Machines research programme.

| Paper | arXiv |
|---|---|
| Type-2 signal detection for LLM metacognition | [2603.14893](https://arxiv.org/abs/2603.14893) |
| Meta-d' and M-ratio for LLM metacognition | [2603.25112](https://arxiv.org/abs/2603.25112) |
| The metacognitive monitoring battery | [2604.15702](https://arxiv.org/abs/2604.15702) |
| Validity scaling for LLM metacognitive self-report | [2604.17707](https://arxiv.org/abs/2604.17707) |
| Screen before you interpret (validity protocol) | [2604.17714](https://arxiv.org/abs/2604.17714) |
| Criterion validation via selective prediction | [2604.17716](https://arxiv.org/abs/2604.17716) |
| AUROC2 is format-stable; M-ratio is not | [2604.08976](https://arxiv.org/abs/2604.08976) |

## Citation

```bibtex
@article{cacioli2026saturation,
  title={Verbal Confidence Saturation in 3--9B Open-Weight Instruction-Tuned LLMs: A Pre-Registered Psychometric Validity Screen},
  author={Cacioli, Jon-Paul},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

## License

Code: MIT. Documentation and pre-registration text: CC-BY-4.0.

## Generative AI

Claude (Anthropic) was used for analysis pipeline design, code generation, and manuscript preparation. All scientific decisions were made by the author.
