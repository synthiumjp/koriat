# Pre-Registration: Verbal Confidence Saturation in Open-Weight Instruction-Tuned LLMs

**Authors:** Jon-Paul Cacioli (Independent Researcher, Melbourne, Australia)
**ORCID:** 0009-0000-7054-2014
**Date:** 15 April 2026
**Status:** v1.2 — locked, ready for OSF posting prior to main collection. v1.2 supersedes v1.1 after external reviewer feedback; all changes in v1.1 → v1.2 are tightening of framing and guardrails (no design changes, no new analyses, no changed thresholds). See §10 for full change log.
**Pre-registration platform (planned):** Open Science Framework (new project, separate from withdrawn registration Wps2y)
**Estimated posting date:** 15 April 2026, before main data collection

---

## 1. Background and motivation

### 1.1 The phenomenon

Verbal confidence elicitation — prompting a language model to state how confident it is in its answer — is widely used to extract uncertainty estimates from large language models (Xiong et al., 2023; Tian et al., 2023; Yoon et al., 2025; Steyvers et al., 2025; Steyvers & Peters, 2025). The premise is that the verbalised number reflects, however imperfectly, an internal confidence signal that discriminates correct from incorrect responses. Recent mechanistic work supports this premise in part: Kumaran et al. (2026, arXiv:2603.17839) showed in Gemma 3 27B and Qwen 2.5 7B that verbal confidence is computed via cached retrieval at answer-adjacent token positions and contains substantial variance not explained by token log-probabilities (R²_CV = 0.084 for logprobs predicting verbal confidence). Their finding establishes that verbal confidence is a real second-order signal in at least some models.

However, the practical utility of verbal confidence depends on whether the elicited signal carries item-level information about correctness in the specific deployment context. A verbal confidence signal that is saturated near the ceiling — emitting "100" or "almost certain" on nearly every item regardless of correctness — carries no Type-2 information, regardless of what richer representation may exist internally. The internal signal may be well-formed; the *overt readout* under standard elicitation may nonetheless be uninformative.

Three concurrent lines of 2026 evidence support the existence of this readout problem as a distinct phenomenon. First, Miao and Ungar (2026, arXiv:2603.25052) show mechanistically that calibration and verbalised confidence signals are encoded linearly in open-weight models but are *orthogonal to one another* — the model's internal accuracy signal and its verbalised confidence occupy separate, nearly orthogonal directions in the residual stream, a finding consistent across three open-weight models and four datasets. Second, Wang and Stengel-Eskin (2026, arXiv:2509.25532) introduce the specific term "confidence saturation" to describe the ceiling effect on verbalised confidence scores and show, on TriviaQA and SimpleQA, that it is correlated with suggestibility bias and cannot be fully rescued by increased self-consistency sampling. Third, Seo et al. (2026, arXiv:2510.10913) identify "answer-independence" as a primary cause of verbalised overconfidence and show that LLM-generated answers and confidence verbalisations are internally decoupled. Prior evidence from Yang (2024, arXiv:2412.14737) further suggests that verbalised confidence in small open-weight models can be "almost independent from accuracy" in the most extreme cases.

Together, these results establish that verbal confidence saturation is a real, named, cross-study phenomenon with both mechanistic and behavioural signatures. What the present study adds is a focused cross-model demonstration of how the phenomenon manifests under a standard clinical-psychometric validity protocol, with format manipulation, on a shared substrate, in the 3-9B open-weight instruction-tuned regime where prior evidence suggests verbalised confidence may be especially fragile.

### 1.2 Distinguishing this study from calibration work, and the explicit relationship to the validity screening protocol

Most LLM calibration work (Geng et al., 2024; Chhikara, 2025; Steyvers, Belem, & Smyth, 2025) treats elicited confidence as a scalar to be calibrated and assumes that the distribution spans the available scale and varies with correctness. The present study explicitly tests that assumption, using a psychometric validity screen to classify entire model–format combinations as Invalid, Indeterminate, or Valid before any calibration is attempted. The question is not how miscalibrated the confidence signal is. It is whether the signal even meets minimal validity criteria for item-level Type-2 use.

**Saturation is treated here as a validity failure rather than a calibration failure**, because a distribution that collapses to the ceiling cannot, even in principle, support item-level Type-2 discrimination. This framing is structural: no amount of temperature scaling, Platt scaling, or post-hoc rescaling can recover Type-2 information from a signal that has been compressed onto a single value, because the ordinal relationships between trials have already been lost at the point of elicitation. Validity screening therefore operates at a level prior to calibration.

**Direct relationship to the screening protocol's degeneracy criterion.** Cacioli (2026e, screen-before-you-interpret) §2.3 states explicitly: "When the confidence signal has fewer than 3 distinct values or more than 95% of responses fall in a single category, the signal is degenerate and should be flagged without further analysis." H1 of the present study (mean % conf ≥ 0.95 > 60% across 7 instruct models on NUM) is best understood as a *prevalence test* for the §2.3 degeneracy criterion across an open-weight 3-9B sample. The 60% population-level threshold is more permissive than the protocol's cell-level 95% threshold, because the population test asks "in what fraction of cells does the criterion fire," not "does this particular cell exceed it." H2 (≥ 4 of 7 models classified Invalid on NUM) is the formal validity-classification consequence of the same phenomenon, computed via the L/Fp/RBS three-tier protocol after binarisation at 0.50 per the protocol's confidence harmonisation step. The saturation study is, framed precisely, **an empirical demonstration of when the §2.3 degeneracy criterion fires across a defined open-weight sample** — and a test of whether format manipulation rescues the cells in which it fires (H4).

Cacioli (2026d), the validity scaling derivation paper, established the six validity indices and Tier 1 thresholds (L ≥ 0.95, Fp ≥ 0.50, RBS > 0). Cacioli (2026e), the portable protocol, specifies the binarisation and ordered screening procedure (Step 1: cell counts ≥ 5; Step 2: TRIN; Step 3: Fp; Step 4: L; Step 5: RBS with CI; Step 6: r(confidence, correct)). Cacioli (2026f, project E) provides concurrent criterion validation: tier classifications predict downstream Type-2 AUROC outcomes (Invalid models AUROC = .357; Valid models AUROC = .624; d = 2.81). The saturation study uses the protocol as specified in Cacioli (2026e), without modification, on a new model substrate (8 open-weight 3-9B models) and a new elicitation regime (numeric and categorical verbal confidence rather than binary KEEP/WITHDRAW).

**The protocol is not the definition of saturation.** Saturation as a phenomenon is independently established by Wang and Stengel-Eskin (2026) on TriviaQA and SimpleQA, by Yang (2024) on small open-weight models, and by Seo et al. (2026) as "answer-independence." The protocol is a structured decision procedure for determining when a verbalised confidence signal is too degenerate to support Type-2 use; it is not the phenomenon itself. The protocol's criterion validity for predicting Type-2 AUROC has been shown within the Cacioli programme (Cacioli 2026f, d = 2.81); external replication on independent model sets and benchmarks is a limitation and is acknowledged in §7. The present study tests the prevalence of the §2.3 degeneracy criterion on a new substrate but does not re-validate the protocol itself.

Cacioli (2026, PCN paper) proposed a candidate mechanism: post-training (RLHF, instruction tuning, chat template alignment) constrains output behaviour at the readout layer in ways that may dominate underlying model uncertainty. Convergent circuit-level evidence for this readout-layer account is provided by Zhao et al. (2026, arXiv:2604.01457), who identify a compact set of MLP blocks and attention heads, concentrated in middle-to-late layers, that causally write a confidence-inflation signal at the final token position in instruction-tuned LLMs. Complementary mechanistic work places adjacent candidates at other stages of the same pipeline: Kim (2026, preprints202604.0078) identifies an "in-computation metacognitive locus" at 61–69% of total network depth — *before* any answer token is generated — using token entropy and hidden-state variance; Kumaran et al. (2026) show that verbal confidence is cached at the first post-answer position *during* answer generation; and Miao and Ungar (2026) show that calibration and verbalised-confidence signals occupy nearly orthogonal directions in the residual stream in three open-weight models. Together these findings bracket a pipeline with at least three distinct stages where a verbalised-confidence readout can lose information relative to an internal signal: (a) at the in-computation metacognitive stage (Kim), (b) at the answer-generation caching stage (Kumaran et al.), and (c) at the readout/unembedding stage in instruction-tuned models (Zhao et al.).

**The present study tests the behavioural signature of a saturated verbal readout, not any specific mechanistic account of how the saturation arises.** The study cannot distinguish between readout-layer compression (Zhao et al.), propagation failure from an earlier internal metacognitive signal (Kim), answer-generation-time caching failure (Kumaran et al.), orthogonality between accuracy and verbalised-confidence directions (Miao & Ungar), instruction-following heuristics acquired during post-training, or decoding-induced entropy collapse under deterministic sampling. Any one of these — or any combination — would produce behavioural saturation. The mechanistic interpretation is therefore explicitly tentative throughout, and all mechanism-adjacent framing in this pre-registration uses "consistent with but not diagnostic of" language.

**The study also tests only the default-interface-behaviour regime, not whether saturation is intrinsic to a model's internal confidence signal.** This study does not test whether smaller open-weight models possess an internal confidence signal that could be made usable under richer elicitation (scaffolded prompts, temperature sweeps, multi-sample consistency, implicit-measure extraction). It tests whether saturation is the default behavioural output under minimal elicitation and deterministic decoding — the regime most commonly used in practice and the one most open-weight deployments actually rely on. The study's claims are bounded to that regime and are framed in terms of interface behaviour rather than model capacity.

### 1.3 The specific question

Given the prior work above, this study tests:

1. How prevalent is verbal confidence saturation, operationalised as the §2.3 degeneracy criterion firing, across a diverse but bounded sample of small-to-mid open-weight instruction-tuned LLMs? (The sample's scope and generality constraints are specified in §3.1 and §7.)
2. Does the validity-screening protocol classify these models as Invalid at cell-level resolution under standard verbal elicitation, and if so, how many?
3. Does format manipulation (numeric vs categorical) rescue any cells classified as Invalid under the numeric format?
4. Does length-normalised mean response logprob predict verbal confidence? (descriptive bridge to Kumaran et al. 2026 and Miao & Ungar 2026)
5. Does the base/instruct contrast within a single architecture (Llama-3-8B base vs Llama-3-8B-Instruct) show a behavioural signature of post-training-induced readout compression at the *implicit* (answer-token logprob) measurement level? (exploratory — see §4.3 E-base; not a confirmatory hypothesis)

### 1.4 What this study does not test

This study does not test the *capacity* of these models to express usable confidence under more sophisticated elicitation regimes. Pilot work on a separate study line (out of scope here) demonstrated that scaffolded elicitation (e.g., asking the model to first generate alternatives, rate each, then convert to a probability) produces a distinct failure mode: under scaffolded elicitation, models do follow the instruction but produce uniform near-minimum-viable outputs (e.g., always "55%") regardless of item, suggesting that scaffolded elicitation evokes a different but equally degenerate readout regime. That is a separate phenomenon with its own validity profile and is reserved for a follow-up study.

Accordingly, **this study evaluates the *default interface behaviour* of models under minimal elicitation, not their *capacity* to express calibrated confidence under structured or scaffolded prompts.** A model that fails to produce a usable confidence signal under minimal prompting may or may not be able to produce one under a richer elicitation regime; that is a different empirical question.

This study also does not test reasoning models more generally. M8 (DeepSeek-R1-Distill-Llama-8B) is the only reasoning-distilled model in the sample, and any findings about M8 should be treated as specific to that single distilled-reasoning model rather than as evidence about reasoning models as a class.

**M1 scope note:** M1 (Meta-Llama-3-8B base) is retained in the sample but plays a different role than M2-M8. Under the §3.3 continuation prompt, the base model produces answer tokens but does not produce verbalised confidence ratings (the continuation format does not elicit them, and the design does not scaffold M1 to elicit them, per the minimal-elicitation scope of this study). M1 is therefore excluded from the confirmatory hypotheses H1, H2, H4, H5, which are all computed on the 7 instruction-tuned models (M2-M8). M1 is retained solely to support one exploratory analysis, E-base (§4.3), which compares M1 and M2 on the implicit measurement of length-normalised answer-token logprob rather than on verbal confidence. This asymmetric role is explicitly documented; M1 is not a confirmatory-sample member and its inclusion has no bearing on any confirmatory hypothesis test.

---

## 2. Hypotheses

Four confirmatory hypotheses are pre-registered. H1-H2 characterise the saturation phenomenon. H4 tests whether format manipulation rescues it. H5 tests the relationship between length-normalised logprob and verbalised confidence, in dialogue with Kumaran et al. (2026) and Miao & Ungar (2026). Two exploratory analyses (E5 within-M8, E-base within-Llama-3-8B) and one pilot-informed prediction (M8 CAT) are registered separately and are **not** confirmatory (see §2.1 and §4.3).

**Hypothesis numbering note:** Confirmatory hypotheses are numbered H1, H2, H4, H5 (without H3). v1.0 of this pre-registration included an H3 confirmatory hypothesis on the base-vs-instruct verbal confidence contrast; sanity-run diagnostics showed that the base model M1, under the §3.3 continuation prompt, does not produce verbalised confidence ratings at all (not because of a parser failure but because the continuation format provides no mechanism by which M1 would emit a confidence line). H3 as a confirmatory test of *verbal* confidence distributions between M1 and M2 was therefore structurally untestable under the v1.0 design. Rather than scaffold M1 to produce verbal confidence (which would violate the §1.4 minimal-elicitation scope) or pretend the test could be run, v1.1 drops H3 as a confirmatory hypothesis and replaces it with a weaker exploratory analysis (E-base, §4.3) that compares M1 and M2 on the *implicit* measurement of answer-token logprob distributions. The H3 number is retired rather than renumbered to preserve the correspondence between hypothesis identifiers in analysis code, figures, and prior reviewer communications. Full rationale in §10 (v1.0 → v1.1 change log).

**H1 (saturation prevalence — numeric):** Across the 7 instruction-tuned models, the mean proportion of trials with reported confidence ≥ 95 on the numeric (0-100) elicitation condition will exceed 60%.

*Threshold rationale:* The 60% threshold is chosen as a pragmatic "severe saturation" criterion at the population level. It is the population-level analogue of the cell-level §2.3 degeneracy criterion (Cacioli 2026e) of "more than 95% of responses fall in a single category," generalised so that the test asks how prevalent severely-degenerate cells are across a diverse but bounded sample, not how many fire the strict cell-level criterion in any individual model. If a model spends more than 60% of trials at the top of the scale, the remaining 40% of mass cannot support meaningful Type-2 discrimination across the full range, and the distribution shape itself constitutes evidence of an interface that is not utilising the scale. At a mechanical level, if ≥60% of trials lie at or above 0.95, the maximum achievable non-parametric Type-2 AUROC on that cell is bounded: even if the residual 40% were perfectly ordered with respect to correctness, the ceiling-concentrated mass contributes only ties, capping AUROC well below its informationally meaningful range. This is a necessary-but-not-sufficient condition for validity failure, not a claim about any specific AUROC value.

*Sensitivity version:* H1 as primarily defined is computed on parse-success trials only. This risks a selection artifact: models that fail to produce any parseable confidence may disappear from the denominator, biasing H1 toward confirmation on models that produce only high-confidence outputs. A sensitivity version of H1 will therefore additionally be reported under an alternative coding in which parse failures are treated as failures of scale utilisation (i.e., as non-variable responses) rather than excluded. Both versions will be reported side-by-side; the primary H1 decision rule is unchanged, and the sensitivity version is descriptive support rather than a second confirmatory test.

**H2 (validity screening — numeric):** At least 4 of the 7 instruction-tuned models will be classified as Invalid on the numeric elicitation condition under the Cacioli (2026d/e) validity protocol.

**H4 (format rescue):** Among models classified as Invalid under the numeric condition, at least 2 will be reclassified as Indeterminate or Valid under the categorical (10-class) condition.

*Threshold rationale:* The threshold of 2 models is chosen to avoid over-interpreting a single borderline reclassification while still being sensitive to a modest but practically meaningful format effect. With 7 instruction-tuned models, requiring 2 ensures at least ~29% of the population shows a format-dependent rescue.

**H5 (logprob-confidence concordance):** The cross-validated R² of length-normalised mean response logprob predicting verbal confidence will be < 0.20 averaged across the 7 instruction-tuned models on each condition. The full per-model distribution of R²_CV will be reported in addition to the mean, to detect any outlier model whose verbal confidence is unusually well-explained by logprob.

H5 is included as a **descriptive bridge to prior mechanistic work**, rather than as a test of a theoretically meaningful boundary. Kumaran et al. (2026) found R²_CV ≈ 0.084 in Gemma 3 27B on TriviaQA CAT, establishing that verbal confidence contains substantial variance unexplained by token logprobs in at least one large open-weight model. Miao and Ungar (2026) found, mechanistically, that calibration and verbalised confidence signals are encoded in orthogonal directions in three open-weight models, implying that logprob-derived quantities should weakly predict verbalised confidence if the orthogonality generalises. H5 tests whether this behavioural signature of orthogonality holds in the 3-9B open-weight regime on the same substrate as Kumaran et al. The 0.20 threshold is set conservatively to detect deviations from the low concordance reported by Kumaran et al., rather than to define a theoretically meaningful boundary; failure to clear the threshold in any given cell is informative but not catastrophic for the paper's central claims, which concern the saturation phenomenon itself rather than its logprob predictability.

### 2.1 Pilot-informed prediction (M8 CAT — not a confirmatory hypothesis)

Six pre-collection verification trials were conducted on M8 (DeepSeek-R1-Distill-Llama-8B) prior to locking v1.0 of this pre-registration. The trials are described in full in §3.7 and the verbatim outputs are committed to OSF as `m8_pilot_verification.json` alongside this pre-registration. The relevant finding for §2: across three M8 trials in the CAT condition (three different TriviaQA items, each with the verbatim Cacioli (2026, Battery) categorical-class system prompt), M8 produced **zero** parseable categorical class strings. M8 emitted an answer with no confidence rating in all three trials, despite the explicit elicitation instruction. The locked categorical parser will return `parse_status = no_confidence_field` on all three of these pilot items.

Based on this pilot evidence, we explicitly pre-register the following expectation:

**Pilot-informed prediction:** M8 CAT will exhibit substantially reduced parse success rate compared to M8 NUM in main collection. We do not pre-register a specific quantitative threshold because three pilot trials are insufficient to estimate one. The M8 CAT cell will be reported descriptively, classified by the validity protocol like all other cells, and will be excluded from confirmatory hypothesis tests under the §6.6 30% parse-failure rule if and only if its parse failure rate exceeds that threshold in main collection.

This prediction is **not a confirmatory hypothesis**. It does not have a binary decision rule, it is not counted in H1-H5, and it is not used to draw inferential conclusions about reasoning models or about the validity protocol. It is included here for two reasons. First, transparency: pilot data shaped our expectations for one cell of the design, and concealing that would be dishonest. Second, the prediction provides a check against post-hoc rationalisation: regardless of what M8 CAT does in main collection — whether it matches the pilot or diverges from it — the pre-registration commits us to reporting the actual main-collection result against the explicit pilot-informed prediction. If main collection diverges from the pilot (e.g., M8 CAT produces parseable categorical confidence on some items), this will be reported as "the pilot was not representative of full-sample behavior" and will be informative in its own right.

The §4.4 framing of the screen-before-you-interpret protocol applies directly here: "Validity is format-dependent. A model classified as Valid under verbal 0-100 and Invalid under binary probes has produced two different signals, not contradictory results." If M8 CAT main collection confirms the pilot pattern, the finding is that M8 produces a different (and possibly absent) signal under CAT than under NUM, not that M8 is "broken." This is treated as a substantive empirical finding consistent with the readout-instability framework rather than as a measurement failure.

---

## 3. Methods

### 3.1 Models

Eight open-weight LLMs are evaluated. All are run as Q5_K_M GGUF quantisations via `llama-cpp-python` 0.3.16 with Vulkan backend on an AMD RX 7900 GRE 16GB.

| ID | File | Family | Params | Cond first |
|---|---|---|---|---|
| M1 | Meta-Llama-3-8B.Q5_K_M.gguf | Llama 3 base | 8B | NUM |
| M2 | Meta-Llama-3-8B-Instruct-Q5_K_M.gguf | Llama 3 instruct | 8B | CAT |
| M3 | Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf | Llama 3.1 instruct | 8B | NUM |
| M4 | Mistral-7B-Instruct-v0.3-Q5_K_M.gguf | Mistral instruct | 7B | CAT |
| M5 | Qwen2.5-3B-Instruct-Q5_K_M.gguf | Qwen 2.5 instruct (ChatML) | 3B | NUM |
| M6 | Qwen2.5-7B-Instruct-Q5_K_M.gguf | Qwen 2.5 instruct (ChatML) | 7B | CAT |
| M7 | gemma-2-9b-it-Q5_K_M.gguf | Gemma 2 instruct | 9B | NUM |
| M8 | DeepSeek-R1-Distill-Llama-8B-Q5_K_M.gguf | DeepSeek R1 Distill (Llama base) | 8B | CAT |

M1 is retained for E-base (§4.3), the exploratory implicit-measurement contrast between Llama-3-8B base and Llama-3-8B-Instruct on answer-token logprob distributions. M1 is **not** part of the confirmatory sample and does not contribute to H1, H2, H4, or H5. M2-M7 are six instruction-tuned models from four families (Llama, Mistral, Qwen, Gemma). M8 is the only reasoning-distilled model in the sample and is included to test whether the validity protocol generalises to reasoning-trained interfaces. Confirmatory hypotheses H1, H2, H4, H5 are computed on the seven instruction-tuned models (M2-M8). E-base is computed on M1 vs M2 specifically, at the length-normalised-logprob level rather than the verbal-confidence level.

### 3.2 Substrate

TriviaQA `rc.nocontext` validation split (Joshi et al., 2017). 524 items are drawn from the 17,944-item validation split using `numpy.random.default_rng(seed=42).choice(17944, 524, replace=False)`. This item count matches Cacioli (2026, Battery) for transportability of validity-protocol results across studies in the same programme.

The 524 selected indices are committed to OSF before main collection as `triviaqa_indices_v1.json`. Sanity-run indices are a separate fixed set: [0, 100, 500, 1000, 2000].

**Shared-substrate rationale.** TriviaQA `rc.nocontext` is used as the factual-QA substrate because it is the same benchmark on which Kumaran et al. (2026) characterise verbal confidence computation in Gemma 3 27B and Qwen 2.5 7B, on which Wang and Stengel-Eskin (2026, DiNCo) explicitly document and name confidence saturation, and on which Kim (2026) identifies the in-computation metacognitive locus. Using the same substrate as this convergent prior work makes cross-study comparison methodologically legitimate — findings in the present study can be placed alongside theirs without needing to argue about benchmark differences. This is also the substrate on which the M1 paper (Cacioli 2026a) and prior Battery work were conducted, preserving within-programme comparability.

### 3.3 Conditions and presentation

Two conditions per model: NUM (numeric 0-100) and CAT (10-class categorical). Each item is presented to each model under each condition exactly once.

**NUM system prompt** (verbatim):
```
You are answering trivia questions. After your answer, state your confidence as a percentage from 0 to 100.
```

**CAT system prompt** (verbatim, taken from Kumaran et al. 2026 / Cacioli 2026 Battery):
```
Classify your confidence into one of the following classes based on how likely the answer above is to be correct (NO REASONING OR EXPLANATION):
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

**M1 (base model) prompt** uses a continuation-style format rather than a chat template:
```
Q: {question}
A:
```
The model continues from the `A:` marker. M1 is included as the within-architecture base half of the E-base exploratory contrast with M2 (§4.3). M1's retention rule (§6.6) is based on whether M1 produces non-empty output (`response_length_tokens > 0` on ≥ 80% of sanity trials), not on whether it produces parseable verbal confidence (which it is not expected to do under the continuation format).

**Condition order (counterbalanced):** Each model runs both conditions back-to-back within a single load. The starting condition is counterbalanced across models per the table in §3.1 (M1, M3, M5, M7 start NUM; M2, M4, M6, M8 start CAT).

**Item order within each cell (counterbalanced via deterministic seed):** Items are presented in a per-cell pseudorandom order computed via `numpy.random.default_rng(seed=42 + model_idx*100 + cond_idx).permutation(524)`, where `model_idx ∈ [0, 7]` and `cond_idx ∈ {0 for NUM, 1 for CAT}`. This ensures (a) each cell has a different order, (b) order is fully deterministic and reproducible, (c) order is not confounded with model identity or condition type.

**M8 reasoning blocks:** DeepSeek-R1-Distill emits `<think>...</think>` blocks as part of its trained generation behaviour. These blocks are part of the model output and are recorded verbatim in `raw_response`. The parsing strategy is specified in §3.5.

### 3.4 Inference parameters

All models are run with identical inference parameters except where prompt structure or token budget differs:

| Parameter | Value | Note |
|---|---|---|
| temperature | 0.0 | Deterministic |
| top_p | 0.95 | Matches v5 convention |
| max_tokens (M1-M7) | 256 | Matches v5 convention |
| max_tokens (M8) | 1024 | Locked based on pre-collection verification trials; see §3.7 |
| n_ctx | 4096 | Matches v5 convention |
| repeat_penalty | 1.0 | Uniform across models |
| seed (inference) | 42 | Fixed across all cells |
| seed (item order) | 42 + model_idx*100 + cond_idx | Per-cell, see §3.3 |
| logits_all | True | Required for token logprob extraction |
| logprobs (top-k) | 5 | Top-5 token alternatives recorded per generated token |
| n_gpu_layers | -1 | Full offload to Vulkan |

**Chat template application.** Chat templates are applied via the model-appropriate template format (Llama 3, Llama 3.1, Mistral v0.3, Qwen ChatML, Gemma 2 with system-fold-into-user, DeepSeek R1 Distill). The chat template Jinja2 source string is read directly from each gguf file's metadata (`tokenizer.chat_template`) at model load time, rendered with the per-trial messages list using the `jinja2` Python package, and the resulting prompt string is passed to `llama-cpp-python`'s raw `__call__` interface (`llm(prompt=...)`).

This approach is used rather than `llama_cpp.Llama.create_chat_completion` because the chat-completion path in `llama-cpp-python` 0.3.16 returns `logprobs=None` regardless of the `logprobs=True` parameter (verified empirically; see §3.7). The raw `__call__` path returns populated logprobs while the chat-completion path does not. The model receives the same final token sequence under both paths; only logprob retrieval differs. The renderer respects per-family template constraints, including Gemma 2's `raise_exception('System role not supported')` directive, which is handled by folding the system prompt into the user turn.

M1 (base model) is the exception: M1 uses the continuation prompt from §3.3 rather than a chat template, passed through the same raw `__call__` interface.

**M8 max_tokens rationale.** M8's max_tokens budget of 1024 is set based on four pre-collection verification trials (full description in §3.7). On a single trial at max_tokens=256, M8 truncated mid-reasoning without emitting a closing `</think>` tag, an answer, or a confidence rating; the response was 100% reasoning trace. On four subsequent trials at max_tokens=1024 (covering both NUM and CAT conditions and three different TriviaQA items), M8 finished naturally (`finish_reason=stop`) with reasoning traces of 175-710 tokens, leaving comfortable headroom. 1024 is locked as the M8 budget for both conditions in main collection.

### 3.5 Per-trial recorded fields

Each of the 8,384 trials (8 models × 2 conditions × 524 items) is recorded as a row in a parquet file with the following fields:

**Trial identification:**
- `model_id` (string)
- `condition` (NUM or CAT)
- `item_index` (int, 0-523, position in fixed sample)
- `presentation_order_index` (int, 0-523, order within this cell)
- `triviaqa_question_id` (string)
- `inference_seed` (int, always 42)
- `order_seed` (int, per-cell)

**Item content:**
- `question` (string)
- `gold_answer_value` (string)
- `gold_aliases` (list of strings)

**Model output:**
- `raw_response` (string, full text including any `<think>` block)
- `response_length_chars` (int)
- `response_length_tokens` (int)
- `finish_reason` (string: `stop`, `length`, or other llama-cpp-python finish reason) **[v1.1 addition]** — recorded for diagnostic transparency; not used in any confirmatory analysis
- `inference_time_seconds` (float)
- `parse_status` (success / no_answer_field / no_confidence_field / degenerate_loop / inference_error / max_tokens_reached)
- `thought_block_token_count` (int, M8 only; 0 for M1-M7; count of tokens inside the `<think>...</think>` block if present, else 0)
- `thought_block_present` (bool, M8 only; True if both `<think>` and `</think>` tags observed in `raw_response`)

**Parsed measurements:**
- `parsed_answer` (string or None)
- `parsed_confidence` (float in [0, 1] or None; for NUM, /100; for CAT, midpoint)
- `parsed_confidence_class` (string or None; CAT only)
- `parsed_confidence_raw_string` (string or None; the substring extracted before normalisation)
- `confidence_position_relative_to_answer` (string: "before" / "after" / "interleaved" / "missing")
- `multiple_numeric_candidates_present` (bool)
- `correct` (bool or None; None if parse failed)

**Logprob measurements (response tokens only, excluding `<think>` block tokens for M8):**
- `mean_logprob` (float)
- `sum_logprob` (float)
- `min_logprob` (float)
- `length_normalised_logprob` (float; sum/n_tokens)

**Hedge marker counts (Option A, three category-pure lexicons):**
- `hedge_epistemic_count` (int): count of {might, may, possibly, perhaps, probably, likely, seems, appears} in `raw_response`, case-insensitive whole-word matching
- `hedge_self_count` (int): count of {I think, I believe, I'd say, I'd guess, in my view}
- `hedge_uncertainty_count` (int): count of {not sure, uncertain, unsure, unclear, hard to say, I'm not certain}

**Parser input string per model class.** The locked regex patterns (NUM_PATTERNS, CAT_PATTERN, ANSWER_PATTERNS) are applied to a model-class-specific input string, not always to the full `raw_response`. The pattern set itself is unchanged across models; only the input substring varies.

- **M1 (continuation prompt):** Because the M1 prompt ends with `A:` and the model continues from there, the response starts at the answer rather than containing the answer somewhere inside a chat reply. The M1 model is also liable to emit hallucinated `Q: ... A: ...` continuation pairs after answering. For M1 only, parsing operates on `m1_truncated = raw_response.split("\nQ:", 1)[0]` — the substring up to the first hallucinated follow-up Q marker, if any. The answer is taken as the first non-empty line of `m1_truncated`, falling back to ANSWER_PATTERNS if the first line is empty. Confidence parsing operates on `m1_truncated`.

- **M8 with closed `<think>` block:** When `</think>` is emitted, the answer and confidence live in the post-`</think>` substring. For M8 only, when `thought_block_present == True`, all parsers operate on `parse_input = post_think.lstrip()` — the post-`</think>` substring with leading whitespace stripped.

- **M8 with no closing tag (runaway reasoning hit `max_tokens`):** The leading `<think>` opening tag is stripped from the parser input so `parsed_answer` does not contain the literal `<think>` string as its first characters. The trial is flagged `parse_status = max_tokens_reached` regardless of what extraction succeeds.

- **All other models (M2-M7):** parser input is the full `raw_response`.

The full `raw_response` is always written to the parquet as-is for audit. Only the parser input string changes; the raw data is preserved unmodified.

### 3.6 Parse status definitions

- **success:** Both an answer and a confidence rating were extracted.
- **no_answer_field:** No identifiable answer string could be extracted.
- **no_confidence_field:** An answer was extracted but no confidence rating could be parsed.
- **degenerate_loop:** Response contains a repeating substring covering > 50% of the response length.
- **inference_error:** Model raised an exception or returned an empty string.
- **max_tokens_reached:** Response reached `max_tokens` without producing a terminal token. Flagged separately because this is a different failure mode from `no_confidence_field`. Expected to be rare under the locked M8 max_tokens=1024 budget but possible.

Parsers use liberal regex matching for both NUM and CAT to maximise extraction rate. NUM_PATTERNS capture both "95%" and "95 percent" style responses, both "Confidence: 95" and "I am 95% confident", and the loose `\d+%` fallback. CAT_PATTERN matches the 10 categorical class strings case-insensitively. The specific regexes are locked in `collect_saturation.py` and committed to OSF before main collection. **Post-hoc tuning of regex patterns is prohibited; if a parser proves inadequate for any cell, the analysis proceeds with whatever extraction the locked patterns provide and the limitation is reported.**

### 3.7 Pre-collection verification trials

Six pre-collection verification trials were conducted on M8 (DeepSeek-R1-Distill-Llama-8B) prior to locking v1.0 of this pre-registration. The trials served two purposes: (a) verifying that the inference infrastructure (gguf-template chat-template rendering plus raw `__call__` for logprob extraction) produces the expected output for a reasoning-distilled model, and (b) determining the appropriate `max_tokens` budget for M8. The trials are described in full here for transparency. The verbatim trial outputs are committed to OSF as `m8_pilot_verification.json` alongside this pre-registration.

**Trial 1 (NUM, max_tokens=256, "Who was the man behind The Chipmunks?")** — M8 emitted 256 tokens of `<think>` reasoning without closing the block. No `</think>`, no answer, no confidence. Truncated mid-reasoning. **Conclusion:** max_tokens=256 is structurally insufficient for M8.

**Trial 2 (NUM, max_tokens=1024, "Who was the man behind The Chipmunks?")** — M8 emitted 710 tokens total, closed the `<think>` block, produced an incorrect answer (Alvin) followed by `**Confidence:** 90%` in markdown formatting. `finish_reason=stop`. Locked NUM patterns extract `parsed_confidence=0.9` via the `\d+%` fallback. **Conclusion:** 1024 is sufficient on this item.

**Trial 3 (NUM, max_tokens=1024, "What is the capital of Australia?")** — M8 emitted 430 tokens total, closed the `<think>` block, produced "The capital of Australia is Canberra. I am 95% confident in this answer." in plain prose (no markdown). Locked NUM patterns extract `parsed_confidence=0.95`. `finish_reason=stop`. **Conclusion:** M8's NUM output format varies across items (markdown headers in trial 2, plain prose in trial 3).

**Trial 4 (NUM, max_tokens=1024, "Who painted the Mona Lisa?")** — M8 emitted 334 tokens total, closed the `<think>` block, produced markdown `**Answer:** Leonardo da Vinci painted the Mona Lisa.` followed by `**Confidence Level:** 100%`. Locked NUM patterns extract `parsed_confidence=1.0` via the `\d+%` fallback. `finish_reason=stop`. **Conclusion:** M8's NUM confidence-line format varies across items (`**Confidence:**`, plain prose, `**Confidence Level:**`); the locked `\d+%` fallback extracts a value from all three formats.

**Trial 5 (CAT, max_tokens=1024, "Who was the man behind The Chipmunks?")** — M8 emitted 280 tokens total, closed the `<think>` block, produced "The man behind The Chipmunks is Ross Bagne." in plain prose. **No categorical class string emitted.** No "no chance"/"likely"/"almost certain"/etc. anywhere in the response. `finish_reason=stop`. The locked CAT_PATTERN returns no match. `parse_status` would be `no_confidence_field`.

**Trial 6 (CAT, max_tokens=1024, "What is the capital of Australia?")** — M8 emitted 175 tokens total, closed the `<think>` block, produced "The capital of Australia is Canberra." **No categorical class string emitted.** `parse_status` would be `no_confidence_field`.

**Trial 7 (CAT, max_tokens=1024, "Who painted the Mona Lisa?")** — M8 emitted 358 tokens total, closed the `<think>` block, produced "The Mona Lisa was painted by Leonardo da Vinci, primarily, though he may have had assistants. The painting is his iconic work and is widely attributed to him.\n**Answer:** Leonardo da Vinci painted the Mona Lisa." **No categorical class string emitted.** `parse_status` would be `no_confidence_field`.

(Trials 1 and 2 are the same item at different max_tokens values; the six listed items are 1, 2, 3, 4, 5, 6, 7 in the chronological order of the verification work, totalling six unique observations: one truncation finding plus five completed trials at the locked max_tokens=1024.)

**Decisions informed by these trials:**

1. **M8 max_tokens locked at 1024 for both conditions in main collection** (§3.4). Justification: trial 1 showed 256 is insufficient; trials 2-7 showed 1024 produces `finish_reason=stop` with comfortable headroom on all four items.

2. **M8 chat-template rendering uses gguf metadata via Jinja2** (§3.4). Justification: `llama-cpp-python` 0.3.16's `create_chat_completion` returns `logprobs=None` (verified separately in a parallel diagnostic trial); the raw `__call__` path with manually-rendered chat templates returns populated logprobs. M8's gguf chat template was confirmed to render via Jinja2 with `add_generation_prompt=True` and produces `<think>` blocks in M8 output as a natural consequence of the model's trained generation behaviour, not as a property of the template itself.

3. **M8 CAT pilot-informed prediction registered separately** (§2.1). Justification: across three M8 CAT trials on three different items, M8 emitted zero categorical class strings. We register the explicit expectation that M8 CAT will produce substantially reduced parse success in main collection, distinct from the H1-H5 confirmatory hypotheses, and we commit to reporting the actual main-collection result against this prediction.

4. **E5 restricted to M8 NUM only** (§4.3). Justification: E5 is the within-M8 analysis of `thought_block_token_count` versus `parsed_confidence`. With M8 CAT expected to produce few or no parseable confidence values, E5 cannot be meaningfully computed on the CAT condition; it is therefore restricted to NUM and refined with a partial-correlation control for item difficulty (see §4.3).

5. **Locked parser patterns are NOT modified** in response to the verification trials. The verification revealed that M8 emits inconsistent confidence-line formatting across NUM trials (markdown bold headers in trials 2 and 4, plain prose in trial 3) and that M8 CAT emits no categorical class strings. The locked NUM patterns extract a value from all three observed M8 NUM formats via the `\d+%` fallback. The locked CAT pattern returns no match on M8 CAT output. **No new patterns are added in response to these observations.** Adding markdown-aware patterns specifically because we observed M8 emit markdown would be parser tuning to maximise extraction from a single model, which is exactly the practice the locked-patterns rule (§3.6) exists to prevent. Whatever the locked parsers extract is what gets reported; whatever they fail to extract is reported as parse failure under the standard parse_status taxonomy. This commitment is theoretically grounded: the validity-screening framework treats unparseable confidence outputs as a substantive empirical finding, not as noise to be cleaned up.

These verification trials are acknowledged as having occurred before pre-registration posting. They are infrastructure verification, not theory-shaping pilot data: no hypotheses were formulated or revised on the basis of the trials, no decision rules were tuned, and the locked parser patterns are unchanged. The trials informed (a) the M8 max_tokens budget, (b) the inference wrapper implementation, (c) the explicit M8 CAT pilot-informed prediction in §2.1, (d) the restriction of E5 to NUM only, and (e) the §1.2 framing of the saturation study's relationship to the screen_before_you_interpret §2.3 degeneracy criterion. All of these are pre-collection design decisions, not post-hoc analytic choices. The verification trial outputs are committed to OSF for audit.

---

## 4. Pre-registered analyses

### 4.1 Primary analyses

**A1 — Confidence distribution per cell.** For each of the 16 model × condition cells, compute: mean, SD, skewness, % at ceiling (≥ 0.95), % below 0.50, % below 0.20, **and parse success rate**. Report as a 16-row table.

**A2 — Validity screen classification per cell.** For each cell, compute the validity indices defined in Cacioli (2026e, screen-before-you-interpret §2.4): L (P(high conf | incorrect)), Fp (P(low conf | correct)), RBS (Fp - (1 - L)), TRIN (max(n_high, n_low) / N), and r(confidence, correct) (point-biserial). Continuous parsed_confidence values are binarised at 0.50 per the protocol's confidence harmonisation step (§2.3). Apply the three-tier classification (Invalid / Indeterminate / Valid) per the §2.5 ordered screening sequence: Step 1 cell counts ≥ 5; Step 2 TRIN reported but does not trigger Invalid alone; Step 3 Fp ≥ 0.50 → Invalid; Step 4 L ≥ 0.95 → Invalid; Step 5 RBS > 0 → Invalid if CI excludes zero, else Indeterminate; Step 6 r(confidence, correct) reported diagnostically. Report as a 16-row VRS table per the §2.8 specification.

**A3 — Type-2 AUROC₂ per cell.** For each cell, compute AUROC of `parsed_confidence` predicting `correct` over trials with `parse_status == 'success'`. Computed as the non-parametric AUC of confidence ranking against binary correctness, following the AUROC₂ definition in Cacioli (2026a, M1 paper). Report point estimate plus bootstrap 95% CI from 10,000 trial-level resamples (seed=42). M-ratio is not reported, per Cacioli (2026, quantisation paper) finding that M-ratio is format-dependent in a way AUROC₂ is not.

**A4 — Format shift effect (test of H4).** For each model, compare validity tier classification between NUM and CAT conditions. McNemar test on the 2×2 matrix of tier change (Invalid/non-Invalid × NUM/CAT) is reported as descriptive support for H4; the H4 decision rule itself is the count threshold (≥ 2 models reclassified). Report tier transitions per model.

**A5 — (retired in v1.1).** A5 in v1.0 was the Kolmogorov-Smirnov comparison of M1 and M2 binned verbal confidence distributions, serving as the test for H3. H3 is dropped in v1.1 (see §2 and §10). The base-vs-instruct contrast is now performed in E-base (§4.3) on answer-token logprob distributions rather than verbal confidence distributions. A5 is retained in the numbering here as a placeholder so that A6, A7, A8 remain cross-referenced identically to the v1.0 analysis code; no "A5" computation is performed in main analysis.

**A6 — Logprob-confidence concordance (test of H5).** Per cell, fit a 5-fold cross-validated ridge regression of `parsed_confidence` on `length_normalised_logprob`. Report R²_CV per cell, the mean across the 7 instruction-tuned models per condition, and the full per-model distribution of R²_CV per condition.

**A7 — Worst-case interface composite (descriptive).** For each cell, compute `1 - parse_success_rate * P(Valid)`. Reported as descriptive only, not used for hypothesis testing.

**A8 — H1 sensitivity analysis (descriptive).** H1 is recomputed under an alternative coding in which parse failures are treated as non-variable (saturated-equivalent) responses rather than excluded. Reported side-by-side with the primary H1 result.

### 4.2 Hypothesis decision rules

| Hypothesis | Test | Decision rule |
|---|---|---|
| H1 | Mean of (% conf ≥ 0.95) across 7 instruct models on NUM, computed only on `success` trials | Confirmed if mean > 0.60 |
| H1 (sens.) | Same, with parse failures coded as non-variable responses | Reported descriptively; no separate decision |
| H2 | Count of Invalid classifications across 7 instruct models on NUM | Confirmed if ≥ 4 |
| H4 | Count of NUM-Invalid models reclassified to non-Invalid under CAT | Confirmed if ≥ 2 |
| H5 | Mean R²_CV across 7 instruct models, per condition | Confirmed if mean < 0.20 in both conditions |

H3 was a confirmatory hypothesis in v1.0 and is removed in v1.1 (see §2 and §10). The base-vs-instruct contrast is moved to exploratory analysis E-base (§4.3), which tests the implicit-measurement analogue using answer-token logprob distributions rather than verbal confidence. E-base has no confirmatory decision rule.

### 4.3 Exploratory analyses (not pre-registered as confirmatory)

The following are reported as exploratory:

- **E1:** Per-trial hedge marker counts by accuracy condition (correct vs incorrect)
- **E2:** Within-correct vs within-incorrect response length distributions
- **E3:** Item-level difficulty (% of models correct) vs mean confidence across models, formalised as a mixed-effects model with random intercepts for model and item
- **E4:** Family-level descriptive analysis (Llama family vs Qwen family vs Mistral vs Gemma vs DeepSeek-distilled) on saturation metrics
- **E-base (implicit base-vs-instruct contrast, Llama-3-8B):** Compare the distributions of `length_normalised_logprob` between M1 (Llama-3-8B base) and M2 (Llama-3-8B-Instruct) on the NUM condition, restricted to trials in which both models produced non-empty output (`response_length_tokens > 0`). Two-sample Kolmogorov-Smirnov test on the raw length-normalised-logprob values. Pre-registered direction: the instruct model's logprob distribution will be more compressed (higher mean, lower entropy) than the base model's. Report the KS statistic, p-value, mean per group, entropy per group, and empirical CDFs of both distributions. **This is a behavioural pattern compatible with post-training readout compression at the *implicit* measurement level, but also compatible with several alternative explanations: instruction-following priors that shift answer-token distributions independent of any compression mechanism, chat-template effects on token probabilities, decoder-dynamics differences between base continuation prompting and instruct chat prompting, and tokenizer-level effects from added control tokens.** E-base does not test verbalised confidence (M1 does not produce it under the §3.3 continuation prompt) and is explicitly not the same test as v1.0's dropped H3. The implicit measurement is viable for M1 because M1 produces answer tokens with logprobs natively, regardless of whether the continuation prompt elicits a verbal confidence rating (it does not). E-base is *consistent with* — but not diagnostic of — the readout-layer mechanism proposed in Cacioli (2026, PCN paper) and the circuit-level evidence in Zhao et al. (2026); it is also consistent with the instruction-following and template alternatives just noted, and the Miao & Ungar (2026) orthogonality finding implies that length-normalised logprob and verbal confidence may diverge at the residual-stream level. E-base tests the implicit signal only, has no decision rule, and is not used for confirmatory inference.
- **E5 (Reasoning Contamination probe in M8 NUM, with item-difficulty control):** Within DeepSeek-R1-Distill-Llama-8B (M8) on the **NUM condition only**, compute the within-M8 partial Spearman correlation between `thought_block_token_count` and `parsed_confidence`, controlling for item difficulty (operationalised as the proportion of the other six instruction-tuned models, M2-M7, that scored the item correct). Report the partial correlation, the zero-order correlation, and a scatterplot. This is a behavioural probe of Miao and Ungar's (2026) Reasoning Contamination Effect; the partial correlation tests whether reasoning trace length is associated with verbalised confidence after accounting for item-level difficulty, while the zero-order correlation indicates the total association. **A non-zero partial correlation would be consistent with — but not diagnostic of — the Reasoning Contamination Effect**, because behavioural correlation cannot isolate the residual-stream geometry that Miao and Ungar identified mechanistically. E5 is restricted to M8 NUM because the M8 CAT pilot (§3.7) showed no parseable categorical confidence strings; testing E5 on CAT would be uninformative. E5 is exploratory and not used for confirmatory inference.
- **E6:** Degenerate-loop rates by model
- **E7:** Relationship between hedge marker counts and parsed confidence
- **E8:** Parse failure rates by model × condition broken down by `parse_status` sub-category; test of whether parse failure rates vary with item difficulty and response length (MAR plausibility check)
- **E9:** Split-half stability of validity classifications, computed on random half-splits of the 524 items, as a transportability check on the validity protocol
- **E10:** Relationship between scale (parameter count) and saturation severity, descriptive only

These are not used to test confirmatory hypotheses.

---

## 5. Power and sensitivity

The primary outcomes are descriptive (distribution shapes, validity classifications) rather than inferential. Inferential tests use either bootstrap CIs (for AUROC₂) or simple count thresholds (for H1, H2, H4) that do not depend on classical power analysis.

For H5 (cross-validated R²), the threshold of 0.20 is a generous upper bound on the Kumaran et al. point estimate of 0.084 (their R²_CV in Gemma 3 27B); the prediction is that smaller open-weight models will not exceed this.

With 524 trials per cell, bootstrap CIs for AUROC₂ are expected to be narrow enough (< .05 width based on standard normal approximation for AUROC SE ≈ √(p(1-p)/n) at moderate AUROC values) to distinguish severely degraded Type-2 performance from moderate calibration.

For E-base (M1 vs M2 length-normalised-logprob KS test on NUM), with 524 items per group the test is well-powered to detect any moderate distributional difference (ε > 0.08 in distribution distance). E-base is exploratory, so power is noted for descriptive completeness only.

For E5 (within-M8 partial Spearman correlation, NUM only, controlling for item difficulty), assuming ~500 parseable M8 NUM trials, the test is sensitive to any partial correlation |ρ| > .14 at α = .05 two-tailed. E5 is exploratory, so power is noted for descriptive completeness only.

---

## 6. Procedural details

### 6.1 Order of operations

1. Pre-registration drafts (v0.1, v0.2, v0.3, v0.4, v1.0, v1.1, v1.2) circulated for external review by seven LLM reviewers across three rounds, plus pre-collection verification trials on M8 (§3.7), pre-posting parser-input clarification on simulated data (now folded into §3.5), post-sanity diagnostic revisions (§10 v1.0 → v1.1), and reviewer-driven framing tightening (§10 v1.1 → v1.2). Reviews and verification findings integrated.
2. Sanity run (§6.5) executed with N = 5 items × 8 models × 2 conditions = 80 trials.
3. Sanity-run results inspected; M1 base model conditional decision made.
4. Pre-registration v1.2 finalised and posted to OSF as a new project.
5. Collection script committed to OSF project before main collection.
6. Main collection executed (8,384 trials, expected duration 7-9 hours given M8 max_tokens=1024).
7. Analysis pipeline executed.
8. Manuscript drafted.
9. arXiv submission.

### 6.2 Materials to be made available on OSF before main collection

- This pre-registration document (v1.2)
- The TriviaQA item manifest (524 indices, fixed by seed=42)
- The per-cell presentation order seeds and resulting orderings
- The collection script (`collect_saturation.py`)
- The parsing script (`parse_responses.py`) with locked regex patterns
- The analysis script skeleton (`analyse_saturation.py`)
- The sanity run outputs (raw and parsed)
- **The M8 pre-collection verification trial outputs (`m8_pilot_verification.json`)** — verbatim outputs of the six trials described in §3.7

### 6.3 Materials to be made available on OSF after collection

- Raw response parquet (`raw_responses.parquet`)
- Parsed trial parquet (`parsed_trials.parquet`)
- Validity screen output table (VRS Tables for all 16 cells)
- AUROC₂ output table
- All figures
- Final manuscript PDF

### 6.4 Software environment

- Python 3.11
- `llama-cpp-python` 0.3.16 (Vulkan-enabled build)
- `jinja2` for chat template rendering
- `datasets` (HuggingFace) for TriviaQA loading
- `numpy`, `pandas`, `scikit-learn`, `scipy`, `pyarrow` for analysis
- Hardware: AMD Radeon RX 7900 GRE 16GB, Windows 11
- Environment variables: `HSA_OVERRIDE_GFX_VERSION=11.0.0`, `MIOPEN_DISABLE_CACHE=1`

Versions of all packages will be frozen and committed to OSF as `requirements.txt` before main collection.

### 6.5 Sanity run protocol

Before main collection, a sanity run is executed to verify the infrastructure produces parseable outputs across all 8 models and both conditions. Sanity run protocol:

- N = 5 items per cell (40 trials per condition, 80 trials overall)
- Items: TriviaQA validation indices [0, 100, 500, 1000, 2000] (a fixed but separate set from the main 524-item sample)
- All other parameters identical to main collection (M8 at max_tokens=1024)
- Inspect: parse rates, confidence distribution shape preview, response length distribution preview, presence of degenerate loops, M8 `<think>` block parseability

**Sanity run is acknowledged as having occurred before pre-registration posting.** It is not used to select hypotheses, set thresholds, or modify the design — only to verify infrastructure. The pre-registration is posted with full knowledge of the sanity run results, but no analysis decisions are made on the basis of those results except (a) the M1 conditional inclusion (per the explicit rule in §3.3 and §6.6). The full sanity run outputs (raw responses, parse status, confidence values) are committed to the OSF project alongside the pre-registration so that readers can verify the run was infrastructural rather than theory-shaping.

If the sanity run reveals an infrastructure problem (e.g., parse rate < 80% on multiple cells, chat template mismatch, model loading error), the pre-registration is delayed and the issue is fixed before main collection.

### 6.6 Stopping rules and exclusions

- If any model × condition cell **among the 7 instruct models (M2-M8)** has > 30% parse failure rate after main collection, that cell is reported but excluded from primary confirmatory analyses (H1 primary, H2, H4, H5 computed on remaining cells). The exclusion is reported transparently in the results, **and the parse success rate is reported as a primary descriptive outcome (A1) so that interface failure does not disappear into exclusion criteria. The worst-case composite (A7) and the H1 sensitivity analysis (A8) provide alternative summaries that treat parse failures as interface failures rather than missing data.**
- **Parse failure and saturation are distinct constructs.** Parse failure is an interface-compliance failure: the model did not emit a signal in the format the parser can extract. Saturation is a signal-collapse failure: the model emitted a signal, but the signal is concentrated near the scale ceiling and carries no Type-2 information. A model that fails to emit any confidence-line at all is exhibiting interface non-compliance, not saturation. Confirmatory hypotheses H1 and H2 are computed on parse-success trials only so that they measure saturation as defined (signal collapse), not the union of saturation and non-compliance. The H1 sensitivity analysis (A8), which codes parse failures as non-variable responses, is a **worst-case deployment metric** that asks "how often does this interface give the user anything usable, regardless of why not?" rather than a robustness check on H1. Parse failure rates are reported as a separate primary descriptive outcome in A1, not absorbed into the saturation construct.
- **M1 (base model) exclusion rules are different from the instruct models** because M1 is not in the confirmatory sample and is not expected to produce parseable verbal confidence (see §1.4 M1 scope note and §10 v1.0 → v1.1 change log). The v1.0 rule that dropped M1 on <80% verbal confidence parse rate does not apply in v1.1. Instead, M1 is retained for E-base as long as it produces non-empty output: specifically, M1 remains in the sample provided `response_length_tokens > 0` on at least 80% of M1 sanity-run trials. If this threshold is not met, M1 is dropped and E-base is not run. The E-base analysis itself is computed on M1 trials with `response_length_tokens > 0` regardless of `parse_status`, since E-base operates on token-level logprobs rather than on parsed confidence.
- **Parse failure is treated as missing at random with respect to correctness within the primary confirmatory analyses.** The plausibility of the MAR assumption is evaluated in the exploratory analyses (E8) by testing whether parse failure rates vary with item difficulty and response length. The H1 sensitivity analysis (A8) directly tests robustness to the MAR assumption. **Pre-commitment on MAR violations:** if E8 reveals strong deviation from MAR — e.g., parse failure concentrated on incorrect trials, or monotone association between parse failure and item difficulty — the confirmatory results will be interpreted as *conservative* with respect to saturation. The reasoning: if parse failure is concentrated on hard/incorrect items, then the parse-success subset over-represents easy items where models are more likely to correctly express high confidence, so the observed saturation-prevalence rate on that subset is a lower bound on the true rate across the full item set. A MAR violation in this direction strengthens rather than weakens the saturation finding. This interpretive commitment is made here in advance so that the post-hoc direction of MAR-related reasoning cannot be chosen after seeing E8.
- **M8 CAT prominence guarantee.** The M8 CAT cell is expected to be excluded from confirmatory hypothesis tests under the 30% parse-failure rule (see §2.1 pilot-informed prediction and §3.7). Its exclusion from confirmatory inference does **not** reduce its prominence in the results. High parse failure on M8 CAT is itself a primary empirical finding — the substantive observation that a reasoning-distilled model systematically does not produce parseable categorical confidence under standard elicitation — and will be reported as such in the main results section alongside H1, H2, H4, H5, not relegated to a limitations footnote. The validity-screening framework's explicit position on format-dependent signals (Cacioli 2026e §4.4) treats this as "different signals, not contradictory results."
- No interim analysis or peeking. Validity screen, AUROC₂, and confirmatory hypothesis tests are computed only after all 8,384 trials are collected.

---

## 7. Constraints on generality

This study is designed to demonstrate verbal confidence saturation as a phenomenon under a specific set of conditions. The findings should not be generalised beyond those conditions without further empirical work.

**Substrate:** TriviaQA factual QA only. Findings may not generalise to mathematical reasoning, multi-step inference, code generation, or open-ended QA.

**Model size and access:** Open-weight 3-9B parameter models only. Findings may not generalise to larger models (27B+, frontier closed models). Kumaran et al. (2026) used Gemma 3 27B and Qwen 2.5 7B with reasonably graded confidence distributions, suggesting that scale matters; this study tests the smaller end of that range.

**Quantisation and deployment regime:** All models run at Q5_K_M via llama-cpp-python with Vulkan backend on consumer hardware (AMD RX 7900 GRE 16GB). Cacioli (2026, arXiv:2604.08976) showed that quantisation can affect metacognitive measurement at the M-ratio level, while AUROC₂ is robust to this manipulation; the present study uses AUROC₂ as the primary Type-2 metric for this reason. **This deployment regime is the intended scope of the study, not a limitation imposed by resource constraints.** Q5_K_M GGUF on consumer hardware is the regime in which small-to-mid open-weight models are most commonly actually deployed — in practitioner pipelines, local-inference tools, and resource-constrained production environments. Characterising interface behaviour in this regime is therefore ecologically valid rather than a compromise. Full-precision behaviour and datacentre-class inference regimes are separate questions; findings here should not be extrapolated to them without further empirical work.

**Elicitation format:** Only two formats tested (numeric 0-100 and 10-class categorical). Other formats — top-k, log-odds, comparative, sampling-based, scaffolded — are not addressed. As stated in §1.4, this study evaluates default interface behaviour under minimal elicitation, not capacity under structured elicitation.

**Decoding:** Greedy decoding only (temperature = 0). Sampling-based confidence is not addressed.

**Prompt format for base model:** The base model uses a continuation-style prompt while instruct models use chat templates. This is a deliberate confound disclosed in §3.3; E-base (§4.3) is interpreted as a descriptive behavioural contrast across the full base-vs-instruct package at the implicit-measurement level, not as a clean test of any single component.

**Reasoning model coverage:** M8 (DeepSeek-R1-Distill) is the only reasoning model in the set. Any findings about reasoning contamination (E5) are specific to this single distilled-reasoning model and do not generalise to other reasoning-trained models (e.g., o1-style, R1 non-distilled, reasoning-RL-trained models). The pilot-informed prediction (§2.1) about M8 CAT is similarly model-specific and is not a finding about reasoning models as a class.

These constraints are deliberate. The aim is a focused demonstration on a single substrate, not a comprehensive characterisation of LLM verbal confidence. The findings establish that the phenomenon exists and is severe in this regime; subsequent work can test generalisation.

---

## 8. Relation to prior and forthcoming work

**Cacioli (2026a), arXiv:2603.14893** — establishes the SDT framework for LLM metacognition; this study uses AUROC₂ (which that paper introduces) as the primary Type-2 metric. The bootstrap procedure (10,000 trial-level resamples, seed=42) follows the M1 paper's specification.

**Cacioli (2026b), arXiv:2603.25112** — establishes M-ratio and the meta-d′ framework for LLMs; this study explicitly does NOT use M-ratio per the format-dependence finding in Cacioli (2026, quantisation paper).

**Cacioli (2026c, Battery)** — provides the 524-item count rationale and the validity screening framework's empirical foundation.

**Cacioli (2026d), validity scaling paper** — provides the six validity indices used in the screening protocol and the Tier 1 thresholds (L ≥ 0.95, Fp ≥ 0.50, RBS > 0).

**Cacioli (2026e), screen-before-you-interpret** — provides the portable three-tier protocol applied here. §2.3 (degeneracy criterion: <3 distinct values OR >95% in single category), §2.5 (ordered screening sequence with cell-count Step 1), §4.4 (validity is format-dependent), and §2.7 (three-tier classification including the Indeterminate tier with CI-straddling rules) are all directly applied in A2.

**Cacioli (2026f, project E), criterion validation** — establishes that validity tier classification predicts AUROC outcomes, providing the deployment-relevance argument for using the protocol here.

**Cacioli (2026, quantisation paper), arXiv:2604.08976** — establishes that AUROC₂ is format-stable while M-ratio is not, justifying the metric choice here.

**Cacioli (2026, PCN paper)** — proposes the readout-layer mechanism that this study tests behaviourally. The present base–instruct contrast provides a behavioural pattern compatible with — but not diagnostic of — the readout-layer mechanism proposed in the PCN paper.

**Kumaran et al. (2026), arXiv:2603.17839** — establishes verbal confidence as a cached second-order signal in Gemma 3 27B. The present study tests whether *behavioural* verbal confidence in smaller open-weight models on the same substrate remains weakly predictable from length-normalised response logprob.

**Miao and Ungar (2026), arXiv:2603.25052** — "Closing the Confidence-Faithfulness Gap." Mechanistically shows that calibration and verbalised confidence signals are encoded in nearly orthogonal directions in three open-weight models across four datasets, and introduces the "Reasoning Contamination Effect." H5 tests the behavioural signature of the orthogonality finding; E5 tests a behavioural probe of the Reasoning Contamination Effect within M8 NUM, with an item-difficulty control. Neither is diagnostic of the underlying mechanism.

**Zhao et al. (2026), arXiv:2604.01457** — "Wired for Overconfidence." Identifies a compact set of MLP blocks and attention heads in middle-to-late layers that causally write the confidence-inflation signal at the final token position in instruction-tuned LLMs. Provides circuit-level convergent evidence for the readout-layer account tested behaviourally here.

**Kim (2026), Preprints.org doi:10.20944/preprints202604.0078.v2** — "Knowing Before Speaking: In-Computation Metacognition Precedes Verbal Confidence in Large Language Models." Proposes the Knowledge Landscape hypothesis and identifies an "in-computation metacognitive locus" at 61–69% of total network depth in Qwen2.5-7B and Mistral-7B on TriviaQA, using token entropy and hidden-state variance with activation-patching causal evidence (aggregate-level Spearman ρ = −1.00, individual-pair Wilcoxon p = 0.165). Temporally precedes the Kumaran et al. answer-generation caching stage. Together with Kumaran et al. and Zhao et al., Kim brackets a multi-stage pipeline through which an internal metacognitive signal must propagate to reach the verbalised readout; the present study tests behavioural saturation at the *final* (verbalised) stage of that pipeline and does not adjudicate which earlier stage is responsible when saturation is observed. Caveats noted in §1.2: Kim is a non-peer-reviewed preprint at posting time, and the Mistral-7B locus replication and individual-pair causal evidence are at n = 50 with non-significant individual-pair p-values; the paper's own limitations section (Kim 2026 §5.4) flags both. The present pre-registration cites Kim as a mechanistically-adjacent reference for §1.2's claim-bracketing, not as a load-bearing element of any confirmatory hypothesis.

**Wang & Stengel-Eskin (2026), arXiv:2509.25532 ("DiNCo")** — introduces the term "confidence saturation" for the ceiling effect on verbalised confidence in LLMs and shows, on TriviaQA and SimpleQA, that it is correlated with suggestibility bias and is not rescued by self-consistency sample size.

**Seo et al. (2026), arXiv:2510.10913 ("ADVICE")** — identifies "answer-independence" as a primary cause of verbalised overconfidence.

**Yang (2024), arXiv:2412.14737** — found that verbalised confidence in small open-weight models (e.g., gemma1.1-2b) can be "almost independent from accuracy." Provides direct prior evidence for the H1/H2 predictions in the small-model regime.

**Vanhoyweghen et al. (2025), arXiv:2508.15842** — closest behavioural prior work; decomposes CoT features for accuracy prediction in DeepSeek-R1 and Claude 3.7 Sonnet. This study differs in (a) operating on final answer text rather than CoT, (b) decomposing verbal confidence rather than predicting accuracy, (c) applying a validity screen as the diagnostic, (d) using a population of 7 instruction-tuned open-weight models, and (e) testing format manipulation.

**Geng et al. (2024), arXiv:2311.08298** — survey of LLM confidence estimation and calibration; treats confidence as a scalar to be calibrated. The present study is positioned as pre-calibration validity screening.

**Tian et al. (2023), arXiv:2305.14975** — found that RLHF-tuned models can yield better-calibrated verbal confidence than raw token probabilities under some QA conditions. Consistent with the present study's framing: post-training changes the readout in ways that are not necessarily monotonic across models or formats.

**Steyvers & Peters (2025)** — Current Directions review establishing the explicit/implicit elicitation distinction. This study operates within the explicit (verbal) regime.

---

## 9. Authorship and conflicts

Single author (JP Cacioli). No conflicts of interest. No funding.

---

## 10. Pre-registration version history

This pre-registration evolved across multiple drafts before reaching v1.2. v1.2 is the version posted to OSF; earlier drafts (v0.1 through v1.1) are documented here for transparency but are not themselves OSF artifacts.

**v0.1 (drafted prior to first review):** Initial draft developed after the v5 Koriat cue-decomposition study was withdrawn following pilot evidence that Llama-3.1-8B verbal confidence was crushed at ceiling on TriviaQA. The withdrawal of OSF Wps2y/Qzumk and the pivot to a saturation-focused study line is documented in the working journal.

**v0.1 → v0.2 (first round of LLM review):** Integrated reviewer feedback on hypothesis framing, KS test justification, validity protocol citation precision, and the relationship to Kumaran et al. (2026).

**v0.2 → v0.3 (second round of LLM review + citation reversal):** Two papers (Zhao et al. 2026 arXiv:2604.01457 and Miao & Ungar 2026 arXiv:2603.25052) had been refused in v0.2 as potentially hallucinated. Both were verified as real on subsequent search and reinstated in v0.3 as convergent mechanistic evidence. The Reasoning Contamination Effect from Miao & Ungar motivated the addition of `thought_block_token_count` as a per-trial field and the original E5 exploratory analysis. Wang & Stengel-Eskin "DiNCo," Seo et al. "ADVICE," and Yang (2024) added as further convergent evidence. Targeted fixes also addressed reviewer concerns on H1 exclusion bias, no-scaffold scope language, H3 alternative explanations, and H5 framing.

**v0.3 → v0.4 (parser-input clarification):** Smoke testing of the collection script on simulated responses (before any real model output existed) surfaced two structural gaps in v0.3's parser specification. v0.4 clarified which input substring the locked patterns operate on for M1 (continuation prompt with hallucinated continuation pairs) and M8 (reasoning model with `<think>` blocks). The locked regex patterns themselves were unchanged.

**v0.4 → v1.0 (pre-collection verification trials and structural cleanup):**

1. Six pre-collection verification trials were conducted on M8 (full description in §3.7). The trials surfaced three findings that informed v1.0: (a) M8 max_tokens=256 is structurally insufficient; 1024 is appropriate. (b) `llama-cpp-python` 0.3.16's `create_chat_completion` returns `logprobs=None`; the inference wrapper uses raw `__call__` with manually-rendered chat templates from gguf metadata. (c) M8 emits zero categorical class strings across three CAT trials on three different items, motivating the explicit pilot-informed prediction in §2.1.

2. The M8 max_tokens budget is locked at 1024 for both conditions in main collection (§3.4), replacing the v0.4 sanity-run-conditional rule that would have raised it to 512 if needed.

3. §3.4 rewritten to reflect the gguf-template-via-Jinja2 + raw `__call__` inference path, replacing the v0.4 reference to `create_chat_completion`.

4. New §3.7 added: full description of all six M8 verification trials with verbatim outputs committed to OSF as `m8_pilot_verification.json`. This is the "full disclosure" pilot documentation route. The alternative — a brief pointer to the OSF supplementary file without describing the trials in the pre-registration text — was considered and rejected as insufficiently transparent given the substantive design implications of the trials.

5. New §2.1 added: pilot-informed prediction for M8 CAT, explicitly distinguished from confirmatory hypotheses H1-H5, with no decision rule of its own. The prediction commits to reporting actual main-collection M8 CAT behaviour against the explicit pilot-informed expectation regardless of which way main collection diverges from the pilot.

6. E5 reframed (§4.3): restricted to M8 NUM only (because M8 CAT has no parseable confidence to correlate with reasoning length); refined with a partial-correlation control for item difficulty (operationalised as proportion of M2-M7 correct on each item); explicitly framed as "consistent with but not diagnostic of" Miao and Ungar's mechanism rather than a clean behavioural test of the Reasoning Contamination Effect.

7. §1.2 expanded to explicitly tie H1's "% conf ≥ 0.95 > 60%" threshold to screen_before_you_interpret §2.3's "more than 95% in a single category → degenerate" criterion. The saturation study is reframed as a focused empirical demonstration of when the §2.3 degeneracy criterion fires across an open-weight 3-9B sample. This strengthens the relationship to the validity-screening protocol.

8. §3.5.1 (parser-input clarification) folded into §3.5 as the locked behaviour. The v0.4 change-log subsection structure within §3.5 is removed because there is no posted prior version for v1.0 to clarify *from*; the parser-input semantics are simply how the parser works as specified.

9. Confirmatory hypotheses H1-H5 are unchanged. Decision rules in §4.2 are unchanged. The 524-item sample is unchanged. Order seeds are unchanged. The validity protocol applied in A2 is unchanged. The locked regex patterns are unchanged.

10. The verification trial outputs are committed to OSF alongside this pre-registration as `m8_pilot_verification.json` for audit. Readers can verify the pilot was a six-trial infrastructure check, not a theory-shaping pilot dataset. The pre-registration is honest about what the pilot showed and what design decisions it informed; it does not pretend the pilot did not happen.

**v1.0 → v1.1 (post-sanity diagnostic revisions, pre-OSF-posting):**

v1.0 was locked and ready for OSF posting. Before posting, the sanity run (80 trials across 8 models × 2 conditions × 5 items) was executed to verify infrastructure. The sanity run surfaced three issues that required addressing before posting; v1.1 is the revised, post-sanity pre-registration that resolves them. All changes are pre-posting; no v1.0 artifact has been deposited to OSF.

1. **Mistral v0.3 template system-role refusal.** v1.0's inference wrapper passed a `system` message to all instruct models including Mistral. Mistral v0.3's gguf chat template raises `raise_exception('Conversation roles must alternate user/assistant/user/assistant/...')` when the first message is anything other than `user`. All 10 M4 sanity trials failed with `inference_error` and zero generated tokens because Jinja2 raised this exception inside `render_chat_template`. v1.1 extends the existing Gemma 2 system-fold-into-user special case to also cover Mistral v0.3, applying the same principle (no separate system role; fold system text into the user turn). This is a script fix with no hypothesis impact; the pre-reg §3.3 system prompt content is unchanged, only its placement in the rendered template differs.

2. **Degenerate-loop check ordering.** v1.0's `process_trial` ran the `degenerate_loop` check on the raw response before applying M1's first-line truncation substring logic from §3.5. As a result, M1 trials where the model answered correctly on the first line and then ran off into hallucinated `Q: ... A: ...` continuation pairs were wrongly flagged as `degenerate_loop` — the continuation tail (which is genuinely repetitive) fired the detector before the M1 truncation logic could isolate the clean first-line answer. v1.1 runs the degenerate_loop check on `parse_input` (the model-specific substring) rather than on `raw_response`. This is a script fix consistent with the intent of v1.0 §3.5, which already specified that model-specific parse-input logic should govern what the parsers operate on. With this fix, M1 parse extraction should recover the answer on the first line regardless of downstream continuation; whether M1 parse rate then reaches 80% depends on Issue 3 below.

3. **H3 is structurally untestable; replaced with E-base exploratory.** The v1.0 sanity run confirmed what the v1.0 test trial had already hinted at: the Meta-Llama-3-8B base model, under the §3.3 continuation prompt, **does not produce verbalised confidence ratings at all**. There is no "Confidence: X%" or equivalent in any of the 10 M1 sanity responses. The base model answers the question and then hallucinates more Q/A pairs; it does not emit a confidence line, because the continuation format provides no mechanism by which it would. This is not a parser problem and not a `max_tokens` problem. It is a structural consequence of the design: the pre-reg §3.3 explicitly uses a minimal continuation prompt for M1, and the §1.4 scope commitment explicitly excludes scaffolded elicitation. Under this combination, M1 cannot produce verbal confidence. H3 (v1.0), which tested the verbal confidence distribution difference between M1 and M2 via a KS test on binned confidence, is therefore structurally untestable: the M1 confidence distribution is empty.

   Three options were considered: (i) scaffold M1 to elicit confidence (rejected: contradicts §1.4 scope commitment), (ii) keep H3 as-written and let the §6.6 drop rule fire (rejected: amounts to pre-registering a hypothesis whose answer is known to be "cannot be tested"), (iii) drop H3 as a confirmatory hypothesis and replace it with a weaker exploratory analysis using the implicit measurement quantity M1 *does* produce — answer-token logprob — rather than verbal confidence. Option (iii) is adopted.

   v1.1 removes H3 from the confirmatory hypothesis list. The H3 number is retired rather than renumbered. v1.1 adds E-base (§4.3) as an exploratory KS test of M1 vs M2 length-normalised-logprob distributions on NUM. E-base is descriptive only, has no decision rule, and is framed as "consistent with but not diagnostic of" the post-training readout-compression account. The analysis uses token-level logprob data that both models produce natively, regardless of whether they produce verbal confidence.

   M1 is retained in the main collection sample for E-base purposes only. §6.6's M1 retention rule is changed from "≥80% parse success in sanity" to "response_length_tokens > 0 on ≥80% of sanity trials" — the weaker criterion reflects that E-base requires only that M1 generates tokens with logprobs, not that its output parses under the verbal-confidence parsers.

   The cost of this change is real: H3 would have been the only within-architecture base-vs-instruct contrast in the study, and the only pre-registered behavioral test linking directly to the Zhao et al. (2026) circuit-level findings on post-training readout effects. E-base preserves the within-architecture contrast at the implicit-measurement level but not at the verbal-confidence level, which is a genuine scientific weakening. The alternative — pretending H3 was testable when it was not — would have been worse.

   The main scientific claims of the study (saturation prevalence via H1, validity-classification prevalence via H2, format rescue via H4, logprob-confidence concordance via H5) are unchanged. The study's central contribution is unchanged.

4. **`finish_reason` added to recorded fields.** The `finish_reason` returned by `llama-cpp-python` (`stop`, `length`, etc.) is added to the per-trial recorded fields in §3.5 so that M8 budget sufficiency and other truncation events can be audited directly in the output parquet. This is a schema addition only; no analysis depends on it, but it improves post-hoc diagnostic transparency.

5. **Locked parser patterns remain unchanged.** None of the v1.0 regex patterns are modified in v1.1. The fixes above affect (a) chat template rendering (Mistral system fold), (b) parse-ordering (degenerate_loop runs on parse_input), and (c) hypothesis structure (H3 dropped, E-base added). The pattern set itself is untouched.

6. **Confirmatory sample:** v1.1 has **four** confirmatory hypotheses (H1, H2, H4, H5) computed on the **seven** instruct models (M2-M8). M1 is in the main collection for E-base only and is not part of the confirmatory sample. The full sample is still 8 models × 2 conditions × 524 items = 8,384 trials because M1 still runs; only M1's *analytic role* has changed.

v1.1 was prepared for OSF posting. Before posting, the v1.1 text was circulated for external review by three LLM reviewers. v1.2 integrates the subset of their feedback that tightens framing without changing the locked design.

**v1.1 → v1.2 (external reviewer feedback, pre-OSF-posting):**

All v1.1 → v1.2 changes are tightening of framing, guardrails, and transparency language. **No design elements change**: the 524-item sample is unchanged; the 8 models are unchanged; the four confirmatory hypotheses (H1, H2, H4, H5) and their decision rules are unchanged; the locked parser patterns are unchanged; the validity protocol applied in A2 is unchanged; M1's E-base role is unchanged; the M8 pilot-informed prediction in §2.1 is unchanged; the exploratory analyses are unchanged; the item order seeds are unchanged; the sanity run outputs are unchanged. **The collection script is also unchanged**: `collect_saturation.py` as posted with v1.2 is bit-identical to the version used for the v1.1 sanity check, because v1.1 → v1.2 is a markdown-only revision. There is no separate "v1.2 script." If a reader diffs v1.1 and v1.2 markdown, every change is additive framing rather than design revision; if a reader diffs the v1.1-era script and the v1.2-era script, there is no diff.

1. **"Representative sample" softened to "diverse but bounded sample"** in §1.3 and H1 threshold rationale. Closes a predictable reviewer attack ("representative of what population, exactly?") while preserving the claim that the sample spans multiple families.

2. **Mechanism-claim guardrail expanded in §1.2.** The readout-layer compression story is now bracketed by three additional candidate mechanisms drawn from concurrent mechanistic work: in-computation metacognition (Kim 2026), answer-generation caching (Kumaran et al. 2026), and residual-stream orthogonality (Miao & Ungar 2026). The pre-reg explicitly states that the present study cannot distinguish these mechanisms and does not attempt to. All mechanism-adjacent framing uses "consistent with but not diagnostic of" language throughout.

3. **Claim-bounding on minimal elicitation + deterministic decoding added to §1.2.** The pre-reg now explicitly states that the study tests whether saturation is the default behavioural output under minimal elicitation and deterministic decoding, *not* whether saturation is intrinsic to models' internal confidence signals. This forecloses the "you guaranteed saturation by using temp=0 and minimal prompts" critique by framing the regime as the intended scope rather than as an uncontrolled confound.

4. **Q5_K_M reframed as ecologically valid deployment regime** in §7, not as a resource-constrained limitation. Q5_K_M GGUF on consumer hardware is the regime in which small-to-mid open-weight models are actually deployed; characterising interface behaviour in this regime is therefore the study's intended scope, not a compromise. Full-precision regimes are explicitly flagged as outside scope.

5. **Protocol-is-not-phenomenon guardrail added to §1.2.** The pre-reg now explicitly states that saturation as a phenomenon is independently established by Wang and Stengel-Eskin (2026), Yang (2024), and Seo et al. (2026), and that the Cacioli (2026e) protocol is a structured decision procedure for when the phenomenon is too degenerate to support Type-2 use — not a definition of the phenomenon itself. This closes the "circularity" critique (the protocol flags saturation because saturation triggers the protocol).

6. **TriviaQA shared-substrate rationale added to §3.2.** TriviaQA is explicitly justified as the substrate on which Kumaran et al. (2026), Wang & Stengel-Eskin (2026), and Kim (2026) also operate, making cross-study comparison methodologically legitimate.

7. **Parse-failure vs saturation distinction added to §6.6.** The pre-reg now explicitly distinguishes interface-compliance failure from signal-collapse failure and documents that confirmatory hypotheses H1 and H2 are computed on parse-success trials precisely so that they measure saturation as defined, not the union of saturation and non-compliance. The H1 sensitivity analysis (A8) is reframed as a worst-case deployment metric rather than a robustness check.

8. **MAR-violation interpretive pre-commitment added to §6.6.** The pre-reg pre-commits to interpreting MAR violations as rendering the confirmatory results conservative with respect to saturation (on the grounds that parse failure concentration on hard/incorrect items would cause the parse-success subset to under-represent the true saturation rate). This forecloses the post-hoc choice of interpretation direction.

9. **M8 CAT prominence guarantee added to §6.6.** The pre-reg now explicitly guarantees that M8 CAT's exclusion from confirmatory inference (expected per §2.1 pilot prediction) does not reduce its prominence in the results section. High parse failure on M8 CAT is itself a primary empirical finding and is reported alongside H1-H5.

10. **H1 threshold grounding strengthened.** The 60% threshold is now explicitly tied to a mechanical Type-2 AUROC ceiling argument — if ≥60% of trials lie at or above 0.95, the maximum achievable non-parametric Type-2 AUROC on that cell is bounded regardless of how the residual 40% is ordered — framing it as a necessary-but-not-sufficient condition for validity failure rather than an arbitrary cutoff.

11. **E-base interpretive language softened.** "Analogue to the post-training readout-compression hypothesis" replaced with "compatible with post-training readout compression, but also compatible with other differences (e.g., instruction-following priors, template effects, decoder-dynamics differences)." This makes the exploratory nature of E-base more visible.

12. **Kim (2026) added to Relation-to-Prior-Work in §8** as the mechanistic reference for the "in-computation metacognitive locus" stage of the pipeline bracketed in §1.2.

13. **No-tuning pre-commitment added.** No thresholds, hypotheses, decision rules, or analytic procedures were tuned on the basis of sanity-run outcomes, pilot outcomes, or any other interaction with real model output on the 524-item sample. All such pre-collection interactions were used solely to ensure that the pre-specified design was implementable. This statement is added here for the record and applies retroactively to all v0.1 through v1.2 decisions that involved sanity or verification trial data.

v1.2 is the version posted to OSF. v1.1 and all earlier drafts are documented here for transparency but are not themselves OSF artefacts. The sanity run outputs from both the v1.0 script and the v1.1 script (the latter being the version used for the actual sanity check before posting) are committed to OSF alongside v1.2 as `sanity_run_v10_pre_fix.parquet` and `sanity_run_v11.parquet` respectively, so that readers can verify the diagnostic provenance of the v1.0 → v1.1 and v1.1 → v1.2 revisions.

---

## 11. Summary of locked v1.2 parameters

For audit convenience, the locked v1.2 design is summarised here. Every parameter listed below is identical to v1.1 — v1.1 → v1.2 changed framing and guardrails only, not design.

- **Sample:** 8 models × 2 conditions × 524 TriviaQA items = 8,384 trials. M1 is in the collection but its analytic role is restricted to E-base (exploratory); confirmatory analyses are computed on 7 instruct models × 2 conditions × 524 items = 7,336 trials.
- **Item draw:** `numpy.random.default_rng(seed=42).choice(17944, 524, replace=False)` from TriviaQA `rc.nocontext` validation split
- **Order seed per cell:** `42 + model_idx*100 + cond_idx`
- **Inference:** temperature=0.0, top_p=0.95, repeat_penalty=1.0, seed=42, n_ctx=4096, n_gpu_layers=-1, logits_all=True, logprobs=5
- **max_tokens:** 256 for M1-M7; 1024 for M8 (locked)
- **Chat template:** Read from gguf metadata (`tokenizer.chat_template`), rendered via Jinja2, passed to raw `__call__`. Mistral v0.3 (M4) and Gemma 2 (M7) use the system-fold-into-user special case because their templates refuse a separate system role.
- **Confirmatory hypotheses (4):** H1 (saturation prevalence > 60% on 7 instruct, NUM), H2 (≥4 of 7 Invalid on NUM), H4 (≥2 NUM-Invalid models reclassified non-Invalid under CAT), H5 (mean R²_CV < 0.20 both conditions). H3 was a confirmatory hypothesis in v1.0 and is retired (see §2 and §10 v1.0 → v1.1 entry).
- **Pilot-informed prediction (not confirmatory):** M8 CAT will show substantially reduced parse rate vs M8 NUM (§2.1)
- **Exploratory analyses:** E1-E10 plus E-base (M1 vs M2 length-normalised-logprob KS test, NUM only; descriptive only) and E5 (within-M8 partial Spearman on `thought_block_token_count` and `parsed_confidence`, NUM only, controlling for item difficulty via M2-M7 proportion correct)
- **Validity protocol applied in A2:** Cacioli (2026e) §2.3 confidence harmonisation, §2.5 ordered screening sequence, §2.7 three-tier classification, §2.8 VRS table reporting
- **M1 retention rule (§6.6):** M1 is retained in main collection if `response_length_tokens > 0` on ≥80% of M1 sanity trials; parse-rate criterion from v1.0 does not apply to M1.
- **Parse failure vs saturation distinction (v1.2 §6.6):** Parse failure is treated as interface non-compliance, distinct from signal-collapse saturation. Confirmatory hypotheses operate on parse-success trials. The H1 sensitivity analysis (A8) is reframed in v1.2 as a worst-case deployment metric, not a robustness check.
- **MAR violation interpretive pre-commitment (v1.2 §6.6):** If E8 reveals strong MAR violation, confirmatory results will be interpreted as conservative with respect to saturation.
- **M8 CAT prominence guarantee (v1.2 §6.6):** M8 CAT exclusion from confirmatory inference does not reduce its prominence in the main results.
- **No-tuning pre-commitment (v1.2 §10):** No thresholds, hypotheses, decision rules, or analytic procedures were tuned on the basis of any sanity-run, pilot, or pre-collection model interaction.

---

*End of pre-registration v1.2.*
