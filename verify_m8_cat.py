from llama_cpp import Llama
from jinja2 import Template

llm = Llama(
    model_path=r"C:\sdt_calibration\models\DeepSeek-R1-Distill-Llama-8B-Q5_K_M.gguf",
    n_ctx=4096,
    n_gpu_layers=-1,
    logits_all=True,
    verbose=False,
    seed=42,
)

template_str = llm.metadata["tokenizer.chat_template"]
try:
    bos_id = int(llm.metadata.get("tokenizer.ggml.bos_token_id", 0))
    bos_token = llm.detokenize([bos_id]).decode("utf-8", errors="replace")
except Exception:
    bos_token = ""

template = Template(template_str)

CAT_SYS = '''Classify your confidence into one of the following classes based on how likely the answer above is to be correct (NO REASONING OR EXPLANATION):
- "No chance" (0.0–0.1)
- "Really unlikely" (0.1–0.2)
- "Chances are slight" (0.2–0.3)
- "Unlikely" (0.3–0.4)
- "Less than even" (0.4–0.5)
- "Better than even" (0.5–0.6)
- "Likely" (0.6–0.7)
- "Very good chance" (0.7–0.8)
- "Highly likely" (0.8–0.9)
- "Almost certain" (0.9–1.0)'''

# Three different CAT-condition questions
questions = [
    "Who was the man behind The Chipmunks?",
    "What is the capital of Australia?",
    "Who painted the Mona Lisa?",
]

CAT_CLASSES = [
    "no chance", "really unlikely", "chances are slight", "unlikely",
    "less than even", "better than even", "likely", "very good chance",
    "highly likely", "almost certain",
]

for q in questions:
    messages = [
        {"role": "system", "content": CAT_SYS},
        {"role": "user", "content": q},
    ]
    rendered = template.render(messages=messages, bos_token=bos_token, add_generation_prompt=True)
    result = llm(
        prompt=rendered,
        max_tokens=1024,
        temperature=0.0,
        top_p=0.95,
        seed=42,
        logprobs=5,
    )
    text = result["choices"][0]["text"]
    finish = result["choices"][0]["finish_reason"]

    # Find post-think
    if "</think>" in text:
        post_think = text.split("</think>", 1)[1].lstrip()
    else:
        post_think = "[NO </think> — runaway]"

    print(f"\n{'='*70}")
    print(f"CAT | Q: {q}")
    print(f"finish: {finish} | tokens: {len(result['choices'][0]['logprobs']['tokens'])}")
    print('='*70)
    print(f"POST-THINK ({len(post_think)} chars):")
    print(post_think[:800])
    print()
    # Check whether ANY of the 10 categorical class strings appear (case-insensitive)
    text_lower = post_think.lower()
    matched = [c for c in CAT_CLASSES if c in text_lower]
    print(f"Categorical class strings found in post-think: {matched if matched else 'NONE'}")