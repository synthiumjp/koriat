# verify_m8_format_consistency.py
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

# Three different questions to see if format is consistent
questions = [
    ("NUM", "What is the capital of Australia?"),
    ("NUM", "Who painted the Mona Lisa?"),
    ("CAT", "What year did World War II end?"),
]

NUM_SYS = "You are answering trivia questions. After your answer, state your confidence as a percentage from 0 to 100."
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

for cond, q in questions:
    sys_msg = NUM_SYS if cond == "NUM" else CAT_SYS
    messages = [
        {"role": "system", "content": sys_msg},
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

    print(f"\n{'='*70}")
    print(f"COND: {cond} | Q: {q}")
    print(f"finish: {finish} | tokens: {len(result['choices'][0]['logprobs']['tokens'])}")
    print('='*70)
    # Find post-think
    if "</think>" in text:
        post_think = text.split("</think>", 1)[1].lstrip()
        print(f"POST-THINK ({len(post_think)} chars):")
        print(post_think[:500])
    else:
        print(f"NO </think> — runaway. Last 300 chars:")
        print(text[-300:])