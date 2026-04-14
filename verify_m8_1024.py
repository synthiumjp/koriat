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

messages = [
    {"role": "system", "content": "You are answering trivia questions. After your answer, state your confidence as a percentage from 0 to 100."},
    {"role": "user", "content": "Who was the man behind The Chipmunks?"},
]

template = Template(template_str)
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

print(f"finish_reason: {finish}")
print(f"response length chars: {len(text)}")
print(f"contains <think>?  {'<think>' in text}")
print(f"contains </think>? {'</think>' in text}")
print(f"n tokens: {len(result['choices'][0]['logprobs']['tokens'])}")
print()
print("LAST 500 CHARS OF RESPONSE:")
print("=" * 70)
print(text[-500:])