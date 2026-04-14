# verify_m1_logprobs.py
from llama_cpp import Llama

llm = Llama(
    model_path=r"C:\sdt_calibration\models\Meta-Llama-3-8B.Q5_K_M.gguf",
    n_ctx=4096,
    n_gpu_layers=-1,
    logits_all=True,
    verbose=False,
    seed=42,
)

result = llm(
    prompt="Q: Who was the man behind The Chipmunks?\nA:",
    max_tokens=64,
    temperature=0.0,
    top_p=0.95,
    seed=42,
    logprobs=5,
)
choice = result["choices"][0]
print(f"text: {choice['text']!r}")
print(f"logprobs keys: {list(choice['logprobs'].keys())}")
print(f"n tokens: {len(choice['logprobs']['tokens'])}")
print(f"n logprobs: {len(choice['logprobs']['token_logprobs'])}")
print(f"first 5 tokens: {choice['logprobs']['tokens'][:5]}")
print(f"first 5 logprobs: {choice['logprobs']['token_logprobs'][:5]}")