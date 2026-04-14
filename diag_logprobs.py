# diag_logprobs.py — run this from C:\sdt_calibration\koriat_project_b
import json
from llama_cpp import Llama

llm = Llama(
    model_path=r"C:\sdt_calibration\models\Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf",
    n_ctx=4096,
    n_gpu_layers=-1,
    logits_all=True,
    verbose=False,
    seed=42,
)

messages = [
    {"role": "system", "content": "You are answering trivia questions. After your answer, state your confidence as a percentage from 0 to 100."},
    {"role": "user", "content": "Who was the man behind The Chipmunks?"},
]

# Try both shapes — boolean and integer
for logprobs_arg in [True, 5]:
    print(f"\n{'='*60}")
    print(f"Trying logprobs={logprobs_arg!r}")
    print('='*60)
    try:
        result = llm.create_chat_completion(
            messages=messages,
            max_tokens=64,
            temperature=0.0,
            top_p=0.95,
            seed=42,
            logprobs=logprobs_arg,
        )
        choice = result["choices"][0]
        print(f"choice keys: {list(choice.keys())}")
        lp = choice.get("logprobs")
        print(f"logprobs type: {type(lp).__name__}")
        if lp is None:
            print("logprobs is None")
        elif isinstance(lp, dict):
            print(f"logprobs dict keys: {list(lp.keys())}")
            for k, v in lp.items():
                if isinstance(v, list):
                    print(f"  {k}: list of len {len(v)}, first item: {v[0] if v else 'empty'}")
                else:
                    print(f"  {k}: {type(v).__name__}")
        else:
            print(f"logprobs value: {lp}")
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")

# Also try the raw __call__ interface for comparison
print(f"\n{'='*60}")
print("Trying raw __call__ with logprobs=5")
print('='*60)
try:
    result = llm(
        prompt="Q: Who was the man behind The Chipmunks?\nA:",
        max_tokens=64,
        temperature=0.0,
        top_p=0.95,
        seed=42,
        logprobs=5,
    )
    choice = result["choices"][0]
    print(f"choice keys: {list(choice.keys())}")
    lp = choice.get("logprobs")
    if lp is None:
        print("logprobs is None")
    elif isinstance(lp, dict):
        print(f"logprobs dict keys: {list(lp.keys())}")
        for k, v in lp.items():
            if isinstance(v, list):
                print(f"  {k}: list of len {len(v)}, first item: {v[0] if v else 'empty'}")
            else:
                print(f"  {k}: {type(v).__name__}")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")