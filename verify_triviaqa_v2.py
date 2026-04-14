"""
TriviaQA verification v2 — unconstrained prompting.
Does removing the format scaffold produce response-style variance?
"""
from datasets import load_dataset
from llama_cpp import Llama
import os

os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
os.environ["MIOPEN_DISABLE_CACHE"] = "1"

ds = load_dataset('mandarjoshi/trivia_qa', 'rc.nocontext', split='validation')
# Mix easy and harder items — pull from a wider range
items = [ds[i] for i in [0, 100, 500, 1000, 2000, 5000, 8000, 12000, 15000, 17000]]

MODEL_PATH = r"C:\sdt_calibration\models\Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf"

print(f"Loading {MODEL_PATH}...")
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=4096,
    n_gpu_layers=-1,
    verbose=False,
    seed=42,
)
print("Model loaded.\n")

# Unconstrained: no format, no length rule, just ask
SYSTEM = "You are answering trivia questions. After your answer, state your confidence as a percentage from 0 to 100."
USER_TEMPLATE = "{question}"

for i, item in enumerate(items):
    q = item['question']
    gold = item['answer']['value']
    
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER_TEMPLATE.format(question=q)},
    ]
    
    out = llm.create_chat_completion(
        messages=messages,
        temperature=0.0,
        max_tokens=256,
        seed=42,
    )
    response = out['choices'][0]['message']['content']
    
    print("=" * 70)
    print(f"Item {i+1}: {q}")
    print(f"Gold: {gold}")
    print(f"Length (chars): {len(response)}")
    print(f"Response:")
    print(response)
    print()

print("=" * 70)
print("DONE")