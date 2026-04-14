"""
TriviaQA verification — Step 4
Runs 5 items through Llama-3.1-8B with unconstrained prompting.
Purpose: check if response length and hedging vary across items.
"""
from datasets import load_dataset
from llama_cpp import Llama
import os

os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
os.environ["MIOPEN_DISABLE_CACHE"] = "1"

# Load 5 items
ds = load_dataset('mandarjoshi/trivia_qa', 'rc.nocontext', split='validation')
items = [ds[i] for i in [0, 100, 500, 1000, 2000]]

# Load model — adjust path if your Llama gguf is elsewhere
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

# Unconstrained prompt — no length limit, no one-sentence rule
SYSTEM = "Answer the following question accurately. Provide a confidence rating from 0 to 100 for your answer."
USER_TEMPLATE = """{question}

Respond in exactly this format:
ANSWER: [your answer]
CONFIDENCE: [integer 0-100]"""

for i, item in enumerate(items):
    q = item['question']
    gold = item['answer']['value']
    aliases = item['answer']['aliases']
    
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
    print(f"Response length (chars): {len(response)}")
    print(f"Response:\n{response}")
    print()

print("=" * 70)
print("DONE")