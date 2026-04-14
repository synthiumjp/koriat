"""
TriviaQA verification v3 — 20 items, check confidence variance.
"""
from datasets import load_dataset
from llama_cpp import Llama
import os
import re

os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
os.environ["MIOPEN_DISABLE_CACHE"] = "1"

ds = load_dataset('mandarjoshi/trivia_qa', 'rc.nocontext', split='validation')
# Spread indices across the validation split
idxs = [0, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000,
        9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 17900]
items = [ds[i] for i in idxs]

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

SYSTEM = "You are answering trivia questions. After your answer, state your confidence as a percentage from 0 to 100."

results = []

for i, item in enumerate(items):
    q = item['question']
    gold = item['answer']['value']
    
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": q},
    ]
    
    out = llm.create_chat_completion(
        messages=messages,
        temperature=0.0,
        max_tokens=256,
        seed=42,
    )
    response = out['choices'][0]['message']['content']
    
    # Extract first 0-100 number appearing after "confidence" (loose)
    conf_match = re.search(r'[Cc]onfidence[^\d]*(\d{1,3})', response)
    conf = int(conf_match.group(1)) if conf_match else None
    if conf is not None and conf > 100:
        conf = None
    
    results.append({
        'idx': i+1,
        'q': q,
        'gold': gold,
        'length': len(response),
        'conf': conf,
        'response': response,
    })
    
    print(f"[{i+1:2d}] len={len(response):4d} conf={conf} | {q[:60]}")

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)

lengths = [r['length'] for r in results]
confs = [r['conf'] for r in results if r['conf'] is not None]

print(f"Length: min={min(lengths)}, max={max(lengths)}, mean={sum(lengths)/len(lengths):.0f}")
print(f"Confidence parsed: {len(confs)}/20")
if confs:
    print(f"Confidence: min={min(confs)}, max={max(confs)}, mean={sum(confs)/len(confs):.1f}")
    print(f"  <100: {sum(1 for c in confs if c < 100)}/{len(confs)}")
    print(f"  <95:  {sum(1 for c in confs if c < 95)}/{len(confs)}")
    print(f"  <90:  {sum(1 for c in confs if c < 90)}/{len(confs)}")
    print(f"  <80:  {sum(1 for c in confs if c < 80)}/{len(confs)}")

print()
print("=" * 70)
print("FULL RESPONSES")
print("=" * 70)
for r in results:
    print(f"\n--- Item {r['idx']} | len={r['length']} | conf={r['conf']} ---")
    print(f"Q: {r['q']}")
    print(f"Gold: {r['gold']}")
    print(f"Response: {r['response'][:400]}{'...' if len(r['response']) > 400 else ''}")