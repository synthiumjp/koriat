# verify_gguf_templates.py
from pathlib import Path
from llama_cpp import Llama

MODELS_DIR = Path(r"C:\sdt_calibration\models")

MODELS = [
    ("M1", "Meta-Llama-3-8B.Q5_K_M.gguf"),
    ("M2", "Meta-Llama-3-8B-Instruct-Q5_K_M.gguf"),
    ("M3", "Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf"),
    ("M4", "Mistral-7B-Instruct-v0.3-Q5_K_M.gguf"),
    ("M5", "Qwen2.5-3B-Instruct-Q5_K_M.gguf"),
    ("M6", "Qwen2.5-7B-Instruct-Q5_K_M.gguf"),
    ("M7", "gemma-2-9b-it-Q5_K_M.gguf"),
    ("M8", "DeepSeek-R1-Distill-Llama-8B-Q5_K_M.gguf"),
]

for model_id, filename in MODELS:
    path = MODELS_DIR / filename
    print(f"\n{'='*70}")
    print(f"{model_id}: {filename}")
    print('='*70)
    if not path.exists():
        print("  FILE NOT FOUND")
        continue
    try:
        # Load with minimal context to be fast
        llm = Llama(
            model_path=str(path),
            n_ctx=512,
            n_gpu_layers=0,    # CPU-only for speed (we're not generating)
            verbose=False,
            seed=42,
        )
        meta = llm.metadata if hasattr(llm, 'metadata') else {}
        template = meta.get("tokenizer.chat_template", None)
        if template is None:
            print("  NO chat_template in metadata")
            print(f"  metadata keys: {list(meta.keys())[:10]}...")
        else:
            print(f"  chat_template length: {len(template)} chars")
            print(f"  first 200 chars: {template[:200]!r}")
            # Check for <think> handling specifically
            if model_id == "M8":
                print(f"  contains '<think>'?  {'<think>' in template}")
                print(f"  contains 'thinking'? {'thinking' in template.lower()}")
        del llm
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")