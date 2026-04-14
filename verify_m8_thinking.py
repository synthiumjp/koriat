# verify_m8_thinking.py
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

# Get the chat template from gguf metadata
template_str = llm.metadata["tokenizer.chat_template"]

# Render it with a NUM-condition system + user message
messages = [
    {"role": "system", "content": "You are answering trivia questions. After your answer, state your confidence as a percentage from 0 to 100."},
    {"role": "user", "content": "Who was the man behind The Chipmunks?"},
]

# Get bos_token from metadata if available
bos_token = llm.metadata.get("tokenizer.ggml.bos_token", "")
# Get bos_token id and look it up
try:
    bos_id = int(llm.metadata.get("tokenizer.ggml.bos_token_id", 0))
    bos_token = llm.detokenize([bos_id]).decode("utf-8", errors="replace")
except Exception:
    bos_token = ""

template = Template(template_str)
try:
    rendered = template.render(
        messages=messages,
        bos_token=bos_token,
        add_generation_prompt=True,
    )
except Exception as e:
    print(f"Template render error: {e}")
    rendered = None

if rendered is not None:
    print("=" * 70)
    print("RENDERED PROMPT (last 300 chars):")
    print("=" * 70)
    print(repr(rendered[-300:]))
    print()
    print(f"Total length: {len(rendered)} chars")
    print(f"Ends with '<think>'? {rendered.rstrip().endswith('<think>')}")
    print()

    # Now actually generate from this prompt
    print("=" * 70)
    print("GENERATION (max 256 tokens):")
    print("=" * 70)
    result = llm(
        prompt=rendered,
        max_tokens=256,
        temperature=0.0,
        top_p=0.95,
        seed=42,
        logprobs=5,
    )
    text = result["choices"][0]["text"]
    print(repr(text))
    print()
    print(f"Response length: {len(text)} chars")
    print(f"Contains '<think>'?  {'<think>' in text}")
    print(f"Contains '</think>'? {'</think>' in text}")
    print(f"Logprobs populated?  {result['choices'][0]['logprobs'] is not None}")
    if result['choices'][0]['logprobs'] is not None:
        print(f"  n tokens: {len(result['choices'][0]['logprobs']['tokens'])}")
        print(f"  n logprobs: {len(result['choices'][0]['logprobs']['token_logprobs'])}")