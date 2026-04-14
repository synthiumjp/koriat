# verify_hf_tokenizer.py
from transformers import AutoTokenizer

# Try Llama 3.1 8B Instruct first — it's the most common and most likely to load
tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")

messages = [
    {"role": "system", "content": "You are answering trivia questions. After your answer, state your confidence as a percentage from 0 to 100."},
    {"role": "user", "content": "Who was the man behind The Chipmunks?"},
]

prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print("Rendered chat template:")
print("---")
print(prompt)
print("---")
print(f"Length: {len(prompt)} chars")