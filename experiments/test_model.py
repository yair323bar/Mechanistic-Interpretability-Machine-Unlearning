import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

print("CUDA available:", torch.cuda.is_available())
print("Loading model:", MODEL_NAME)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
).to("cuda")

print("Model loaded successfully!")
print("Model device:", next(model.parameters()).device)

prompt = "What is the capital of France?"

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=False
    )

answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\nModel output:")
print(answer)

print(
    "\nGPU memory allocated:",
    round(torch.cuda.memory_allocated() / 1024**3, 2),
    "GB"
)

print("\nCandidate tokenization:")

for letter in ["A", "B", "C", "D"]:
    token_ids = tokenizer(
        letter,
        add_special_tokens=False
    ).input_ids

    print(
        f"{letter}: "
        f"token_ids={token_ids}, "
        f"num_tokens={len(token_ids)}"
    )