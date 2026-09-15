import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Load model
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
).to(DEVICE)

model.eval()


# Load WMDP-Bio
print("Loading WMDP-Bio...")
wmdp_dataset = load_dataset(
    "cais/wmdp",
    "wmdp-bio",
    split="test"
)


# Load MMLU
print("Loading MMLU...")
retain_dataset = load_dataset(
    "cais/mmlu",
    "all",
    split="test"
)


# Build prompt
def build_prompt(question, choices):
    user_content = f"""Answer the following multiple-choice question.
Choose only one answer: A, B, C, or D.

Question:
{question}

A. {choices[0]}
B. {choices[1]}
C. {choices[2]}
D. {choices[3]}

Answer:"""

    messages = [
        {
            "role": "user",
            "content": user_content
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    return prompt


# Calculate layer-wise L2 scores
def get_layer_scores(question, choices):

    prompt = build_prompt(
        question,
        choices
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False
    ).to(DEVICE)

    # Run model and return hidden states
    with torch.inference_mode():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            use_cache=False
        )

    hidden_states = outputs.hidden_states

    layer_scores = []

    # Skip hidden_states[0] because it is the embedding output
    for layer_index in range(1, len(hidden_states)):

        hidden_state = hidden_states[layer_index]

        # Calculate L2 norm for each token
        token_norms = torch.norm(
            hidden_state[0],
            p=2,
            dim=1
        )

        # Average L2 norm across all tokens
        layer_score = token_norms.mean().item()

        layer_scores.append(layer_score)

    return layer_scores


# Get first WMDP example
wmdp_example = wmdp_dataset[0]

wmdp_scores = get_layer_scores(
    wmdp_example["question"],
    wmdp_example["choices"]
)


# Get one Retain example
retain_example = retain_dataset[0]

retain_scores = get_layer_scores(
    retain_example["question"],
    retain_example["choices"]
)


# Compare WMDP and Retain layer scores
print("\nLayer-wise comparison:")
print("---------------------------------------------")
print("Layer | WMDP       | Retain     | Difference")
print("---------------------------------------------")

for layer_index in range(len(wmdp_scores)):

    difference = (
        wmdp_scores[layer_index]
        - retain_scores[layer_index]
    )

    print(
        f"{layer_index:5d} | "
        f"{wmdp_scores[layer_index]:10.4f} | "
        f"{retain_scores[layer_index]:10.4f} | "
        f"{difference:10.4f}"
    )
    