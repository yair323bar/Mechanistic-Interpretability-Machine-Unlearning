import torch
import random
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

SAMPLES_PER_SUBJECT = 100
RANDOM_SEED = 42
EPSILON = 1e-8

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Selected retain subjects
RETAIN_SUBJECTS = [
    "elementary_mathematics",
    "formal_logic",
    "high_school_geography",
    "high_school_world_history",
    "high_school_macroeconomics",
    "high_school_computer_science"
]


# Load model
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
).to(DEVICE)

model.eval()


# Load datasets
print("Loading WMDP-Bio...")
wmdp_dataset = load_dataset(
    "cais/wmdp",
    "wmdp-bio",
    split="test"
)

print("Loading MMLU...")
mmlu_dataset = load_dataset(
    "cais/mmlu",
    "all",
    split="test"
)


# Build retain dataset
random.seed(RANDOM_SEED)

retain_dataset = []

for subject in RETAIN_SUBJECTS:

    subject_examples = [
        example for example in mmlu_dataset
        if example["subject"] == subject
    ]

    selected_examples = random.sample(
        subject_examples,
        SAMPLES_PER_SUBJECT
    )

    retain_dataset.extend(selected_examples)


print("\nDatasets ready!")
print(f"WMDP examples: {len(wmdp_dataset)}")
print(f"Retain examples: {len(retain_dataset)}")


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


# Calculate residual update L2 scores for one example
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

    # Calculate how much each layer changes the representation
    for layer_index in range(len(hidden_states) - 1):

        hidden_state_before = hidden_states[layer_index]
        hidden_state_after = hidden_states[layer_index + 1]

        # Calculate the residual update added by the layer
        residual_update = (
            hidden_state_after
            - hidden_state_before
        )

        # Calculate L2 norm for each token
        token_norms = torch.norm(
            residual_update[0],
            p=2,
            dim=1
        )

        # Average L2 norm across all tokens
        layer_score = token_norms.mean().item()

        layer_scores.append(layer_score)

    return layer_scores


# Calculate average layer scores for a dataset
def calculate_dataset_scores(dataset, dataset_name):

    total_scores = None

    num_examples = len(dataset)

    print(
        f"\nCalculating localization scores for "
        f"{dataset_name}..."
    )

    for i in range(num_examples):

        example = dataset[i]

        scores = get_layer_scores(
            example["question"],
            example["choices"]
        )

        if total_scores is None:
            total_scores = [
                0.0 for _ in range(len(scores))
            ]

        for layer_index, score in enumerate(scores):
            total_scores[layer_index] += score

        print(
            f"{dataset_name} "
            f"{i + 1}/{num_examples}"
        )

    average_scores = [
        score / num_examples
        for score in total_scores
    ]

    return average_scores


# WMDP localization
wmdp_scores = calculate_dataset_scores(
    wmdp_dataset,
    "WMDP"
)


# Retain localization
retain_scores = calculate_dataset_scores(
    retain_dataset,
    "Retain"
)


# Calculate selective localization score
selective_scores = []

for layer_index in range(len(wmdp_scores)):

    wmdp_score = wmdp_scores[layer_index]
    retain_score = retain_scores[layer_index]

    selective_score = (
        (wmdp_score - retain_score)
        /
        (wmdp_score + retain_score + EPSILON)
    )

    selective_scores.append(selective_score)


# Build results table
results = []

for layer_index in range(len(wmdp_scores)):

    results.append(
        {
            "layer": layer_index,
            "wmdp_score": wmdp_scores[layer_index],
            "retain_score": retain_scores[layer_index],
            "selective_score": selective_scores[layer_index]
        }
    )

results_df = pd.DataFrame(results)


# Calculate layer rankings
results_df["wmdp_rank"] = (
    results_df["wmdp_score"]
    .rank(
        ascending=False,
        method="min"
    )
    .astype(int)
)

results_df["selective_rank"] = (
    results_df["selective_score"]
    .rank(
        ascending=False,
        method="min"
    )
    .astype(int)
)


# Show results
print("\nLayer-wise Localization Results:")
print(
    "--------------------------------------------------------------------------"
)
print(
    "Layer | WMDP Score | Retain Score | Selective Score | WMDP Rank | Selective Rank"
)
print(
    "--------------------------------------------------------------------------"
)

for _, row in results_df.iterrows():

    print(
        f"{int(row['layer']):5d} | "
        f"{row['wmdp_score']:10.4f} | "
        f"{row['retain_score']:12.4f} | "
        f"{row['selective_score']:15.6f} | "
        f"{int(row['wmdp_rank']):9d} | "
        f"{int(row['selective_rank']):14d}"
    )


# Show WMDP ranking
print("\nLayers ranked by WMDP score:")

wmdp_ranking = results_df.sort_values(
    "wmdp_score",
    ascending=False
)

for _, row in wmdp_ranking.iterrows():

    print(
        f"Layer {int(row['layer'])}: "
        f"{row['wmdp_score']:.4f}"
    )


# Show selective ranking
print("\nLayers ranked by selective score:")

selective_ranking = results_df.sort_values(
    "selective_score",
    ascending=False
)

for _, row in selective_ranking.iterrows():

    print(
        f"Layer {int(row['layer'])}: "
        f"{row['selective_score']:.6f}"
    )


# Show bottom WMDP ranking
print("\nLayers ranked from lowest WMDP score:")

bottom_ranking = results_df.sort_values(
    "wmdp_score",
    ascending=True
)

for _, row in bottom_ranking.iterrows():

    print(
        f"Layer {int(row['layer'])}: "
        f"{row['wmdp_score']:.4f}"
    )


# Save results to CSV
results_df.to_csv(
    "results/localization_scores.csv",
    index=False
)

print("\nResults saved to:")
print("results/localization_scores.csv")
