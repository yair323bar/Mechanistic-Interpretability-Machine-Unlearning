import random
import torch
import pandas as pd
import matplotlib.pyplot as plt

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

RANDOM_SEED = 42
SAMPLES_PER_SUBJECT = 100

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LETTERS = ["A", "B", "C", "D"]

BASELINE_FILE = "results/retain_baseline_predictions.csv"

RESULTS_FILE = "results/causal_validation_results.csv"

SELECTED_LAYERS = [
    11,
    13
]

ALPHA = 0.25


# Selected retain subjects
RETAIN_SUBJECTS = [
    "elementary_mathematics",
    "formal_logic",
    "high_school_geography",
    "high_school_world_history",
    "high_school_macroeconomics",
    "high_school_computer_science"
]


# Load baseline results
print("Loading Retain baseline...")

baseline_df = pd.read_csv(
    BASELINE_FILE
)


# Calculate baseline for each subject
baseline_subject_results = {}

for subject in RETAIN_SUBJECTS:

    subject_data = baseline_df[
        baseline_df["subject"] == subject
    ]

    correct = (
        subject_data["correct"]
        == subject_data["prediction"]
    ).sum()

    accuracy = (
        correct / len(subject_data)
    )

    baseline_subject_results[
        subject
    ] = accuracy


# Show baseline results
print("\nBaseline accuracy by subject:")

for subject, accuracy in baseline_subject_results.items():

    print(
        f"{subject}: "
        f"{accuracy:.2%}"
    )


# Load model
print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

print("Loading model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
).to(DEVICE)

model.eval()


# Load MMLU
print("Loading MMLU...")

mmlu_dataset = load_dataset(
    "cais/mmlu",
    "all",
    split="test"
)


# Build same retain dataset
random.seed(
    RANDOM_SEED
)

retain_dataset = []

for subject in RETAIN_SUBJECTS:

    subject_examples = [
        example
        for example in mmlu_dataset
        if example["subject"] == subject
    ]

    selected_examples = random.sample(
        subject_examples,
        SAMPLES_PER_SUBJECT
    )

    retain_dataset.extend(
        selected_examples
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


# Score all candidate answers
def score_candidates(prompt):

    prompt_ids = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False
    ).input_ids.to(DEVICE)

    with torch.inference_mode():

        outputs = model(
            input_ids=prompt_ids,
            use_cache=False
        )

    logits = outputs.logits

    # Last position predicts the next token
    next_token_logits = logits[
        0,
        -1
    ]

    log_probs = torch.log_softmax(
        next_token_logits,
        dim=-1
    )

    scores = []

    for letter in LETTERS:

        token_id = tokenizer(
            letter,
            add_special_tokens=False
        ).input_ids[0]

        scores.append(
            log_probs[token_id].item()
        )

    return scores


# Create intervention
def create_intervention_hook(alpha):

    def intervention_hook(
        module,
        inputs,
        output
    ):

        hidden_state_before = inputs[0]

        if isinstance(output, tuple):

            hidden_state_after = output[0]

            # Calculate residual update
            residual_update = (
                hidden_state_after
                - hidden_state_before
            )

            # Scale residual update
            modified_hidden_state = (
                hidden_state_before
                + alpha * residual_update
            )

            return (
                modified_hidden_state,
                *output[1:]
            )

        hidden_state_after = output

        # Calculate residual update
        residual_update = (
            hidden_state_after
            - hidden_state_before
        )

        # Scale residual update
        modified_hidden_state = (
            hidden_state_before
            + alpha * residual_update
        )

        return modified_hidden_state

    return intervention_hook


# Register intervention
def register_intervention():

    hooks = []

    for layer_index in SELECTED_LAYERS:

        hook = (
            model.model.layers[layer_index]
            .register_forward_hook(
                create_intervention_hook(
                    ALPHA
                )
            )
        )

        hooks.append(
            hook
        )

    return hooks


# Remove intervention
def remove_hooks(hooks):

    for hook in hooks:
        hook.remove()


# Run causal validation
print("\nStarting causal validation...")

print(
    f"Layers: {SELECTED_LAYERS}"
)

print(
    f"Alpha: {ALPHA}"
)

hooks = register_intervention()

intervention_results = {
    subject: {
        "correct": 0,
        "total": 0
    }
    for subject in RETAIN_SUBJECTS
}


try:

    for i, example in enumerate(
        retain_dataset
    ):

        subject = example[
            "subject"
        ]

        prompt = build_prompt(
            example["question"],
            example["choices"]
        )

        scores = score_candidates(
            prompt
        )

        predicted_answer = max(
            range(len(scores)),
            key=lambda x: scores[x]
        )

        correct_answer = example[
            "answer"
        ]

        intervention_results[
            subject
        ]["total"] += 1

        if (
            predicted_answer
            == correct_answer
        ):

            intervention_results[
                subject
            ]["correct"] += 1

        print(
            f"Question "
            f"{i + 1}/{len(retain_dataset)} | "
            f"{subject}"
        )

finally:

    # Remove hooks
    remove_hooks(
        hooks
    )


# Compare baseline and intervention
results = []

for subject in RETAIN_SUBJECTS:

    baseline_accuracy = (
        baseline_subject_results[
            subject
        ]
    )

    intervention_accuracy = (
        intervention_results[
            subject
        ]["correct"]
        / intervention_results[
            subject
        ]["total"]
    )

    delta = (
        baseline_accuracy
        - intervention_accuracy
    )

    results.append(
        {
            "subject": subject,
            "baseline_accuracy": baseline_accuracy,
            "intervention_accuracy": intervention_accuracy,
            "delta": delta
        }
    )


results_df = pd.DataFrame(
    results
)


# Save results
results_df.to_csv(
    RESULTS_FILE,
    index=False
)


# Show results
print("\n--------------------------------")
print("Causal Validation Results")
print("--------------------------------")

for _, row in results_df.iterrows():

    print(
        f"{row['subject']}: "
        f"Baseline={row['baseline_accuracy']:.2%} | "
        f"Intervention={row['intervention_accuracy']:.2%} | "
        f"Delta={row['delta']:.2%}"
    )


# Create graph
plot_df = results_df.copy()

plot_df[
    "baseline_accuracy"
] *= 100

plot_df[
    "intervention_accuracy"
] *= 100


ax = plot_df.plot(
    x="subject",
    y=[
        "baseline_accuracy",
        "intervention_accuracy"
    ],
    kind="bar",
    figsize=(11, 6)
)

ax.set_xlabel(
    "Retain Subject"
)

ax.set_ylabel(
    "Accuracy (%)"
)

ax.set_title(
    "Causal Validation by Retain Subject"
)

plt.xticks(
    rotation=35,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    "results/causal_validation.png",
    dpi=300
)

plt.close()


print("\nResults saved to:")
print(
    RESULTS_FILE
)

print(
    "results/causal_validation.png"
)
