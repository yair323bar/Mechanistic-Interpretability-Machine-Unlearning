import os
import random
import torch
import pandas as pd

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

RANDOM_SEED = 42
SAMPLES_PER_SUBJECT = 100

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LETTERS = ["A", "B", "C", "D"]


# Test mode
# True  = small test before the full experiment
# False = full experiment
TEST_MODE = False #True

TEST_WMDP_EXAMPLES = 20
TEST_RETAIN_EXAMPLES = 20


# Baseline results
BASELINE_WMDP = 586 / 1273
BASELINE_RETAIN = 200 / 600


# Intervention strength
ALPHAS = [
    0.75,
    0.50,
    0.25,
    0.00
]


# Selected retain subjects
RETAIN_SUBJECTS = [
    "elementary_mathematics",
    "formal_logic",
    "high_school_geography",
    "high_school_world_history",
    "high_school_macroeconomics",
    "high_school_computer_science"
]


# Layer groups from localization
TOP_WMDP_LAYERS = [
    15,
    14
]

TOP_SELECTIVE_LAYERS = [
    11,
    13
]

BOTTOM_WMDP_LAYERS = [
    0,
    2
]


# Create random control
# Exclude layers already used by the other groups
# This keeps the random control separate from the selected groups
excluded_layers = set(
    TOP_WMDP_LAYERS
    + TOP_SELECTIVE_LAYERS
    + BOTTOM_WMDP_LAYERS
)

random_candidates = [
    layer
    for layer in range(16)
    if layer not in excluded_layers
]

random.seed(RANDOM_SEED)

RANDOM_LAYERS = random.sample(
    random_candidates,
    2
)


# Intervention methods
METHODS = {
    "top_wmdp": TOP_WMDP_LAYERS,
    "top_selective": TOP_SELECTIVE_LAYERS,
    "random": RANDOM_LAYERS,
    "bottom_wmdp": BOTTOM_WMDP_LAYERS
}


# Results file
if TEST_MODE:
    RESULTS_FILE = (
        "results/intervention_test_results.csv"
    )
else:
    RESULTS_FILE = (
        "results/intervention_results.csv"
    )


# Load model
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

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


# Use small datasets in test mode
if TEST_MODE:

    wmdp_eval_dataset = [
        wmdp_dataset[i]
        for i in range(TEST_WMDP_EXAMPLES)
    ]

    retain_eval_dataset = (
        retain_dataset[:TEST_RETAIN_EXAMPLES]
    )

else:

    wmdp_eval_dataset = wmdp_dataset
    retain_eval_dataset = retain_dataset


print("\nExperiment configuration:")
print(
    f"Test mode: {TEST_MODE}"
)

print(
    f"WMDP examples: "
    f"{len(wmdp_eval_dataset)}"
)

print(
    f"Retain examples: "
    f"{len(retain_eval_dataset)}"
)

print(
    f"Top WMDP layers: "
    f"{TOP_WMDP_LAYERS}"
)

print(
    f"Top Selective layers: "
    f"{TOP_SELECTIVE_LAYERS}"
)

print(
    f"Random layers: "
    f"{RANDOM_LAYERS}"
)

print(
    f"Bottom WMDP layers: "
    f"{BOTTOM_WMDP_LAYERS}"
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


# Create residual update intervention
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


# Register intervention hooks
def register_intervention(
    selected_layers,
    alpha
):

    hooks = []

    for layer_index in selected_layers:

        hook = (
            model.model.layers[layer_index]
            .register_forward_hook(
                create_intervention_hook(
                    alpha
                )
            )
        )

        hooks.append(hook)

    return hooks


# Remove intervention hooks
def remove_hooks(hooks):

    for hook in hooks:
        hook.remove()


# Evaluate dataset
def evaluate_dataset(
    dataset,
    dataset_name,
    selected_layers,
    alpha
):

    hooks = register_intervention(
        selected_layers,
        alpha
    )

    correct = 0

    print(
        f"\nEvaluating "
        f"{dataset_name} | "
        f"alpha={alpha}"
    )

    try:

        for i, example in enumerate(dataset):

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

            correct_answer = (
                example["answer"]
            )

            if (
                predicted_answer
                == correct_answer
            ):
                correct += 1

            print(
                f"{dataset_name} "
                f"{i + 1}/{len(dataset)}"
            )

    finally:

        # Always remove hooks
        remove_hooks(hooks)

    accuracy = (
        correct / len(dataset)
    )

    return accuracy


# Load previous results if they exist
if os.path.exists(RESULTS_FILE):

    results_df = pd.read_csv(
        RESULTS_FILE
    )

    results = results_df.to_dict(
        "records"
    )

    print(
        "\nExisting results found."
    )

else:

    results = []


# Check if condition already finished
def condition_completed(
    method_name,
    alpha
):

    for result in results:

        if (
            result["method"]
            == method_name
            and float(result["alpha"])
            == float(alpha)
        ):
            return True

    return False


# Run intervention experiment
for method_name, selected_layers in METHODS.items():

    for alpha in ALPHAS:

        # Skip finished condition
        if condition_completed(
            method_name,
            alpha
        ):

            print(
                f"\nSkipping completed condition: "
                f"{method_name}, "
                f"alpha={alpha}"
            )

            continue


        print("\n================================")
        print("Starting intervention condition")
        print("================================")

        print(
            f"Method: {method_name}"
        )

        print(
            f"Layers: {selected_layers}"
        )

        print(
            f"Alpha: {alpha}"
        )


        # Evaluate WMDP
        wmdp_accuracy = evaluate_dataset(
            wmdp_eval_dataset,
            "WMDP",
            selected_layers,
            alpha
        )


        # Evaluate Retain
        retain_accuracy = evaluate_dataset(
            retain_eval_dataset,
            "Retain",
            selected_layers,
            alpha
        )


        # Calculate delta from baseline
        delta_wmdp = (
            BASELINE_WMDP
            - wmdp_accuracy
        )

        delta_retain = (
            BASELINE_RETAIN
            - retain_accuracy
        )


        # Save condition result
        result = {
            "method": method_name,
            "layers": str(
                selected_layers
            ),
            "alpha": alpha,
            "wmdp_accuracy": (
                wmdp_accuracy
            ),
            "delta_wmdp": (
                delta_wmdp
            ),
            "retain_accuracy": (
                retain_accuracy
            ),
            "delta_retain": (
                delta_retain
            )
        }

        results.append(result)


        # Save checkpoint
        results_df = pd.DataFrame(
            results
        )

        results_df.to_csv(
            RESULTS_FILE,
            index=False
        )


        print("\nCondition result:")

        print(
            f"WMDP Accuracy: "
            f"{wmdp_accuracy:.2%}"
        )

        print(
            f"Delta WMDP: "
            f"{delta_wmdp:.2%}"
        )

        print(
            f"Retain Accuracy: "
            f"{retain_accuracy:.2%}"
        )

        print(
            f"Delta Retain: "
            f"{delta_retain:.2%}"
        )

        print(
            f"\nCheckpoint saved to:"
        )

        print(
            RESULTS_FILE
        )


# Show final results
results_df = pd.DataFrame(
    results
)

print("\n================================")
print("Intervention Experiment Results")
print("================================")

print(
    results_df.to_string(
        index=False
    )
)

print("\nResults saved to:")
print(RESULTS_FILE)
