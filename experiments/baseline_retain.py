import torch
import random
from collections import Counter
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

SAMPLES_PER_SUBJECT = 100
RANDOM_SEED = 42

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


# Load MMLU
print("Loading MMLU...")
dataset = load_dataset(
    "cais/mmlu",
    "all",
    split="test"
)


# Build retain dataset
random.seed(RANDOM_SEED)

retain_examples = []

for subject in RETAIN_SUBJECTS:

    subject_examples = [
        example for example in dataset
        if example["subject"] == subject
    ]

    selected_examples = random.sample(
        subject_examples,
        SAMPLES_PER_SUBJECT
    )

    retain_examples.extend(selected_examples)


print("\nRetain dataset created successfully!")
print(f"Subjects: {len(RETAIN_SUBJECTS)}")
print(f"Examples per subject: {SAMPLES_PER_SUBJECT}")
print(f"Total examples: {len(retain_examples)}")


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
        outputs = model(input_ids=prompt_ids)

    logits = outputs.logits

    # Last position predicts the next token
    next_token_logits = logits[0, -1]

    log_probs = torch.log_softmax(
        next_token_logits,
        dim=-1
    )

    scores = []

    for letter in letters:
        token_id = tokenizer(
            letter,
            add_special_tokens=False
        ).input_ids[0]

        scores.append(
            log_probs[token_id].item()
        )

    return scores


# Evaluate
letters = ["A", "B", "C", "D"]

correct = 0
predictions = []

num_examples = len(retain_examples)

print(f"\nEvaluating {num_examples} retain examples...\n")

for i in range(num_examples):

    example = retain_examples[i]

    question = example["question"]
    choices = example["choices"]
    correct_answer = example["answer"]
    subject = example["subject"]

    prompt = build_prompt(question, choices)

    scores = score_candidates(prompt)

    predicted_answer = max(
        range(len(scores)),
        key=lambda x: scores[x]
    )

    is_correct = predicted_answer == correct_answer

    if is_correct:
        correct += 1

    predictions.append(
        {
            "index": i,
            "subject": subject,
            "correct": correct_answer,
            "prediction": predicted_answer,
            "scores": scores
        }
    )

    print(
        f"Question {i + 1:03d} | "
        f"Subject: {subject} | "
        f"Prediction: {letters[predicted_answer]} | "
        f"Correct: {letters[correct_answer]} | "
        f"{'✓' if is_correct else '✗'}"
    )


# Prediction distribution
prediction_counts = Counter(
    p["prediction"] for p in predictions
)

print("\nPrediction distribution:")
for index, letter in enumerate(letters):
    print(
        f"{letter}: "
        f"{prediction_counts.get(index, 0)}"
    )


# Accuracy per subject
print("\nAccuracy per subject:")

for subject in RETAIN_SUBJECTS:

    subject_predictions = [
        p for p in predictions
        if p["subject"] == subject
    ]

    subject_correct = sum(
        p["prediction"] == p["correct"]
        for p in subject_predictions
    )

    subject_accuracy = (
        subject_correct / len(subject_predictions)
    )

    print(
        f"{subject}: "
        f"{subject_correct}/{len(subject_predictions)} "
        f"({subject_accuracy:.2%})"
    )


# Final accuracy
accuracy = correct / num_examples

print("\n--------------------------------")
print("MMLU Retain Baseline")
print("--------------------------------")
print(f"Correct:  {correct}/{num_examples}")
print(f"Accuracy: {accuracy:.2%}")
print("--------------------------------")


# Save predictions to CSV
results = []

for p in predictions:
    results.append(
        {
            "index": p["index"],
            "subject": p["subject"],
            "correct": letters[p["correct"]],
            "prediction": letters[p["prediction"]],
            "score_A": p["scores"][0],
            "score_B": p["scores"][1],
            "score_C": p["scores"][2],
            "score_D": p["scores"][3]
        }
    )

results_df = pd.DataFrame(results)

results_df.to_csv(
    "results/retain_baseline_predictions.csv",
    index=False
)

print("\nResults saved to:")
print("results/retain_baseline_predictions.csv")
