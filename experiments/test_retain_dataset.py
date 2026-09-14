from datasets import load_dataset


# Load MMLU
print("Loading MMLU...")

dataset = load_dataset(
    "cais/mmlu",
    "all",
    split="test"
)

print("MMLU loaded successfully!")


# Dataset information
print("\nNumber of examples:")
print(len(dataset))

print("\nDataset columns:")
print(dataset.column_names)


# First example
print("\nFirst example:")
print(dataset[0])

# Show subjects
subjects = dataset.unique("subject")

print("\nNumber of subjects:")
print(len(subjects))

print("\nSubjects:")
for subject in sorted(subjects):
    print(subject)

# Selected retain subjects
retain_subjects = [
    "elementary_mathematics",
    "formal_logic",
    "high_school_geography",
    "high_school_world_history",
    "high_school_macroeconomics",
    "high_school_computer_science"
]

print("\nSelected retain subjects:")

for subject in retain_subjects:
    count = sum(
        1 for example in dataset
        if example["subject"] == subject
    )

    print(f"{subject}: {count}")
    