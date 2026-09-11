from datasets import load_dataset

DATASET_NAME = "cais/wmdp"
CONFIG_NAME = "wmdp-bio"

print("Loading WMDP-Bio...")

dataset = load_dataset(
    DATASET_NAME,
    CONFIG_NAME,
    split="test"
)

print("Dataset loaded successfully!")

print("\nDataset columns:")
print(dataset.column_names)

print("\nNumber of examples:")
print(len(dataset))

print("\nFirst example:")
print(dataset[0])

print("\nCorrect answer index:")
print(dataset[0]["answer"])

print("\nCorrect answer text:")
print(dataset[0]["choices"][dataset[0]["answer"]])

