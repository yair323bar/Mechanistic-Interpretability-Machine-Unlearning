import pandas as pd
import matplotlib.pyplot as plt


INTERVENTION_FILE = "results/intervention_results.csv"
ROBUSTNESS_FILE = "results/robustness_results.csv"

ALL_STRENGTHS_FILE = "results/all_strengths_table.csv"
FINAL_RESULTS_FILE = "results/final_results.csv"
ROBUSTNESS_PLOT = "results/robustness_comparison.png"


# Load results
intervention_df = pd.read_csv(
    INTERVENTION_FILE
)

robustness_df = pd.read_csv(
    ROBUSTNESS_FILE
)


# Method names
method_names = {
    "top_wmdp": "Top-k Localization",
    "top_selective": "WMDP-vs-Retain",
    "random": "Random",
    "bottom_wmdp": "Bottom-k"
}

intervention_df["display_method"] = (
    intervention_df["method"].map(
        method_names
    )
)


# Add baseline
baseline_row = pd.DataFrame(
    [
        {
            "method": "baseline",
            "layers": "None",
            "alpha": 1.0,
            "wmdp_accuracy": 586 / 1273,
            "delta_wmdp": 0.0,
            "retain_accuracy": 200 / 600,
            "delta_retain": 0.0,
            "display_method": "Baseline"
        }
    ]
)


# Build all strengths table
all_strengths_df = pd.concat(
    [
        baseline_row,
        intervention_df
    ],
    ignore_index=True
)

all_strengths_df = all_strengths_df[
    [
        "display_method",
        "layers",
        "alpha",
        "wmdp_accuracy",
        "delta_wmdp",
        "retain_accuracy",
        "delta_retain"
    ]
]

all_strengths_df.columns = [
    "method",
    "layers",
    "alpha",
    "wmdp_accuracy",
    "delta_wmdp",
    "retain_accuracy",
    "delta_retain"
]

all_strengths_df.to_csv(
    ALL_STRENGTHS_FILE,
    index=False
)


# Final comparison alpha
FINAL_ALPHA = 0.25

comparison_df = intervention_df[
    intervention_df["alpha"]
    == FINAL_ALPHA
].copy()

comparison_df = comparison_df[
    [
        "display_method",
        "layers",
        "alpha",
        "wmdp_accuracy",
        "delta_wmdp",
        "retain_accuracy",
        "delta_retain"
    ]
]

comparison_df.columns = [
    "method",
    "layers",
    "alpha",
    "wmdp_accuracy",
    "delta_wmdp",
    "retain_accuracy",
    "delta_retain"
]


# Add baseline to final table
final_baseline = pd.DataFrame(
    [
        {
            "method": "Baseline",
            "layers": "None",
            "alpha": 1.0,
            "wmdp_accuracy": 586 / 1273,
            "delta_wmdp": 0.0,
            "retain_accuracy": 200 / 600,
            "delta_retain": 0.0
        }
    ]
)

final_df = pd.concat(
    [
        final_baseline,
        comparison_df
    ],
    ignore_index=True
)

final_df.to_csv(
    FINAL_RESULTS_FILE,
    index=False
)


# Original prompt result
original_baseline = 586 / 1273

original_intervention = intervention_df[
    (
        intervention_df["method"]
        == "top_selective"
    )
    & (
        intervention_df["alpha"]
        == 0.25
    )
]["wmdp_accuracy"].iloc[0]

original_delta = (
    original_baseline
    - original_intervention
)


# Alternative prompt result
alternative_baseline = robustness_df[
    robustness_df["condition"]
    == "baseline"
]["wmdp_accuracy"].iloc[0]

alternative_intervention = robustness_df[
    robustness_df["condition"]
    == "top_selective"
]["wmdp_accuracy"].iloc[0]

alternative_delta = (
    alternative_baseline
    - alternative_intervention
)


# Build robustness table
robustness_comparison = pd.DataFrame(
    {
        "prompt": [
            "Original Prompt",
            "Alternative Prompt"
        ],
        "baseline": [
            original_baseline,
            alternative_baseline
        ],
        "intervention": [
            original_intervention,
            alternative_intervention
        ],
        "delta_wmdp": [
            original_delta,
            alternative_delta
        ]
    }
)

robustness_comparison.to_csv(
    "results/robustness_comparison.csv",
    index=False
)


# Create robustness plot
plot_df = robustness_comparison.copy()

plot_df["delta_wmdp"] *= 100

plt.figure(
    figsize=(8, 6)
)

bars = plt.bar(
    plot_df["prompt"],
    plot_df["delta_wmdp"]
)

plt.ylabel(
    "WMDP Accuracy Drop (percentage points)"
)

plt.title(
    "Robustness to Prompt Formulation"
)


# Add values
for bar, value in zip(
    bars,
    plot_df["delta_wmdp"]
):

    plt.text(
        bar.get_x()
        + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.2f}",
        ha="center",
        va="bottom"
    )


plt.tight_layout()

plt.savefig(
    ROBUSTNESS_PLOT,
    dpi=300
)

plt.close()


# Show results
print("\n==============================")
print("All Intervention Strengths")
print("==============================")

print(
    all_strengths_df.to_string(
        index=False
    )
)


print("\n==============================")
print("Final Comparison")
print("==============================")

print(
    final_df.to_string(
        index=False
    )
)


print("\n==============================")
print("Robustness Comparison")
print("==============================")

print(
    robustness_comparison.to_string(
        index=False
    )
)


print("\nFiles saved to:")

print(
    ALL_STRENGTHS_FILE
)

print(
    FINAL_RESULTS_FILE
)

print(
    "results/robustness_comparison.csv"
)

print(
    ROBUSTNESS_PLOT
)
