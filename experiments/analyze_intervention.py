import pandas as pd
import matplotlib.pyplot as plt


RESULTS_FILE = "results/intervention_results.csv"

BASELINE_WMDP = 586 / 1273
BASELINE_RETAIN = 200 / 600


# Load results
print("Loading intervention results...")

df = pd.read_csv(
    RESULTS_FILE
)


# Method names
METHOD_NAMES = {
    "top_wmdp": "Top-k Localization",
    "top_selective": "WMDP-vs-Retain",
    "random": "Random",
    "bottom_wmdp": "Bottom-k"
}


# Add baseline for alpha 1
# Alpha 1 means no intervention
baseline_rows = []

for method in df["method"].unique():

    baseline_rows.append(
        {
            "method": method,
            "alpha": 1.0,
            "wmdp_accuracy": BASELINE_WMDP,
            "retain_accuracy": BASELINE_RETAIN,
            "delta_wmdp": 0.0,
            "delta_retain": 0.0
        }
    )


plot_df = pd.concat(
    [
        pd.DataFrame(baseline_rows),
        df
    ],
    ignore_index=True
)


# Sort results by alpha
plot_df = plot_df.sort_values(
    [
        "method",
        "alpha"
    ]
)


# Show results
print("\nIntervention Results:")

print(
    df[
        [
            "method",
            "layers",
            "alpha",
            "wmdp_accuracy",
            "delta_wmdp",
            "retain_accuracy",
            "delta_retain"
        ]
    ].to_string(
        index=False
    )
)


# WMDP graph
plt.figure(
    figsize=(9, 6)
)

for method in plot_df["method"].unique():

    method_data = plot_df[
        plot_df["method"] == method
    ].sort_values(
        "alpha"
    )

    plt.plot(
        method_data["alpha"],
        method_data["wmdp_accuracy"] * 100,
        marker="o",
        label=METHOD_NAMES[method]
    )


plt.xlabel(
    "Intervention Strength (Alpha)"
)

plt.ylabel(
    "WMDP Accuracy (%)"
)

plt.title(
    "WMDP Accuracy vs Intervention Strength"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "results/wmdp_intervention_curve.png",
    dpi=300
)

plt.close()


# Retain graph
plt.figure(
    figsize=(9, 6)
)

for method in plot_df["method"].unique():

    method_data = plot_df[
        plot_df["method"] == method
    ].sort_values(
        "alpha"
    )

    plt.plot(
        method_data["alpha"],
        method_data["retain_accuracy"] * 100,
        marker="o",
        label=METHOD_NAMES[method]
    )


plt.xlabel(
    "Intervention Strength (Alpha)"
)

plt.ylabel(
    "Retain Accuracy (%)"
)

plt.title(
    "Retain Accuracy vs Intervention Strength"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "results/retain_intervention_curve.png",
    dpi=300
)

plt.close()


# Compare WMDP drop and Retain drop
plt.figure(
    figsize=(9, 6)
)

for method in df["method"].unique():

    method_data = df[
        df["method"] == method
    ].sort_values(
        "alpha"
    )

    plt.plot(
        method_data["delta_retain"] * 100,
        method_data["delta_wmdp"] * 100,
        marker="o",
        label=METHOD_NAMES[method]
    )


plt.xlabel(
    "Retain Accuracy Drop (%)"
)

plt.ylabel(
    "WMDP Accuracy Drop (%)"
)

plt.title(
    "Selective Unlearning Trade-off"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "results/selectivity_tradeoff.png",
    dpi=300
)

plt.close()


# Create summary table
summary_rows = []

for method in df["method"].unique():

    method_data = df[
        df["method"] == method
    ]

    # Find strongest WMDP reduction
    best_index = (
        method_data["delta_wmdp"]
        .idxmax()
    )

    best_row = df.loc[
        best_index
    ]

    summary_rows.append(
        {
            "method": METHOD_NAMES[method],
            "layers": best_row["layers"],
            "alpha": best_row["alpha"],
            "wmdp_accuracy": best_row["wmdp_accuracy"],
            "delta_wmdp": best_row["delta_wmdp"],
            "retain_accuracy": best_row["retain_accuracy"],
            "delta_retain": best_row["delta_retain"]
        }
    )


summary_df = pd.DataFrame(
    summary_rows
)


# Add baseline to summary
baseline_row = pd.DataFrame(
    [
        {
            "method": "Baseline",
            "layers": "-",
            "alpha": 1.0,
            "wmdp_accuracy": BASELINE_WMDP,
            "delta_wmdp": 0.0,
            "retain_accuracy": BASELINE_RETAIN,
            "delta_retain": 0.0
        }
    ]
)


summary_df = pd.concat(
    [
        baseline_row,
        summary_df
    ],
    ignore_index=True
)


# Save summary
summary_df.to_csv(
    "results/intervention_summary.csv",
    index=False
)


# Show summary
print("\n--------------------------------")
print("Summary Table")
print("--------------------------------")

print(
    summary_df.to_string(
        index=False
    )
)


print("\nResults saved to:")
print("results/wmdp_intervention_curve.png")
print("results/retain_intervention_curve.png")
print("results/selectivity_tradeoff.png")
print("results/intervention_summary.csv")
