import pandas as pd
import matplotlib.pyplot as plt


RESULTS_FILE = "results/localization_scores.csv"

WMDP_PLOT = "results/wmdp_localization.png"
SELECTIVE_PLOT = "results/selective_localization.png"


# Load results
df = pd.read_csv(
    RESULTS_FILE
)


# Show columns
print("Columns:")
print(
    df.columns.tolist()
)

print("\nLocalization results:")
print(
    df.to_string(
        index=False
    )
)


# WMDP localization plot
plt.figure(
    figsize=(10, 6)
)

plt.plot(
    df["layer"],
    df["wmdp_score"],
    marker="o"
)

plt.xlabel(
    "Layer"
)

plt.ylabel(
    "Residual-Update L2 Score"
)

plt.title(
    "WMDP-Bio Localization Across Layers"
)

plt.xticks(
    df["layer"]
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    WMDP_PLOT,
    dpi=300
)

plt.close()


# Selective localization plot
plt.figure(
    figsize=(10, 6)
)

plt.plot(
    df["layer"],
    df["selective_score"],
    marker="o"
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel(
    "Layer"
)

plt.ylabel(
    "WMDP-vs-Retain Selective Score"
)

plt.title(
    "WMDP-vs-Retain Selective Localization"
)

plt.xticks(
    df["layer"]
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    SELECTIVE_PLOT,
    dpi=300
)

plt.close()


# Show top layers
top_wmdp = df.sort_values(
    "wmdp_score",
    ascending=False
).head(5)

top_selective = df.sort_values(
    "selective_score",
    ascending=False
).head(5)


print("\nTop WMDP layers:")

print(
    top_wmdp[
        [
            "layer",
            "wmdp_score"
        ]
    ].to_string(
        index=False
    )
)


print("\nTop Selective layers:")

print(
    top_selective[
        [
            "layer",
            "selective_score"
        ]
    ].to_string(
        index=False
    )
)


print("\nPlots saved to:")

print(
    WMDP_PLOT
)

print(
    SELECTIVE_PLOT
)
