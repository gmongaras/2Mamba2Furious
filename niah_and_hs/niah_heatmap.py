import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

lengths = [1024, 2048, 4096, 8192, 16384]
lengths_tok = [355, 631, 1182, 2286, 4497]
vals = {
    "Mamba-2": [0.94, 0.45, 0.15, 0.027, 0.025],
    "Softmax": [1.0, 1.0, 1.0, 0.94, 0.38],
    "2Mamba": [1.0, 1.0, 1.0, 1.0, 0.95],
}

row_labels = vals.keys()
# col_labels = [f"{l}/{l_t}" for l, l_t in zip(lengths, lengths_tok)]
col_labels = lengths_tok
data = np.array(
    [v for v in vals.values()]
)
df = pd.DataFrame(data, index=row_labels, columns=col_labels)

# Colormap
colors = [
    (0.85, 0.15, 0.15),  # red
    (0.15, 0.70, 0.25)   # green
]
n_bins = 100
cmap_name = "accuracy_red_yellow_green"
cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

# Create the heatmap
plt.figure(figsize=(8, data.shape[0]+1))
sns.heatmap(df, annot=True, fmt=".2f", cmap=cm, annot_kws={"size": 14}, cbar_kws={'label': 'proportion correct'})
plt.title('Needle in a haystack benchmark', fontsize=16, fontweight='bold')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12, rotation=0)
plt.ylabel("Model Type", fontsize=14, fontweight='bold')
plt.xlabel("Context Length (tokens)", fontsize=14, fontweight='bold')
plt.savefig("niah.svg", format="svg", bbox_inches="tight")

