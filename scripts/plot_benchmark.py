"""Plot the H2 accuracy bar chart (results/h2/figures/accuracy_benchmark.png).

The energies are hard-coded below; no script in this repository produces them.
"""

import matplotlib.pyplot as plt

# Hard-coded; the run that produced these values is not recorded in this repo.
energies = [-1.13731, -1.11666, -1.11567]
labels = ['Exact (FCI)', 'SPSA (Cloud)', 'COBYLA (Cloud)']
colors = ['black', 'green', 'blue']

plt.figure(figsize=(8, 6))
bars = plt.bar(labels, energies, color=colors, alpha=0.7, width=0.5)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval - 0.002, round(yval, 5), 
             ha='center', va='top', color='white', fontweight='bold')

# Narrow y-range: the differences are a few mHa on a ~1.1 Ha scale.
plt.ylim(-1.145, -1.10)
plt.ylabel('Ground State Energy (Ha)')
plt.title('Hardware Validation: Accuracy Benchmark')
plt.grid(axis='y', alpha=0.3)

plt.savefig('results/h2/figures/accuracy_benchmark.png')
print("Graph saved to results/h2/figures/accuracy_benchmark.png")