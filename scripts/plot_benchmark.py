import matplotlib.pyplot as plt

# Data from your experiment
energies = [-1.13731, -1.11666, -1.11567]
labels = ['Exact (FCI)', 'SPSA (Cloud)', 'COBYLA (Cloud)']
colors = ['black', 'green', 'blue']

plt.figure(figsize=(8, 6))
bars = plt.bar(labels, energies, color=colors, alpha=0.7, width=0.5)

# Add value labels on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval - 0.002, round(yval, 5), 
             ha='center', va='top', color='white', fontweight='bold')

# Zoom in to see the difference (Molecular scale)
plt.ylim(-1.145, -1.10)
plt.ylabel('Ground State Energy (Ha)')
plt.title('Hardware Validation: Accuracy Benchmark')
plt.grid(axis='y', alpha=0.3)

plt.savefig('results/h2/figures/accuracy_benchmark.png')
print("Graph saved to results/h2/figures/accuracy_benchmark.png")