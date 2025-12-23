import matplotlib.pyplot as plt

def plot_convergence(history, exact_energy=None, title="VQE Convergence", save_path=None):
    plt.figure(figsize=(10, 6))
    plt.plot(history, label="VQE Energy", linewidth=2)
    
    if exact_energy is not None:
        plt.axhline(exact_energy, color='k', linestyle='--', label="Exact")
        
    plt.xlabel("Iterations")
    plt.ylabel("Energy (Ha)")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")