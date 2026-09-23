"""Convergence plotting."""

import matplotlib.pyplot as plt

def plot_convergence(history, exact_energy=None, title="VQE Convergence", save_path=None):
    """Plot the energy at each optimizer evaluation, optionally against the exact value.

    Opens a new figure and saves it only if ``save_path`` is given; the figure
    is neither shown nor closed.
    """
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