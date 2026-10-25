"""
=============================================================
  Soft Computing Mini Project 2
  Title  : Hopfield Network for Pattern Storage and Retrieval
  Author : Abhishek Yadav
  System id: 2026303464
=============================================================

Description:
  This project implements a Hopfield Network from scratch using
  only NumPy. The network:
    - Stores binary patterns (5x5 pixel letters/symbols)
    - Retrieves the original pattern from a noisy/corrupted version
    - Demonstrates associative memory behaviour

  Key concepts demonstrated:
    - Hebbian learning rule for weight computation
    - Synchronous / asynchronous update dynamics
    - Energy function landscape
    - Pattern recall from partial/noisy inputs
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
#  1.  HOPFIELD NETWORK CLASS
# ─────────────────────────────────────────────────────────────

class HopfieldNetwork:
    """
    Discrete Hopfield Neural Network.

    Attributes:
        n_neurons (int)         : number of neurons (= pattern size)
        W (np.ndarray)          : n x n symmetric weight matrix (diagonal = 0)
        stored_patterns (list)  : list of stored bipolar patterns
    """

    def __init__(self, n_neurons):
        """Initialise the network with zero weights.
        
        Args:
            n_neurons : int, number of neurons (must equal pattern dimensionality)
        """
        self.n_neurons = n_neurons
        self.W = np.zeros((n_neurons, n_neurons))
        self.stored_patterns = []

    # ── 1a. Learning (Hebbian rule) ───────────────────────────

    def train(self, patterns):
        """Store patterns using the outer-product (Hebb) learning rule.
        
        Weight update:  W += (1/N) * p * p^T,   diagonal forced to 0.
        
        Args:
            patterns : list of 1-D arrays of shape (n_neurons,), values in {-1, +1}
        """
        self.stored_patterns = list(patterns)
        N = self.n_neurons
        self.W = np.zeros((N, N))

        for p in patterns:
            p = np.array(p, dtype=float).reshape(-1)
            assert len(p) == N, "Pattern size must match n_neurons"
            self.W += np.outer(p, p)

        # Normalise and zero diagonal (no self-connections)
        self.W /= N
        np.fill_diagonal(self.W, 0)

    # ── 1b. Energy function ───────────────────────────────────

    def energy(self, state):
        """Compute the network energy E = -0.5 * s^T W s.
        
        Lower energy = more stable attractor state.
        
        Args:
            state : 1-D array of shape (n_neurons,), values in {-1, +1}
        Returns:
            float energy value
        """
        s = np.array(state, dtype=float)
        return -0.5 * s @ self.W @ s

    # ── 1c. Asynchronous update ───────────────────────────────

    def update_async(self, state, max_iter=20):
        """Run asynchronous update until convergence or max_iter.
        
        At each step a random neuron is chosen and its activation
        is computed; neurons are updated one at a time.
        
        Args:
            state    : initial state array (values -1 or +1)
            max_iter : maximum number of full passes over all neurons
        Returns:
            final_state   : converged state
            energy_history: list of energy values at each step
        """
        s = np.array(state, dtype=float).copy()
        energy_history = [self.energy(s)]
        n = self.n_neurons

        for iteration in range(max_iter):
            changed = False
            order = np.random.permutation(n)
            for idx in order:
                h_i = np.dot(self.W[idx], s)
                new_val = 1.0 if h_i >= 0 else -1.0
                if new_val != s[idx]:
                    s[idx] = new_val
                    changed = True
            energy_history.append(self.energy(s))
            if not changed:
                break   # converged

        return s, energy_history

    # ── 1d. Synchronous update ────────────────────────────────

    def update_sync(self, state, max_iter=20):
        """Run synchronous update (all neurons updated simultaneously).
        
        Args:
            state    : initial state array (values -1 or +1)
            max_iter : maximum number of iterations
        Returns:
            final_state   : converged state
            energy_history: list of energy values
        """
        s = np.array(state, dtype=float).copy()
        energy_history = [self.energy(s)]

        for _ in range(max_iter):
            s_new = np.sign(self.W @ s)
            s_new[s_new == 0] = 1   # tie-break
            energy_history.append(self.energy(s_new))
            if np.array_equal(s_new, s):
                break
            s = s_new

        return s, energy_history

    # ── 1e. Recall helper ─────────────────────────────────────

    def recall(self, noisy_pattern, mode='async', max_iter=20):
        """Recall a stored pattern from a noisy/partial probe.
        
        Args:
            noisy_pattern : 1-D array, the corrupted/partial input
            mode          : 'async' or 'sync'
            max_iter      : maximum update steps
        Returns:
            recalled_pattern, energy_history
        """
        if mode == 'async':
            return self.update_async(noisy_pattern, max_iter)
        else:
            return self.update_sync(noisy_pattern, max_iter)


# ─────────────────────────────────────────────────────────────
#  2.  PATTERN DEFINITIONS  (5x5 pixel letters)
# ─────────────────────────────────────────────────────────────

def define_patterns():
    """Define binary (bipolar) patterns representing 5x5 pixel letters.
    
    +1 = pixel ON  (black)
    -1 = pixel OFF (white)
    
    Returns:
        dict mapping letter name -> 1-D bipolar array of length 25
    """

    # Letter "A"
    A = [
        -1,  1, -1,  1, -1,
         1, -1,  1, -1,  1,
         1,  1,  1,  1,  1,
         1, -1,  1, -1,  1,
         1, -1,  1, -1,  1,
    ]

    # Letter "B"
    B = [
         1,  1,  1,  1, -1,
         1, -1,  1, -1,  1,
         1,  1,  1,  1, -1,
         1, -1,  1, -1,  1,
         1,  1,  1,  1, -1,
    ]

    # Letter "C"
    C = [
        -1,  1,  1,  1, -1,
         1, -1, -1, -1, -1,
         1, -1, -1, -1, -1,
         1, -1, -1, -1, -1,
        -1,  1,  1,  1, -1,
    ]

    # Letter "T"
    T = [
         1,  1,  1,  1,  1,
        -1, -1,  1, -1, -1,
        -1, -1,  1, -1, -1,
        -1, -1,  1, -1, -1,
        -1, -1,  1, -1, -1,
    ]

    patterns = {"A": A, "B": B, "C": C, "T": T}
    return {k: np.array(v, dtype=float) for k, v in patterns.items()}


# ─────────────────────────────────────────────────────────────
#  3.  NOISE INJECTION UTILITY
# ─────────────────────────────────────────────────────────────

def add_noise(pattern, noise_level=0.2, seed=None):
    """Corrupt a bipolar pattern by randomly flipping bits.
    
    Args:
        pattern     : 1-D bipolar array
        noise_level : fraction of bits to flip (0.0 = no noise, 1.0 = all flipped)
        seed        : optional random seed for reproducibility
    Returns:
        noisy_pattern : corrupted copy of input
    """
    rng = np.random.default_rng(seed)
    noisy = pattern.copy()
    n_flip = int(noise_level * len(pattern))
    flip_idx = rng.choice(len(pattern), size=n_flip, replace=False)
    noisy[flip_idx] *= -1
    return noisy


# ─────────────────────────────────────────────────────────────
#  4.  SIMILARITY / OVERLAP METRIC
# ─────────────────────────────────────────────────────────────

def overlap(p1, p2):
    """Compute normalised overlap (similarity) between two bipolar patterns.
    
    Overlap m = (1/N) * sum(p1_i * p2_i)
    m = 1  -> identical
    m = -1 -> inverted
    m = 0  -> uncorrelated
    
    Args:
        p1, p2 : bipolar 1-D arrays of same length
    Returns:
        float in [-1, 1]
    """
    return float(np.dot(p1, p2)) / len(p1)


# ─────────────────────────────────────────────────────────────
#  5.  MAIN DEMONSTRATION
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Soft Computing Project 2 - Hopfield Network")
    print("  Task: Pattern Storage and Retrieval")
    print("=" * 60)

    # 5a. Define and store patterns
    patterns = define_patterns()
    pattern_names = list(patterns.keys())
    pattern_list  = [patterns[k] for k in pattern_names]

    N = 25   # 5x5 grid
    net = HopfieldNetwork(N)
    net.train(pattern_list)

    print(f"\n  Network size   : {N} neurons")
    print(f"  Stored patterns: {pattern_names}")
    print(f"  Theoretical capacity: ~{int(0.138 * N)} patterns (0.138N rule)")
    print(f"  Weight matrix  : {N}x{N}, symmetric, zero diagonal\n")

    # 5b. Test recall at various noise levels
    noise_levels = [0.0, 0.10, 0.20, 0.30, 0.40]
    results = {}

    print("  Recall Accuracy vs Noise Level:")
    print("  " + "-" * 50)
    print(f"  {'Noise':>8}  {'Pattern':>10}  {'Overlap Before':>14}  {'Overlap After':>13}")
    print("  " + "-" * 50)

    for noise in noise_levels:
        results[noise] = {}
        for name in pattern_names:
            orig    = patterns[name]
            noisy   = add_noise(orig, noise_level=noise, seed=42)
            recalled, _ = net.recall(noisy, mode='async', max_iter=30)
            ov_before = overlap(orig, noisy)
            ov_after  = overlap(orig, recalled)
            results[noise][name] = {
                "noisy": noisy,
                "recalled": recalled,
                "overlap_before": ov_before,
                "overlap_after": ov_after
            }
            print(f"  {noise:>8.0%}  {name:>10}  {ov_before:>14.3f}  {ov_after:>13.3f}")
    print()

    # 5c. Energy convergence for one example
    orig  = patterns["A"]
    noisy = add_noise(orig, noise_level=0.30, seed=7)
    _, energy_history = net.recall(noisy, mode='async', max_iter=30)

    # ─────────────────────────────────────────────────────────
    #  6.  VISUALISATIONS
    # ─────────────────────────────────────────────────────────

    def show_pattern(ax, pattern, title, cmap='binary'):
        """Helper: display a single 5x5 pattern on an axes."""
        grid = pattern.reshape(5, 5)
        ax.imshow(grid, cmap=cmap, vmin=-1, vmax=1, interpolation='nearest')
        ax.set_title(title, fontsize=9)
        ax.axis('off')

    # ── Figure 1: Stored Patterns ─────────────────────────────
    fig1, axes = plt.subplots(1, len(pattern_names), figsize=(12, 3))
    fig1.suptitle("Stored Patterns (5x5 Pixel Letters)", fontsize=13, fontweight='bold')
    for ax, name in zip(axes, pattern_names):
        show_pattern(ax, patterns[name], f"'{name}'")
    plt.tight_layout()
    plt.savefig('hopfield_stored_patterns.png', dpi=150, bbox_inches='tight')

    # ── Figure 2: Recall pipeline for all patterns at 20% noise ──
    fig2 = plt.figure(figsize=(16, 4 * len(pattern_names)))
    fig2.suptitle("Hopfield Network - Pattern Recall Pipeline (20% Noise)", 
                  fontsize=13, fontweight='bold')
    gs = gridspec.GridSpec(len(pattern_names), 4, figure=fig2, hspace=0.5, wspace=0.3)

    for row, name in enumerate(pattern_names):
        orig     = patterns[name]
        noisy    = results[0.20][name]["noisy"]
        recalled = results[0.20][name]["recalled"]
        ov_b     = results[0.20][name]["overlap_before"]
        ov_a     = results[0.20][name]["overlap_after"]

        ax_o = fig2.add_subplot(gs[row, 0])
        ax_n = fig2.add_subplot(gs[row, 1])
        ax_r = fig2.add_subplot(gs[row, 2])
        ax_s = fig2.add_subplot(gs[row, 3])

        show_pattern(ax_o, orig,     f"Original '{name}'")
        show_pattern(ax_n, noisy,    f"Noisy (m={ov_b:.2f})")
        show_pattern(ax_r, recalled, f"Recalled (m={ov_a:.2f})")

        # Difference map
        diff = (orig != recalled).reshape(5, 5).astype(float)
        ax_s.imshow(diff, cmap='Reds', vmin=0, vmax=1, interpolation='nearest')
        n_wrong = int(diff.sum())
        ax_s.set_title(f"Error Map ({n_wrong} bit{'s' if n_wrong != 1 else ''} wrong)", fontsize=9)
        ax_s.axis('off')

    plt.savefig('hopfield_recall_pipeline.png', dpi=150, bbox_inches='tight')

    # ── Figure 3: Noise robustness curves ────────────────────
    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))
    fig3.suptitle("Hopfield Network - Noise Robustness Analysis", fontsize=13, fontweight='bold')

    ax_curve  = axes3[0]
    ax_energy = axes3[1]

    noise_pct = [n * 100 for n in noise_levels]
    for name in pattern_names:
        overlaps_after = [results[n][name]["overlap_after"] for n in noise_levels]
        ax_curve.plot(noise_pct, overlaps_after, marker='o', lw=2, label=f"'{name}'")

    ax_curve.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='0.8 threshold')
    ax_curve.set_title('Recall Overlap vs Noise Level')
    ax_curve.set_xlabel('Noise Level (% bits flipped)')
    ax_curve.set_ylabel('Overlap with Original Pattern')
    ax_curve.set_ylim(-0.1, 1.1)
    ax_curve.legend()
    ax_curve.grid(alpha=0.3)

    # Energy convergence curve
    ax_energy.plot(range(len(energy_history)), energy_history,
                   'b-o', lw=2, markersize=5)
    ax_energy.set_title("Energy Convergence (Pattern 'A', 30% noise)")
    ax_energy.set_xlabel('Update Step')
    ax_energy.set_ylabel('Network Energy')
    ax_energy.grid(alpha=0.3)
    ax_energy.fill_between(range(len(energy_history)), energy_history,
                            alpha=0.15, color='blue')

    plt.tight_layout()
    plt.savefig('hopfield_analysis.png', dpi=150, bbox_inches='tight')

    # ── Figure 4: Weight matrix heatmap ──────────────────────
    fig4, (ax_w, ax_ov) = plt.subplots(1, 2, figsize=(14, 6))
    fig4.suptitle("Hopfield Network - Weight Matrix & Pattern Correlations",
                  fontsize=13, fontweight='bold')

    im = ax_w.imshow(net.W, cmap='RdBu', interpolation='nearest')
    ax_w.set_title('Weight Matrix W (25x25)')
    ax_w.set_xlabel('Neuron j'); ax_w.set_ylabel('Neuron i')
    plt.colorbar(im, ax=ax_w, fraction=0.046, pad=0.04)

    # Pattern correlation matrix
    n_pat = len(pattern_names)
    corr_matrix = np.zeros((n_pat, n_pat))
    for i, ni in enumerate(pattern_names):
        for j, nj in enumerate(pattern_names):
            corr_matrix[i, j] = overlap(patterns[ni], patterns[nj])
    im2 = ax_ov.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    ax_ov.set_xticks(range(n_pat)); ax_ov.set_yticks(range(n_pat))
    ax_ov.set_xticklabels(pattern_names); ax_ov.set_yticklabels(pattern_names)
    ax_ov.set_title('Pattern Correlation Matrix')
    plt.colorbar(im2, ax=ax_ov, fraction=0.046, pad=0.04)
    for i in range(n_pat):
        for j in range(n_pat):
            ax_ov.text(j, i, f'{corr_matrix[i,j]:.2f}', ha='center', va='center',
                       fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig('hopfield_weights.png', dpi=150, bbox_inches='tight')

    print("  Plots saved:")
    print("    -> hopfield_stored_patterns.png")
    print("    -> hopfield_recall_pipeline.png")
    print("    -> hopfield_analysis.png")
    print("    -> hopfield_weights.png")
    plt.show()

    print("\n" + "=" * 60)
    print("  Hopfield Network Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
