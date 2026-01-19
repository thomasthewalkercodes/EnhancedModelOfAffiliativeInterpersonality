"""
Plotting functions for the simulation.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_probabilities(p_history, q_history, scenario_name, save_path=None):
    """
    Plot the probability trajectories over time.

    For multiple actions, shows probability of each action for each agent.
    """

    rounds = len(p_history)
    t = np.arange(rounds)

    # Check if 1D (old format) or 2D (new format)
    if p_history.ndim == 1:
        # Old format: 2 actions
        p = p_history
        q = q_history

        prob_AA = p * q  # Both Approach
        prob_AV = p * (1 - q)  # A1 Approach, A2 Avoid
        prob_VA = (1 - p) * q  # A1 Avoid, A2 Approach
        prob_VV = (1 - p) * (1 - q)  # Both Avoid

        # Create plot
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        # Plot 1: Individual probabilities
        ax1 = axes[0]
        ax1.plot(t, p_history, "b-", label="P(Agent1 = Approach)", linewidth=2)
        ax1.plot(t, q_history, "r-", label="P(Agent2 = Approach)", linewidth=2)
        ax1.set_xlabel("Round")
        ax1.set_ylabel("Probability")
        ax1.set_title(f"Individual Action Probabilities - {scenario_name}")
        ax1.legend()
        ax1.set_ylim(-0.05, 1.05)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Joint probabilities
        ax2 = axes[1]
        ax2.plot(t, prob_AA, "g-", label="Both Approach", linewidth=2)
        ax2.plot(t, prob_AV, "m-", label="A1 Approach, A2 Avoid", linewidth=2)
        ax2.plot(t, prob_VA, "c-", label="A1 Avoid, A2 Approach", linewidth=2)
        ax2.plot(t, prob_VV, "y-", label="Both Avoid", linewidth=2)
        ax2.set_xlabel("Round")
        ax2.set_ylabel("Probability")
        ax2.set_title(f"Joint Outcome Probabilities - {scenario_name}")
        ax2.legend()
        ax2.set_ylim(-0.05, 1.05)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Plot saved to: {save_path}")

        plt.show()

        # Print final state
        print(f"\nFinal probabilities (round {rounds}):")
        print(f"  P(Agent1 = Approach) = {p_history[-1]:.3f}")
        print(f"  P(Agent2 = Approach) = {q_history[-1]:.3f}")
        print(f"\nFinal joint probabilities:")
        print(f"  Both Approach:         {prob_AA[-1]:.3f}")
        print(f"  A1 Approach, A2 Avoid: {prob_AV[-1]:.3f}")
        print(f"  A1 Avoid, A2 Approach: {prob_VA[-1]:.3f}")
        print(f"  Both Avoid:            {prob_VV[-1]:.3f}")

        # Determine most likely outcome
        final_probs = {
            "Both Approach": prob_AA[-1],
            "A1 Approach, A2 Avoid": prob_AV[-1],
            "A1 Avoid, A2 Approach": prob_VA[-1],
            "Both Avoid": prob_VV[-1],
        }
        most_likely = max(final_probs, key=final_probs.get)
        print(f"\nMost likely outcome: {most_likely} ({final_probs[most_likely]:.3f})")

    else:
        # New format: multiple actions
        n_actions = p_history.shape[1]

        # Create plot for each agent's action probabilities
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # Plot Agent 1 action probabilities
        ax1 = axes[0]
        for i in range(n_actions):
            ax1.plot(t, p_history[:, i], label=f"Action {i}", linewidth=2)
        ax1.set_xlabel("Round")
        ax1.set_ylabel("Probability")
        ax1.set_title(f"Agent 1 Action Probabilities - {scenario_name}")
        ax1.legend()
        ax1.set_ylim(-0.05, 1.05)
        ax1.grid(True, alpha=0.3)

        # Plot Agent 2 action probabilities
        ax2 = axes[1]
        for i in range(n_actions):
            ax2.plot(t, q_history[:, i], label=f"Action {i}", linewidth=2)
        ax2.set_xlabel("Round")
        ax2.set_ylabel("Probability")
        ax2.set_title(f"Agent 2 Action Probabilities - {scenario_name}")
        ax2.legend()
        ax2.set_ylim(-0.05, 1.05)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Plot saved to: {save_path}")

        plt.show()

        # Print final state
        print(f"\nFinal action probabilities (round {rounds}):")
        print("Agent 1:")
        for i in range(n_actions):
            print(f"  P(Action {i}) = {p_history[-1, i]:.3f}")
        print("Agent 2:")
        for i in range(n_actions):
            print(f"  P(Action {i}) = {q_history[-1, i]:.3f}")

        # Most likely actions
        a1_most = np.argmax(p_history[-1])
        a2_most = np.argmax(q_history[-1])
        print(f"\nMost likely actions: Agent1 = {a1_most}, Agent2 = {a2_most}")
