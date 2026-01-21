"""
Interface for running evolutionary tournaments.

Modify the CONFIG below to change experiment parameters.
"""

import numpy as np
from tournament import run_tournament, OCTANT_ORDER

# =============================================================================
# EXPERIMENT CONFIGURATION - MODIFY THESE
# =============================================================================

CONFIG = {
    # Population settings
    "population_size": 20,  # Number of agents in population
    "generations": 1000,  # Number of evolutionary generations
    # Interaction settings
    "interactions_per_agent": 5,  # How many partners each agent meets per gen
    "rounds_per_interaction": 1000,  # Rounds per relationship
    # Learning parameters
    "l_rate": 0.1,  # Q-learning rate (alpha)
    "e_rate": 5.0,  # Softmax temperature (beta)
    "update_interval": 10,  # How often to update self-model and motives
    # Evolution parameters
    "survival_ratio": 0.4,  # Top 30% survive and reproduce
    "mutation_rate": 0.1,  # Probability of mutating each trait
    # Output
    "verbose": True,  # Print progress each generation
}


# =============================================================================
# PAYOFF MATRIX - MODIFY THIS
# =============================================================================


def create_payoff_matrix(n_actions=8):
    """
    Create the base payoff matrix.

    Modify this function to change the game being played.
    Current: Complementarity - reward for circumplex-adjacent actions.
    """
    base = np.zeros((n_actions, n_actions))
    for i in range(n_actions):
        for j in range(n_actions):
            # Circular distance on the circumplex
            dist = min(abs(i - j), n_actions - abs(i - j))
            # Higher payoff for closer actions (complementarity)
            base[i, j] = 1.0 - dist / (n_actions / 2)
    return base


# Alternative payoff matrices you can use:


def create_matching_matrix(n_actions=8):
    """Reward for exact matching (identity)."""
    return np.eye(n_actions)


def create_warmth_matrix(n_actions=8):
    """Reward warm responses regardless of own action."""
    # LM=0, NO=1, PA=2 are warm; DE=4, FG=5, HI=6 are cold
    warmth = np.array([1.0, 0.7, 0.7, 0.0, -1.0, -0.7, -0.7, 0.0])
    base = np.zeros((n_actions, n_actions))
    for i in range(n_actions):
        for j in range(n_actions):
            base[i, j] = warmth[j]  # Only other's action matters
    return base


def create_dominance_game(n_actions=8):
    """Asymmetric: dominant actions beat submissive."""
    # BC=3, DE=4 are dominant; HI=6, JK=7 are submissive
    dom = np.array([0.0, 0.3, 0.5, 1.0, 0.8, 0.3, -0.5, -0.8])
    base = np.zeros((n_actions, n_actions))
    for i in range(n_actions):
        for j in range(n_actions):
            base[i, j] = dom[i] - dom[j]  # Win if more dominant
    return base


# =============================================================================
# RUN EXPERIMENT
# =============================================================================

if __name__ == "__main__":
    # Select payoff matrix
    CONFIG["base_payoff"] = create_payoff_matrix(n_actions=8)

    # Run tournament
    print("=" * 60)
    print("EVOLUTIONARY TOURNAMENT")
    print("=" * 60)
    print(f"Population: {CONFIG['population_size']}")
    print(f"Generations: {CONFIG['generations']}")
    print(f"Interactions/agent: {CONFIG['interactions_per_agent']}")
    print(f"Rounds/interaction: {CONFIG['rounds_per_interaction']}")
    print("=" * 60)

    history, final_population = run_tournament(CONFIG)

    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print("\nTop 5 agents:")
    top5 = sorted(final_population, key=lambda g: g.fitness, reverse=True)[:5]
    for i, g in enumerate(top5, 1):
        print(
            f"  {i}. {g.book_name:8s} | tol={g.tolerance:.2f} | "
            f"sens={g.sensitivity:.2f} | fitness={g.fitness:.3f}"
        )

    print("\nBook distribution in final population:")
    books = [g.book_name for g in final_population]
    for book in OCTANT_ORDER + ["uniform"]:
        count = books.count(book)
        if count > 0:
            bar = "#" * count
            print(f"  {book:8s}: {bar} ({count})")

    print("\nEvolution summary:")
    print(f"  Initial mean fitness: {history[0]['mean_fitness']:.3f}")
    print(f"  Final mean fitness:   {history[-1]['mean_fitness']:.3f}")
    print(f"  Best ever fitness:    {max(h['max_fitness'] for h in history):.3f}")

    # Trait evolution
    print("\nTrait evolution:")
    print(
        f"  Tolerance:   {history[0]['mean_tolerance']:.2f} -> "
        f"{history[-1]['mean_tolerance']:.2f}"
    )
    print(
        f"  Sensitivity: {history[0]['mean_sensitivity']:.2f} -> "
        f"{history[-1]['mean_sensitivity']:.2f}"
    )
