"""
Main interface for Genetic Evolution of Agent Parameters.

This system evolves agent parameters (alpha, beta, trust) over multiple generations
to find optimal strategies against a fixed opponent.

Evolution Process:
1. Initial generation has diverse random parameters
2. Each agent plays against the same fixed opponent
3. Agents accumulate fitness (total points)
4. Top performers are selected for reproduction
5. Offspring are created with slight mutations
6. Repeat for N generations

Results are saved to CSV files showing parameter evolution and best performers.
"""

from Matrices import get_scenario
from Nash_Calc import print_game_info
from GeneticEvolution import GeneticEvolution

# =============================================================================
# EVOLUTIONARY PARAMETERS - Modify these to control evolution
# =============================================================================

# Population settings
POPULATION_SIZE = 50  # Number of agents per generation (larger = more diverse)
NUM_GENERATIONS = 20  # How many generations to evolve (more = better convergence)

# Selection and mutation
SELECTION_RATE = 0.2  # Top 20% survive to next generation (0.1 to 0.5 typical)
MUTATION_RATE = 0.1  # How much parameters mutate (0.05 = small, 0.2 = large)

# Match settings
ROUNDS_PER_MATCH = (
    200  # How many rounds each agent plays (more = more accurate fitness)
)
AMBIGUOUS_FREQ = 10  # Frequency of ambiguous situations (0 = disabled)

# =============================================================================
# OPPONENT AGENT PARAMETERS - The fixed agent all evolving agents play against
# =============================================================================

# You can modify these to test evolution against different opponent strategies
OPPONENT_ALPHA = 0.1  # Opponent's learning rate
OPPONENT_BETA = 3.0  # Opponent's temperature
OPPONENT_TRUST = 0.8  # Opponent's trust score

# =============================================================================
# GAME SCENARIO
# =============================================================================

# Scenario selection
# Options: "double", "single", "circular"
#   - "double": Two Nash equilibria (bistable)
#   - "single": One Nash equilibrium (globally stable)
#   - "circular": No pure Nash (cyclical dynamics)
SCENARIO = "double"

# =============================================================================
# OUTPUT SETTINGS
# =============================================================================

# Results folder name - change this to organize different evolution runs
RESULTS_FOLDER = "evolution_results"

# =============================================================================
# RUN EVOLUTION
# =============================================================================

if __name__ == "__main__":
    # Get payoff matrices for the scenario
    A1, A2 = get_scenario(SCENARIO)

    # Calculate Nash equilibrium and print game info
    p_init, q_init = print_game_info(A1, A2, SCENARIO)

    # Create genetic evolution system
    evolution = GeneticEvolution(
        population_size=POPULATION_SIZE,
        num_generations=NUM_GENERATIONS,
        selection_rate=SELECTION_RATE,
        mutation_rate=MUTATION_RATE,
        rounds_per_match=ROUNDS_PER_MATCH,
        ambiguous_freq=AMBIGUOUS_FREQ,
        A1=A1,
        A2=A2,
        p_init=p_init,
        q_init=q_init,
        opponent_alpha=OPPONENT_ALPHA,
        opponent_beta=OPPONENT_BETA,
        opponent_trust=OPPONENT_TRUST,
        results_folder=RESULTS_FOLDER,
    )

    # Run evolution
    evolution.run_evolution()

    print(f"\n{'='*70}")
    print("NEXT STEPS:")
    print(f"{'='*70}")
    print(
        f"1. Check '{RESULTS_FOLDER}/top_performers.csv' for best parameter combinations"
    )
    print(
        f"2. Review '{RESULTS_FOLDER}/generation_stats.csv' to see evolution progress"
    )
    print(f"3. Analyze '{RESULTS_FOLDER}/all_agents.csv' for complete agent history")
    print(f"{'='*70}\n")
