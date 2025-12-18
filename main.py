"""
Main interface for the Affiliative Interpersonality Simulation.

Based on: Westermann & Banisch (2025) - A Formal Model of Affiliative Interpersonality

This simulates two agents learning to interact using Q-learning,
where each agent can choose to Approach (friendly) or Avoid (distance).
"""

from matrices import get_scenario
from nash_calc import solve_nash, print_game_info
from q_learning import simulate
from plotting import plot_probabilities

# =============================================================================
# PARAMETERS - Modify these to control the simulation
# =============================================================================

# Learning parameters
ALPHA = 0.1  # Learning rate (0 to 1)
BETA = 3.0  # Softmax temperature (higher = more exploitation)

# Simulation parameters
ROUNDS = 300  # Number of interaction rounds

# Scenario selection
# Options: "double", "single", "circular"
#   - "double": Two Nash equilibria (bistable - converges to one of two attractors)
#   - "single": One Nash equilibrium (globally stable - always converges to same point)
#   - "circular": No pure Nash (cyclical dynamics - oscillates)
SCENARIO = "double"

# Output
SAVE_PLOT = True  # Whether to save the plot
PLOT_PATH = "simulation_result.png"

# =============================================================================
# RUN SIMULATION
# =============================================================================

if __name__ == "__main__":
    # Get payoff matrices for the scenario
    A1, A2 = get_scenario(SCENARIO)

    # Calculate Nash equilibrium and print game info
    p_init, q_init = print_game_info(A1, A2, SCENARIO)

    print(f"\nSimulation Parameters:")
    print(f"  Alpha (learning rate): {ALPHA}")
    print(f"  Beta (temperature):    {BETA}")
    print(f"  Rounds:                {ROUNDS}")

    # Run simulation
    print("\nRunning simulation...")
    result = simulate(A1, A2, p_init, q_init, ALPHA, BETA, ROUNDS)

    # Plot results
    save_path = PLOT_PATH if SAVE_PLOT else None
    plot_probabilities(result["p_history"], result["q_history"], SCENARIO, save_path)
