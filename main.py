"""
Main interface for the Affiliative Interpersonality Simulation.

Based on: Westermann & Banisch (2025) - A Formal Model of Affiliative Interpersonality

This simulates two agents learning to interact using Q-learning,
where each agent can choose to Approach (friendly) or Avoid (distance).
"""

from Matrices import get_scenario
from Nash_Calc import solve_nash, print_game_info
from QLearning import simulate
from Plotting import plot_probabilities

# =============================================================================
# PARAMETERS - Modify these to control the simulation
# =============================================================================

# Learningparameters
ALPHA = 0.5  # Learning rate (0 to 1) (0.1 for normali-ish, deterministic)
BETA = 1  # Softmax temperature (higher = more exploitation) (3 more normal-ish, deterministic)

# Simulation parameters
ROUNDS = 200  # Number of interaction rounds

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
    # if you want to skip the nash calc
    p_init = 0.5
    q_init = 0.5

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
