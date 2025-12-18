"""
Main interface for the Enhanced Affiliative Interpersonality Simulation with Ambiguous Situations.

Based on: Westermann & Banisch (2025) - A Formal Model of Affiliative Interpersonality

Enhanced features:
- Individual learning rates (alpha) and temperatures (beta) for each agent
- Ambiguous situations where agents can't observe each other's actions
- Trust scores that determine how agents interpret ambiguous situations
"""

from Matrices import get_scenario
from Nash_Calc import print_game_info
from AmbiguousQLearning import simulate_ambiguous
from Plotting import plot_probabilities

# =============================================================================
# PARAMETERS - Modify these to control the simulation
# =============================================================================

# Individual learning parameters for Agent 1
ALPHA1 = 0.8  # Learning rate for Agent1 (0 to 1) - how quickly Agent1 learns
BETA1 = 4.0  # Temperature for Agent1 (higher = more exploitation)

# Individual learning parameters for Agent 2
ALPHA2 = 0.8  # Learning rate for Agent2 (0 to 1) - how quickly Agent2 learns
BETA2 = 4.0  # Temperature for Agent2 (higher = more exploitation)

# Trust parameters (0 to 1)
# Trust determines how agents interpret ambiguous situations:
#   1.0 = Full trust - always assume the other is being friendly (Approach)
#   0.5 = Neutral - 50/50 interpretation
#   0.0 = No trust - always assume the other is being hostile (Avoid)
TRUST1 = 1  # Agent1's trust in Agent2
TRUST2 = 0  # Agent2's trust in Agent1

# Ambiguous situation frequency
# Every X rounds, an ambiguous situation occurs where agents can't see each other's actions
# Set to 0 to disable ambiguous situations
AMBIGUOUS_FREQ = 10  # Ambiguous situation every X rounds (0 = disabled)

# Simulation parameters
ROUNDS = 500  # Number of interaction rounds

# Scenario selection
# Options: "double", "single", "circular"
#   - "double": Two Nash equilibria (bistable - converges to one of two attractors)
#   - "single": One Nash equilibrium (globally stable - always converges to same point)
#   - "circular": No pure Nash (cyclical dynamics - oscillates)
SCENARIO = "double"

# Output
SAVE_PLOT = True  # Whether to save the plot
PLOT_PATH = "simulation_ambiguous_result.png"

# =============================================================================
# RUN SIMULATION
# =============================================================================

if __name__ == "__main__":
    # Get payoff matrices for the scenario
    A1, A2 = get_scenario(SCENARIO)

    # Calculate Nash equilibrium and print game info
    p_init, q_init = print_game_info(A1, A2, SCENARIO)

    print(f"\n{'='*60}")
    print("ENHANCED SIMULATION WITH AMBIGUOUS SITUATIONS")
    print(f"{'='*60}")

    print(f"\nAgent 1 Parameters:")
    print(f"  Alpha (learning rate): {ALPHA1}")
    print(f"  Beta (temperature):    {BETA1}")
    print(f"  Trust score:           {TRUST1:.2f}")

    print(f"\nAgent 2 Parameters:")
    print(f"  Alpha (learning rate): {ALPHA2}")
    print(f"  Beta (temperature):    {BETA2}")
    print(f"  Trust score:           {TRUST2:.2f}")

    print(f"\nAmbiguous Situations:")
    if AMBIGUOUS_FREQ > 0:
        print(f"  Frequency:             Every {AMBIGUOUS_FREQ} rounds")
        print(f"  Total expected:        ~{ROUNDS // AMBIGUOUS_FREQ} times")
    else:
        print(f"  Disabled (AMBIGUOUS_FREQ = 0)")

    print(f"\nSimulation:")
    print(f"  Rounds:                {ROUNDS}")

    # Run enhanced simulation
    print("\nRunning enhanced simulation...")
    result = simulate_ambiguous(
        A1,
        A2,
        p_init,
        q_init,
        ALPHA1,
        BETA1,
        ALPHA2,
        BETA2,
        TRUST1,
        TRUST2,
        AMBIGUOUS_FREQ,
        ROUNDS,
    )

    # Print ambiguous situation info
    if result["ambiguous_rounds"]:
        print(
            f"\nAmbiguous situations occurred at rounds: {result['ambiguous_rounds'][:10]}{'...' if len(result['ambiguous_rounds']) > 10 else ''}"
        )
        print(f"Total ambiguous situations: {len(result['ambiguous_rounds'])}")

    # Plot results (uses same plotting function as original)
    save_path = PLOT_PATH if SAVE_PLOT else None
    plot_probabilities(
        result["p_history"], result["q_history"], f"{SCENARIO} (Ambiguous)", save_path
    )

    print(f"\n{'='*60}")
    print("Simulation complete!")
    print(f"{'='*60}")
