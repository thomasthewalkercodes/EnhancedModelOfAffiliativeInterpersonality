"""
Run multiple simulations and generate CSV summary with statistics.

This script runs the ambiguous simulation multiple times and calculates:
- Mean points collected for each agent
- Standard deviation of points for each agent
- Results are saved to a CSV file
"""

import numpy as np
import pandas as pd
from Matrices import get_scenario
from Nash_Calc import print_game_info
from AmbiguousQLearning import simulate_ambiguous

# =============================================================================
# PARAMETERS - Modify these to control the simulation
# =============================================================================

# Number of simulations to run
NUM_SIMULATIONS = 100

# Individual learning parameters for Agent 1
ALPHA1 = 0.8  # Learning rate for Agent1 (0 to 1)
BETA1 = 5  # Temperature for Agent1 (higher = more exploitation)

# Individual learning parameters for Agent 2
ALPHA2 = 0.1  # Learning rate for Agent2 (0 to 1)
BETA2 = 3  # Temperature for Agent2 (higher = more exploitation)

# Trust parameters (0 to 1)
TRUST1 = 0  # Agent1's trust in Agent2
TRUST2 = 0.8  # Agent2's trust in Agent1

# Ambiguous situation frequency
AMBIGUOUS_FREQ = 10  # Ambiguous situation every X rounds (0 = disabled)

# Simulation parameters
ROUNDS = 900  # Number of interaction rounds

# Scenario selection
SCENARIO = "double"  # Options: "double", "single", "circular"

# Output CSV file
OUTPUT_CSV = "simulation_summary.csv"

# =============================================================================
# RUN MULTIPLE SIMULATIONS
# =============================================================================

if __name__ == "__main__":
    print(f"{'='*60}")
    print(f"RUNNING {NUM_SIMULATIONS} SIMULATIONS")
    print(f"{'='*60}")

    # Get payoff matrices for the scenario
    A1, A2 = get_scenario(SCENARIO)

    # Calculate Nash equilibrium (suppress output)
    print("\nCalculating Nash equilibrium...")
    p_init, q_init = print_game_info(A1, A2, SCENARIO)
    # Alternative: skip Nash calculation and use default
    # p_init = 0.5
    # q_init = 0.5

    print(f"\nSimulation Parameters:")
    print(f"  Scenario: {SCENARIO}")
    print(f"  Rounds per simulation: {ROUNDS}")
    print(f"  Agent 1: alpha={ALPHA1}, beta={BETA1}, trust={TRUST1}")
    print(f"  Agent 2: alpha={ALPHA2}, beta={BETA2}, trust={TRUST2}")
    print(f"  Ambiguous frequency: {AMBIGUOUS_FREQ}")

    # Storage for all simulation results
    total_points_agent1 = []
    total_points_agent2 = []

    # Run simulations
    print(f"\nRunning {NUM_SIMULATIONS} simulations...")
    for sim in range(NUM_SIMULATIONS):
        if (sim + 1) % 10 == 0:
            print(f"  Completed {sim + 1}/{NUM_SIMULATIONS} simulations...")

        # Run simulation
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

        # Calculate total points for each agent
        total_agent1 = np.sum(result["payoff_history1"])
        total_agent2 = np.sum(result["payoff_history2"])

        total_points_agent1.append(total_agent1)
        total_points_agent2.append(total_agent2)

    print(f"  Completed {NUM_SIMULATIONS}/{NUM_SIMULATIONS} simulations!")

    # Calculate statistics
    mean_agent1 = np.mean(total_points_agent1)
    std_agent1 = np.std(total_points_agent1)
    mean_agent2 = np.mean(total_points_agent2)
    std_agent2 = np.std(total_points_agent2)

    # Create summary dataframe
    summary_data = {
        "Agent": ["Agent 1", "Agent 2"],
        "Mean_Points": [mean_agent1, mean_agent2],
        "Std_Deviation": [std_agent1, std_agent2],
        "Min_Points": [np.min(total_points_agent1), np.min(total_points_agent2)],
        "Max_Points": [np.max(total_points_agent1), np.max(total_points_agent2)],
        "Alpha": [ALPHA1, ALPHA2],
        "Beta": [BETA1, BETA2],
        "Trust": [TRUST1, TRUST2],
    }

    summary_df = pd.DataFrame(summary_data)

    # Save to CSV
    summary_df.to_csv(OUTPUT_CSV, index=False)

    # Print results
    print(f"\n{'='*60}")
    print("SIMULATION RESULTS")
    print(f"{'='*60}")
    print(f"\nAgent 1:")
    print(f"  Mean points: {mean_agent1:.2f}")
    print(f"  Std deviation: {std_agent1:.2f}")
    print(f"  Min: {np.min(total_points_agent1):.2f}")
    print(f"  Max: {np.max(total_points_agent1):.2f}")

    print(f"\nAgent 2:")
    print(f"  Mean points: {mean_agent2:.2f}")
    print(f"  Std deviation: {std_agent2:.2f}")
    print(f"  Min: {np.min(total_points_agent2):.2f}")
    print(f"  Max: {np.max(total_points_agent2):.2f}")

    print(f"\nResults saved to: {OUTPUT_CSV}")
    print(f"\n{'='*60}")
