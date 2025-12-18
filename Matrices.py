"""
Payoff matrices for different game scenarios.
Each scenario returns two 2x2 matrices (A1 for Agent1, A2 for Agent2).
Rows: Agent1 actions (Approach, Avoid)
Cols: Agent2 actions (Approach, Avoid)
"""

import numpy as np


def get_scenario(name):
    """
    Returns payoff matrices for the specified scenario.

    Scenarios:
    - "double": Two Nash equilibria (bistable - both approach OR both avoid)
    - "single": One Nash equilibrium (globally stable)
    - "circular": No pure Nash equilibrium (cyclical dynamics)
    """

    if name == "double":
        # Bistable: Both prefer matching behaviors
        # Nash at (Approach, Approach) and (Avoid, Avoid)
        A1 = np.array(
            [
                [3, 0],  # Approach: 3 if other approaches, 0 if other avoids
                [0, 2],  # Avoid: 0 if other approaches, 2 if other avoids
            ]
        )
        A2 = np.array(
            [
                [3, 0],  # Approach: 3 if other approaches, 0 if other avoids
                [0, 2],  # Avoid: 0 if other approaches, 2 if other avoids
            ]
        )

    elif name == "single":
        # Single Nash: One agent has dominant strategy
        # Nash at (Approach, Approach) only
        A1 = np.array([[3, 2], [1, 0]])  # Approach always better for A1
        A2 = np.array([[3, 0], [0, 1]])

    elif name == "circular":
        # No pure Nash: Matching pennies / RPS-like
        # A1 wants to match, A2 wants to mismatch
        A1 = np.array([[3, 0], [0, 3]])  # A1 prefers matching
        A2 = np.array([[0, 3], [3, 0]])  # A2 prefers mismatching

    else:
        raise ValueError(
            f"Unknown scenario: {name}. Choose from 'double', 'single', 'circular'"
        )

    return A1, A2


# Action labels for readability
ACTIONS = ["Approach", "Avoid"]
