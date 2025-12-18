"""
Nash equilibrium calculation for 2x2 games.
"""

import numpy as np


def solve_nash(A1, A2):
    """
    Solve for Nash equilibrium in a 2x2 game.

    Returns:
        p_init: probability of Agent1 playing Approach
        q_init: probability of Agent2 playing Approach
        equilibria: list of pure strategy Nash equilibria found
    """

    # Check for pure strategy Nash equilibria
    pure_nash = []

    # Check all four cells
    for i in range(2):  # Agent1's action
        for j in range(2):  # Agent2's action
            # Is this a Nash equilibrium?
            # Agent1 best response given j
            a1_payoffs_given_j = A1[:, j]
            a1_best = np.argmax(a1_payoffs_given_j)

            # Agent2 best response given i
            a2_payoffs_given_i = A2[i, :]
            a2_best = np.argmax(a2_payoffs_given_i)

            if a1_best == i and a2_best == j:
                pure_nash.append((i, j))

    # Calculate mixed strategy equilibrium
    # For Agent2 to make Agent1 indifferent:
    # A1[0,0]*q + A1[0,1]*(1-q) = A1[1,0]*q + A1[1,1]*(1-q)
    a, b = A1[0, 0], A1[0, 1]
    c, d = A1[1, 0], A1[1, 1]

    denom1 = a - b - c + d
    if denom1 != 0:
        q_star = (d - b) / denom1
    else:
        q_star = None

    # For Agent1 to make Agent2 indifferent:
    e, f = A2[0, 0], A2[0, 1]
    g, h = A2[1, 0], A2[1, 1]

    denom2 = e - g - f + h
    if denom2 != 0:
        p_star = (h - g) / denom2
    else:
        p_star = None

    # Determine initial probabilities
    if len(pure_nash) == 1:
        # Single pure Nash - use it
        i, j = pure_nash[0]
        p_init = 1.0 if i == 0 else 0.0
        q_init = 1.0 if j == 0 else 0.0
    elif len(pure_nash) == 2:
        # Two pure Nash - use mixed equilibrium if valid
        if p_star is not None and 0 <= p_star <= 1:
            p_init = p_star
        else:
            p_init = 0.5
        if q_star is not None and 0 <= q_star <= 1:
            q_init = q_star
        else:
            q_init = 0.5
    else:
        # No pure Nash - use mixed if valid, else 0.5
        if p_star is not None and 0 <= p_star <= 1:
            p_init = p_star
        else:
            p_init = 0.5
        if q_star is not None and 0 <= q_star <= 1:
            q_init = q_star
        else:
            q_init = 0.5

    return p_init, q_init, pure_nash


def print_game_info(A1, A2, scenario_name):
    """Print game matrices and Nash equilibria."""
    print(f"\n=== Scenario: {scenario_name} ===")
    print("\nPayoff Matrix (Agent1, Agent2):")
    print("              Agent2")
    print("              Approach  Avoid")
    print(f"Agent1 Approach  ({A1[0,0]},{A2[0,0]})    ({A1[0,1]},{A2[0,1]})")
    print(f"       Avoid     ({A1[1,0]},{A2[1,0]})    ({A1[1,1]},{A2[1,1]})")

    p_init, q_init, pure_nash = solve_nash(A1, A2)

    print(f"\nPure Nash Equilibria: {pure_nash}")
    print(
        f"Initial probabilities: p(A1=Approach)={p_init:.3f}, p(A2=Approach)={q_init:.3f}"
    )

    return p_init, q_init
