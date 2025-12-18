"""
Q-Learning simulation for two agents in a 2x2 game.
"""

import numpy as np


def softmax_policy(Q, beta):
    """
    Softmax action selection.
    Returns probability of choosing action 0 (Approach).
    """
    exp_Q = np.exp(beta * Q)
    return exp_Q[0] / np.sum(exp_Q)


def simulate(A1, A2, p_init, q_init, alpha, beta, rounds):
    """
    Simulate repeated interactions using Q-learning.

    Args:
        A1, A2: Payoff matrices
        p_init, q_init: Initial probabilities from Nash calculation
        alpha: Learning rate
        beta: Softmax temperature (higher = more exploitation)
        rounds: Number of rounds

    Returns:
        dict with histories of probabilities, actions, payoffs
    """

    # Initialize Q-values based on Nash equilibrium probabilities
    Q1 = np.array([p_init, 1 - p_init])
    Q2 = np.array([q_init, 1 - q_init])

    # History storage
    p_history = np.zeros(rounds)
    q_history = np.zeros(rounds)
    action_history1 = np.zeros(rounds, dtype=int)
    action_history2 = np.zeros(rounds, dtype=int)
    payoff_history1 = np.zeros(rounds)
    payoff_history2 = np.zeros(rounds)

    # Set initial values
    p_history[0] = p_init
    q_history[0] = q_init

    for t in range(1, rounds):
        # Calculate action probabilities using softmax
        p_approach = softmax_policy(Q1, beta)
        q_approach = softmax_policy(Q2, beta)

        # Store probabilities
        p_history[t] = p_approach
        q_history[t] = q_approach

        # Choose actions
        action1 = 0 if np.random.random() < p_approach else 1
        action2 = 0 if np.random.random() < q_approach else 1

        action_history1[t] = action1
        action_history2[t] = action2

        # Get payoffs
        payoff1 = A1[action1, action2]
        payoff2 = A2[action1, action2]

        payoff_history1[t] = payoff1
        payoff_history2[t] = payoff2

        # Q-learning update
        Q1[action1] = Q1[action1] + alpha * (payoff1 - Q1[action1])
        Q2[action2] = Q2[action2] + alpha * (payoff2 - Q2[action2])

    return {
        "p_history": p_history,
        "q_history": q_history,
        "action_history1": action_history1,
        "action_history2": action_history2,
        "payoff_history1": payoff_history1,
        "payoff_history2": payoff_history2,
    }
