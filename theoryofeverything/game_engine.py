"""
Q-Learning simulation for two agents in an n x n game.
"""

import numpy as np


def softmax_policy(Q, beta):
    """
    Softmax action selection.
    Returns probability distribution over all actions.
    """
    # Numerical stability: subtract max before exp to avoid overflow
    scaled_Q = beta * Q
    max_Q = np.max(scaled_Q)
    exp_Q = np.exp(scaled_Q - max_Q)
    return exp_Q / np.sum(exp_Q)


def simulation(P1, P2, l_rate, e_rate, rounds):

    n_actions = P1.shape[0]  # Number of actions

    # Initialize Q-values to 0 (standard Q-learning initialization)
    Q1 = np.zeros(n_actions)
    Q2 = np.zeros(n_actions)

    # But if p_init and q_init are provided as arrays or something, use them
    # For simplicity, assume they are scalars and distribute

    # History storage
    p_history = np.zeros((rounds, n_actions))
    q_history = np.zeros((rounds, n_actions))
    action_history1 = np.zeros(rounds, dtype=int)
    action_history2 = np.zeros(rounds, dtype=int)
    payoff_history1 = np.zeros(rounds)
    payoff_history2 = np.zeros(rounds)

    # Set initial values - uniform for all actions
    p_history[0] = np.full(n_actions, 1.0 / n_actions)
    q_history[0] = np.full(n_actions, 1.0 / n_actions)

    alpha = l_rate  # learning rate
    beta = e_rate  # temperature

    for t in range(1, rounds):
        # Calculate action probabilities using softmax
        p_probs = softmax_policy(Q1, beta)
        q_probs = softmax_policy(Q2, beta)

        # Store probabilities
        p_history[t] = p_probs
        q_history[t] = q_probs

        # Choose actions based on probabilities
        action1 = np.random.choice(n_actions, p=p_probs)
        action2 = np.random.choice(n_actions, p=q_probs)

        action_history1[t] = action1
        action_history2[t] = action2

        # Get payoffs
        payoff1 = P1[action1, action2]
        payoff2 = P2[action1, action2]

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
