"""
Enhanced Q-Learning simulation with individual learning parameters and ambiguous situations.

New features:
- Individual alpha (learning rate) and beta (temperature) for each agent
- Ambiguous situations every X steps where agents can't observe the other's action
- Trust scores (0-1) that determine how agents interpret ambiguous situations
  - High trust: interpret ambiguously (assume friendly/Approach)
  - Low trust: interpret coldly (assume hostile/Avoid)
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


def interpret_ambiguous_action(trust_score):
    """
    Interpret the other agent's action in an ambiguous situation based on trust.

    Args:
        trust_score: Float between 0 and 1
                    - 1.0 = full trust, always assume Approach (warm interpretation)
                    - 0.0 = no trust, always assume Avoid (cold interpretation)
                    - 0.5 = neutral, 50/50 chance

    Returns:
        Interpreted action: 0 (Approach) or 1 (Avoid)
    """
    # Sample from trust probability: higher trust = more likely to assume Approach
    return 0 if np.random.random() < trust_score else 1


def simulate_ambiguous(
    A1,
    A2,
    p_init,
    q_init,
    alpha1,
    beta1,
    alpha2,
    beta2,
    trust1,
    trust2,
    ambiguous_freq,
    rounds,
):
    """
    Simulate repeated interactions with individual parameters and ambiguous situations.

    Args:
        A1, A2: Payoff matrices for Agent1 and Agent2
        p_init, q_init: Initial probabilities from Nash calculation
        alpha1, alpha2: Individual learning rates (0 to 1)
        beta1, beta2: Individual softmax temperatures (higher = more exploitation)
        trust1, trust2: Trust scores (0 to 1) for interpreting ambiguous situations
        ambiguous_freq: How often ambiguous situations occur (every X steps)
        rounds: Number of rounds

    Returns:
        dict with histories of probabilities, actions, payoffs, and ambiguous rounds
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
    ambiguous_rounds = []  # Track which rounds were ambiguous

    # Set initial values
    p_history[0] = p_init
    q_history[0] = q_init

    for t in range(1, rounds):
        # Calculate action probabilities using individual betas
        p_approach = softmax_policy(Q1, beta1)
        q_approach = softmax_policy(Q2, beta2)

        # Store probabilities
        p_history[t] = p_approach
        q_history[t] = q_approach

        # Choose actions
        action1 = 0 if np.random.random() < p_approach else 1
        action2 = 0 if np.random.random() < q_approach else 1

        action_history1[t] = action1
        action_history2[t] = action2

        # Check if this is an ambiguous situation
        is_ambiguous = (ambiguous_freq > 0) and (t % ambiguous_freq == 0)

        if is_ambiguous:
            ambiguous_rounds.append(t)

            # Agents can't observe each other's actual actions
            # They interpret based on their trust scores
            perceived_action2 = interpret_ambiguous_action(
                trust1
            )  # A1's interpretation of A2
            perceived_action1 = interpret_ambiguous_action(
                trust2
            )  # A2's interpretation of A1

            # Get payoffs based on perceived actions (not actual actions!)
            payoff1 = A1[action1, perceived_action2]
            payoff2 = A2[perceived_action1, action2]
        else:
            # Normal situation: agents observe actual actions
            payoff1 = A1[action1, action2]
            payoff2 = A2[action1, action2]

        payoff_history1[t] = payoff1
        payoff_history2[t] = payoff2

        # Q-learning update with individual learning rates
        Q1[action1] = Q1[action1] + alpha1 * (payoff1 - Q1[action1])
        Q2[action2] = Q2[action2] + alpha2 * (payoff2 - Q2[action2])

    return {
        "p_history": p_history,
        "q_history": q_history,
        "action_history1": action_history1,
        "action_history2": action_history2,
        "payoff_history1": payoff_history1,
        "payoff_history2": payoff_history2,
        "ambiguous_rounds": ambiguous_rounds,
        "trust1": trust1,
        "trust2": trust2,
    }
