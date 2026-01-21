"""
Q-Learning simulation for two agents in an n x n game.
"""

import numpy as np


def softmax_policy(Q, beta):
    scaled_Q = beta * Q  # high beta amplifies the differences between Q-values
    max_Q = np.max(scaled_Q)
    exp_Q = np.exp(scaled_Q - max_Q)
    return exp_Q / np.sum(exp_Q)


def get_current_book(
    book_name,
    n_actions,
    amplitude=1.0,
):
    if book_name == "uniform":
        current_book = np.full(n_actions, 1.0 / n_actions)
    else:
        peaks = {
            "LM": 0,
            "NO": 1,
            "PA": 2,
            "BC": 3,
            "DE": 4,
            "FG": 5,
            "HI": 6,
            "JK": 7,
        }
        if book_name in peaks:
            peak = peaks[book_name]
            x = np.arange(n_actions)
            phase_shift = np.pi / 2 - np.pi * peak / n_actions
            current_book = amplitude * np.sin(np.pi * x / n_actions + phase_shift) + 1
            current_book /= np.sum(current_book)
        else:
            # fallback: uniform
            current_book = np.full(n_actions, 1.0 / n_actions)
    return current_book


def mistake_counter(action_history, current_book):
    eps = 1e-12  # to avoid log(0)
    log_likelihoods = []
    for a in action_history:
        prob = current_book[a]
        log_likelihoods.append(np.log(prob + eps))
    # Negative average log-likelihood as a "mistake" or "surprise" parameter
    mistake_ratio = -np.mean(log_likelihoods)

    return mistake_ratio


def change_book(action_history, n_actions, amplitude, mistake_ratio, threshold=0.5):
    """
    If mistake_ratio > threshold, choose the book (from all possible) whose distribution
    best fits the observed action_history (lowest mistake_ratio).
    """
    if mistake_ratio > threshold:
        book_names = ["LM", "NO", "PA", "BC", "DE", "FG", "HI", "JK", "uniform"]
        best_book = None
        best_dist = None
        best_mistake = float("inf")
        for book_name in book_names:
            candidate_dist = get_current_book(book_name, n_actions, amplitude)
            candidate_mistake = mistake_counter(action_history, candidate_dist)
            if candidate_mistake < best_mistake:
                best_mistake = candidate_mistake
                best_book = book_name
                best_dist = candidate_dist
        return best_book, best_dist
    else:
        return None, None


def update_current_book(current_book, action_history, exclude_n=0, prior_weight=0.5):
    """
    Update the current_book (prior) with the empirical distribution (likelihood) from action_history,
    excluding the last `exclude_n` actions. Returns the posterior as a convex combination.
    """
    n_actions = len(current_book)
    # Exclude the last `exclude_n` actions if needed
    if exclude_n > 0:
        relevant_history = action_history[:-exclude_n]
    else:
        relevant_history = action_history
    # Compute empirical distribution (likelihood)
    empirical = np.zeros(n_actions)
    for a in relevant_history:
        empirical[a] += 1
    total = np.sum(empirical)
    if total > 0:
        empirical /= total
    else:
        empirical = np.full(n_actions, 1.0 / n_actions)
    # Posterior: weighted average of prior and empirical
    posterior = prior_weight * current_book + (1 - prior_weight) * empirical
    posterior /= np.sum(posterior)  # ensure normalization
    current_book = posterior
    return posterior


class Agent:
    """An agent with beliefs about self (book) and adaptive payoffs (motives)."""

    def __init__(
        self, n_actions, payoff_matrix,
        book_name="uniform", amplitude=1.0, tolerance=1.0
    ):
        self.n_actions = n_actions
        self.base_payoff = payoff_matrix.copy()
        self.payoff = payoff_matrix.copy()
        self.Q = np.zeros(n_actions)
        self.book = get_current_book(book_name, n_actions, amplitude)
        self.book_name = book_name
        self.amplitude = amplitude
        self.tolerance = tolerance
        self.action_history = []
        self.other_history = []
        self.motive_idx = 0
        self.mismatch = 0.0
        self.active = True  # False when relationship aborted

    def act(self, beta):
        """Choose action via softmax policy."""
        probs = softmax_policy(self.Q, beta)
        action = np.random.choice(self.n_actions, p=probs)
        self.action_history.append(action)
        return action, probs

    def observe(self, own_action, other_action, payoff):
        """Record other's action and update Q-value."""
        self.other_history.append(other_action)

    def learn(self, action, payoff, alpha):
        """Q-learning update."""
        self.Q[action] += alpha * (payoff - self.Q[action])

    def update_self_model(self, threshold=0.5, prior_weight=0.5):
        """Update book based on own behavior - how we see ourselves."""
        if len(self.action_history) < 2:
            return
        mistake = mistake_counter(self.action_history, self.book)
        new_name, new_book = change_book(
            self.action_history, self.n_actions, self.amplitude, mistake, threshold
        )
        if new_book is not None:
            self.book = new_book
            self.book_name = new_name
        else:
            self.book = update_current_book(self.book, self.action_history, prior_weight=prior_weight)

    def update_motives(self, motive_matrices, sensitivity=1.0):
        """
        Update payoff matrix based on mismatch with self-concept.

        If other deviates from my self-concept, my need for that
        behavior increases (compensatory). If mismatch exceeds
        tolerance, abort relationship.
        """
        if len(self.other_history) < 2 or motive_matrices is None:
            return
        from motive_system import select_motive_matrix, check_relationship_viable
        self.payoff, self.motive_idx, self.mismatch = select_motive_matrix(
            self.book, self.other_history, motive_matrices, sensitivity
        )
        self.active = check_relationship_viable(self.mismatch, self.tolerance)


def q_as_if_new_book(Q, current_book, payoff_matrix):
    """
    Update Q-values as if the agent were to act according to the new current_book distribution.
    For each action, set Q[a] to the expected payoff against the current_book distribution.
    """
    n_actions = len(current_book)
    new_Q = np.zeros_like(Q)
    # For each action, compute expected payoff against the new distribution
    for a in range(n_actions):
        # Expected payoff for action a against all possible opponent actions
        expected_payoff = np.dot(payoff_matrix[a], current_book)
        new_Q[a] = expected_payoff
    return new_Q


def motivation_system(action_history_other, value_distribution, motive_matrices):
    """
    Compare observed action behavior with value_distribution.
    Larger deviations yield stronger motives, which select/update motive_matrices.
    Returns the selected motive matrix.
    """
    n_actions = len(value_distribution)
    # Compute empirical distribution from observed actions
    empirical = np.zeros(n_actions)
    for a in action_history_other:
        empirical[a] += 1
    total = np.sum(empirical)
    if total > 0:
        empirical /= total
    else:
        empirical = np.full(n_actions, 1.0 / n_actions)
    # Compute deviation (e.g., L2 norm)
    deviation_strength = np.linalg.norm(empirical - value_distribution)
    # Map deviation_strength to a motive index (e.g., discretize or threshold)
    # Example: pick the motive with the closest threshold
    motive_idx = np.argmin(
        [abs(deviation_strength - m["threshold"]) for m in motive_matrices]
    )
    selected_motive_matrix = motive_matrices[motive_idx]["matrix"]
    return selected_motive_matrix


def simulation(
    P1, P2, l_rate, e_rate, rounds,
    book1="uniform", book2="uniform",
    tolerance1=1.0, tolerance2=1.0,
    motive_matrices1=None, motive_matrices2=None,
    update_interval=10, sensitivity=1.0
):
    """
    Two agents optimizing behavior to each other.

    Each agent:
    - Learns Q-values from payoffs (behavior optimization)
    - Updates self-model (book) based on own actions (self-perception)
    - Updates motives based on mismatch (compensatory needs)
    - Aborts relationship if mismatch exceeds their tolerance
    """
    n_actions = P1.shape[0]

    agent1 = Agent(n_actions, P1, book_name=book1, tolerance=tolerance1)
    agent2 = Agent(n_actions, P2, book_name=book2, tolerance=tolerance2)

    # History storage
    p_history = np.zeros((rounds, n_actions))
    q_history = np.zeros((rounds, n_actions))
    action_history1 = np.zeros(rounds, dtype=int)
    action_history2 = np.zeros(rounds, dtype=int)
    payoff_history1 = np.zeros(rounds)
    payoff_history2 = np.zeros(rounds)
    book_history1 = []
    book_history2 = []
    motive_history1 = []
    motive_history2 = []
    abort_round = None

    alpha = l_rate
    beta = e_rate

    for t in range(rounds):
        # Check if either agent aborted
        if not agent1.active or not agent2.active:
            abort_round = t
            break

        # Act
        action1, probs1 = agent1.act(beta)
        action2, probs2 = agent2.act(beta)

        # Store
        p_history[t] = probs1
        q_history[t] = probs2
        action_history1[t] = action1
        action_history2[t] = action2

        # Payoffs (using current, possibly motive-updated matrices)
        payoff1 = agent1.payoff[action1, action2]
        payoff2 = agent2.payoff[action2, action1]
        payoff_history1[t] = payoff1
        payoff_history2[t] = payoff2

        # Observe and learn
        agent1.observe(action1, action2, payoff1)
        agent2.observe(action2, action1, payoff2)
        agent1.learn(action1, payoff1, alpha)
        agent2.learn(action2, payoff2, alpha)

        # Periodically update self-models and motives
        if t > 0 and t % update_interval == 0:
            agent1.update_self_model()
            agent2.update_self_model()
            agent1.update_motives(motive_matrices1, sensitivity)
            agent2.update_motives(motive_matrices2, sensitivity)
            book_history1.append((t, agent1.book_name))
            book_history2.append((t, agent2.book_name))
            motive_history1.append((t, agent1.motive_idx, agent1.mismatch))
            motive_history2.append((t, agent2.motive_idx, agent2.mismatch))

    return {
        "p_history": p_history,
        "q_history": q_history,
        "action_history1": action_history1,
        "action_history2": action_history2,
        "payoff_history1": payoff_history1,
        "payoff_history2": payoff_history2,
        "book_history1": book_history1,
        "book_history2": book_history2,
        "motive_history1": motive_history1,
        "motive_history2": motive_history2,
        "abort_round": abort_round,
        "agent1": agent1,
        "agent2": agent2,
    }
