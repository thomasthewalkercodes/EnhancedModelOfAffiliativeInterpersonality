"""
Motive system for interpersonal dynamics.

Core idea: My self-concept (book) defines what behavior I need from others.
When others deviate from my self-concept, my need for that behavior increases (compensatory).
If deviation exceeds my tolerance, I abort the relationship.
"""

import numpy as np

OCTANT_ORDER = ["LM", "NO", "PA", "BC", "DE", "FG", "HI", "JK"]


def compute_mismatch(self_book, other_distribution):
    """
    Compute how much other's behavior deviates from my self-concept.

    Returns:
        mismatch: scalar (0 = perfect match, higher = more deviation)
        deviation_vector: signed difference per action (positive = other does less than I need)
    """
    deviation_vector = self_book - other_distribution
    mismatch = np.linalg.norm(deviation_vector)
    return mismatch, deviation_vector


def compute_need_increase(deviation_vector, sensitivity=1.0):
    """
    Compensatory: where other falls short, my need increases.

    deviation_vector[i] > 0 means other does action i less than my self-concept expects.
    This increases my need for action i from them.
    """
    need_increase = np.clip(deviation_vector * sensitivity, 0, None)
    return need_increase


def select_motive_matrix(self_book, other_history, motive_matrices, sensitivity=1.0):
    """
    Select motive matrix based on mismatch between self-concept and other's behavior.

    Args:
        self_book: My self-concept distribution (what I need from others)
        other_history: List of other's actions
        motive_matrices: List of n×n payoff matrices (one per octant)
        sensitivity: How reactive to deviations

    Returns:
        selected_matrix, motive_index, mismatch_score
    """
    n_actions = len(self_book)

    # Compute empirical distribution of other's behavior
    other_dist = np.zeros(n_actions)
    for a in other_history:
        other_dist[a] += 1
    if np.sum(other_dist) > 0:
        other_dist /= np.sum(other_dist)
    else:
        other_dist = np.full(n_actions, 1.0 / n_actions)

    # Compute mismatch
    mismatch, deviation_vector = compute_mismatch(self_book, other_dist)

    # Find which action I need most (largest positive deviation)
    need_increase = compute_need_increase(deviation_vector, sensitivity)
    motive_idx = np.argmax(need_increase) if np.max(need_increase) > 0 else np.argmax(self_book)

    return motive_matrices[motive_idx], motive_idx, mismatch


def check_relationship_viable(mismatch, tolerance):
    """
    Check if relationship should continue.

    Returns:
        True if relationship is viable, False if should abort
    """
    return mismatch <= tolerance


def create_motive_matrices(base_matrix, n_actions=8, motive_bonus=0.5):
    """
    Generate motive matrices from a base payoff matrix.

    Each motive matrix increases payoff for receiving that type of behavior from other.
    Motive i rewards other's action i most strongly.

    Args:
        base_matrix: The base n×n payoff matrix
        n_actions: Number of actions/motives
        motive_bonus: How much to boost payoffs for motive-congruent responses

    Returns:
        List of n×n motive matrices
    """
    matrices = []

    for m in range(n_actions):
        matrix = base_matrix.copy()
        # Motive m emphasizes column m (other's action m)
        # Bonus decays with distance from target action
        for j in range(n_actions):
            distance = min(abs(m - j), n_actions - abs(m - j))  # circular distance
            alignment = 1.0 - distance / (n_actions / 2)
            alignment = max(0, alignment)
            matrix[:, j] += motive_bonus * alignment

        matrices.append(matrix)

    return matrices
