"""
Two-agent simulation with visualization.

- Book (other-concept): What I think the other will do
- Self-concept: What I need from others (for motives)
- When other's behavior deviates from my self-concept needs → quit
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Optional

OCTANT_ORDER = ["LM", "NO", "PA", "BC", "DE", "FG", "HI", "JK"]


def softmax_policy(Q, beta):
    scaled_Q = beta * Q
    max_Q = np.max(scaled_Q)
    exp_Q = np.exp(scaled_Q - max_Q)
    return exp_Q / np.sum(exp_Q)


def get_distribution(name, n_actions=8, amplitude=1.0):
    """Get a distribution peaked at a specific octant."""
    if name == "uniform":
        return np.full(n_actions, 1.0 / n_actions)

    peaks = {o: i for i, o in enumerate(OCTANT_ORDER)}
    if name in peaks:
        peak = peaks[name]
        x = np.arange(n_actions)
        # Compute circular (minimal) distance from peak for each octant
        circ_dist = np.minimum(np.abs(x - peak), n_actions - np.abs(x - peak))
        # Use a cosine function for smooth circular distribution
        # Highest at peak (cos(0)=1), lowest at opposite (cos(pi)=-1)
        dist = amplitude * (np.cos(np.pi * circ_dist / (n_actions / 2)))
        # Shift to be all positive
        dist = dist - dist.min() + 0.01
        dist = dist / np.sum(dist)
        return dist
    return np.full(n_actions, 1.0 / n_actions)


def empirical_distribution(history, n_actions=8):
    """Convert action history to distribution."""
    dist = np.zeros(n_actions)
    for a in history:
        dist[a] += 1
    if np.sum(dist) > 0:
        dist /= np.sum(dist)
    else:
        dist = np.full(n_actions, 1.0 / n_actions)
    return dist


# =============================================================================
# COMPLEMENTARITY & STATISTICS
# =============================================================================


def compute_complementarity(action1, action2, n_actions=8):
    """
    Compute complementarity between two actions on the circumplex.

    Complementarity = 1 when actions are adjacent/same on circumplex.
    Complementarity = 0 when actions are opposite.
    """
    dist = min(abs(action1 - action2), n_actions - abs(action1 - action2))
    return 1.0 - dist / (n_actions / 2)


def compute_reciprocity(action1, action2, n_actions=8):
    """
    Compute reciprocity (exact matching).

    Returns 1 if same action, 0 otherwise.
    """
    return 1.0 if action1 == action2 else 0.0


def compute_warmth_score(action, n_actions=8):
    """
    Score how 'warm' an action is.

    Warm: LM=0, NO=1, JK=7 (+1)
    Cold: BC=3, DE=4, FG=5 (-1)
    Neutral: PA=2, HI=6 (0)
    """
    warmth = np.array([1.0, 1.0, 0.0, -1.0, -1.0, -1.0, 0.0, 1.0])
    return warmth[action]


def compute_dominance_score(action, n_actions=8):
    """
    Score how 'dominant' an action is.

    Dominant: NO=1, PA=2, BC=3 (+1)
    Submissive: FG=5, HI=6, JK=7 (-1)
    Neutral: LM=0, DE=4 (0)
    """
    dominance = np.array([0.0, 1.0, 1.0, 1.0, 0.0, -1.0, -1.0, -1.0])
    return dominance[action]


def compute_relationship_statistics(history, agent1, agent2):
    """
    Compute comprehensive statistics for a relationship.

    Returns dict with:
    - complementarity: mean complementarity score
    - reciprocity: mean exact matching
    - warmth_agent1/2: mean warmth of each agent's behavior
    - dominance_agent1/2: mean dominance of each agent's behavior
    - need_fulfillment1/2: how well each agent's needs were met
    - correlation: action correlation between agents
    """
    actions1 = history["actions1"]
    actions2 = history["actions2"]
    n = len(actions1)
    n_actions = agent1.n_actions

    if n == 0:
        return {}

    # Complementarity and reciprocity
    comp_scores = [
        compute_complementarity(a1, a2, n_actions) for a1, a2 in zip(actions1, actions2)
    ]
    recip_scores = [
        compute_reciprocity(a1, a2, n_actions) for a1, a2 in zip(actions1, actions2)
    ]

    # Warmth and dominance
    warmth1 = [compute_warmth_score(a, n_actions) for a in actions1]
    warmth2 = [compute_warmth_score(a, n_actions) for a in actions2]
    dom1 = [compute_dominance_score(a, n_actions) for a in actions1]
    dom2 = [compute_dominance_score(a, n_actions) for a in actions2]

    # Need fulfillment: how well other met my needs
    dist1 = empirical_distribution(actions1, n_actions)
    dist2 = empirical_distribution(actions2, n_actions)

    # Agent1's needs met by Agent2's behavior
    need_met1 = np.sum(agent1.self_concept * dist2)
    # Agent2's needs met by Agent1's behavior
    need_met2 = np.sum(agent2.self_concept * dist1)

    # Behavioral correlation
    if n > 1:
        correlation = np.corrcoef(actions1, actions2)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0
    else:
        correlation = 0.0

    # Warmth reciprocity: do warm actions elicit warm responses?
    warmth_recip = np.corrcoef(warmth1[:-1], warmth2[1:])[0, 1] if n > 2 else 0.0
    if np.isnan(warmth_recip):
        warmth_recip = 0.0

    # Dominance complementarity: do dominant actions elicit submissive responses?
    dom_comp = np.corrcoef(dom1[:-1], dom2[1:])[0, 1] if n > 2 else 0.0
    if np.isnan(dom_comp):
        dom_comp = 0.0

    return {
        "complementarity": np.mean(comp_scores),
        "reciprocity": np.mean(recip_scores),
        "warmth_agent1": np.mean(warmth1),
        "warmth_agent2": np.mean(warmth2),
        "dominance_agent1": np.mean(dom1),
        "dominance_agent2": np.mean(dom2),
        "need_fulfillment1": need_met1,
        "need_fulfillment2": need_met2,
        "action_correlation": correlation,
        "warmth_reciprocity": warmth_recip,
        "dominance_complementarity": -dom_comp,  # Negative = complementary
        "total_rounds": n,
        "total_payoff1": sum(history["payoffs1"]),
        "total_payoff2": sum(history["payoffs2"]),
    }


def print_relationship_stats(stats, agent1_name, agent2_name):
    """Print formatted statistics for a relationship."""
    print(f"\n  Interaction Patterns:")
    print(
        f"    Complementarity:     {stats['complementarity']:.3f} (1=perfect match on circumplex)"
    )
    print(f"    Reciprocity:         {stats['reciprocity']:.3f} (1=exact matching)")
    print(f"    Action correlation:  {stats['action_correlation']:.3f}")

    print(f"\n  Interpersonal Style:")
    print(f"    {agent1_name} warmth:    {stats['warmth_agent1']:+.3f} (+warm, -cold)")
    print(f"    {agent2_name} warmth:    {stats['warmth_agent2']:+.3f}")
    print(
        f"    {agent1_name} dominance: {stats['dominance_agent1']:+.3f} (+dominant, -submissive)"
    )
    print(f"    {agent2_name} dominance: {stats['dominance_agent2']:+.3f}")

    print(f"\n  Circumplex Dynamics:")
    print(
        f"    Warmth reciprocity:       {stats['warmth_reciprocity']:+.3f} (+reciprocal)"
    )
    print(
        f"    Dominance complementarity:{stats['dominance_complementarity']:+.3f} (+complementary)"
    )

    print(f"\n  Need Fulfillment:")
    print(f"    {agent1_name}'s needs met: {stats['need_fulfillment1']:.3f}")
    print(f"    {agent2_name}'s needs met: {stats['need_fulfillment2']:.3f}")

    print(f"\n  Outcomes:")
    print(f"    Total rounds: {stats['total_rounds']}")
    print(f"    {agent1_name} total payoff: {stats['total_payoff1']:.2f}")
    print(f"    {agent2_name} total payoff: {stats['total_payoff2']:.2f}")


@dataclass
class Agent:
    """
    Agent with:
    - book: current belief about the other's behavioral tendencies
    - self_concept: what I need from the other
    - Q_library: learned Q-values for different book states
    - relationship_memory: history of breakups to learn from

    Key insight: Q-values are specific to the current book (model of other).
    When the book changes, retrieve or interpolate Q-values for that context.

    Retrospective correction: When detecting a book change, we look back to find
    when the shift actually started and "unlearn" contaminated interactions from
    the old book, re-attributing them to the new book.
    """

    name: str
    n_actions: int

    # Self-concept: what I need from the other (my desires)
    self_concept_name: str = "uniform"
    self_concept: np.ndarray = field(default=None)

    # Book: my current belief about the other's behavioral tendencies
    book_name: str = "uniform"
    book: np.ndarray = field(default=None)

    # Tolerance for mismatch - can adapt based on relationship history
    tolerance: float = 1.0
    base_tolerance: float = 1.0

    # Learning parameters
    alpha: float = 0.1
    beta: float = 5.0

    # Q-values library: maps book_name -> learned Q-values
    # Each book state has its own learned behavior
    Q_library: dict = field(default_factory=dict)
    Q: np.ndarray = field(default=None)

    # Relationship memory: track breakups to learn from
    breakup_count: int = 0
    successful_relationships: int = 0

    # State
    active: bool = True
    action_history: List[int] = field(default_factory=list)
    other_history: List[int] = field(default_factory=list)
    payoff_history: List[float] = field(default_factory=list)

    # Tracking
    mismatch_history: List[float] = field(default_factory=list)
    book_history: List[np.ndarray] = field(default_factory=list)

    # For retrospective correction: track Q-values at each book update
    # Stores (interaction_index, book_snapshot, Q_snapshot)
    book_change_log: List[tuple] = field(default_factory=list)
    last_book_update_idx: int = 0

    def __post_init__(self):
        if self.book is None:
            self.book = get_distribution(self.book_name, self.n_actions)
        if self.self_concept is None:
            self.self_concept = get_distribution(self.self_concept_name, self.n_actions)
        self.base_tolerance = self.tolerance
        # Initialize Q-values for starting book
        self._init_q_for_current_book()
        # Log initial state
        self.book_change_log = [(0, self.book.copy(), self.Q.copy())]

    def _get_book_key(self, book_dist):
        """Convert book distribution to a hashable key for Q_library."""
        # Discretize to avoid floating point issues
        return tuple(np.round(book_dist, 2))

    def _init_q_for_current_book(self):
        """Initialize or retrieve Q-values for current book state."""
        key = self._get_book_key(self.book)
        if key in self.Q_library:
            self.Q = self.Q_library[key].copy()
        else:
            # New book state: interpolate from similar known states
            self.Q = self._interpolate_q(self.book)
            self.Q_library[key] = self.Q.copy()

    def _interpolate_q(self, target_book):
        """
        Interpolate Q-values for a new book state from known states.

        Weighted average based on similarity (inverse distance).
        """
        if not self.Q_library:
            return np.zeros(self.n_actions)

        total_weight = 0.0
        weighted_q = np.zeros(self.n_actions)

        for key, q_vals in self.Q_library.items():
            known_book = np.array(key)
            distance = np.linalg.norm(target_book - known_book)
            if distance < 0.001:
                return q_vals.copy()
            weight = 1.0 / (distance + 0.1)
            weighted_q += weight * q_vals
            total_weight += weight

        if total_weight > 0:
            return weighted_q / total_weight
        return np.zeros(self.n_actions)

    def _save_q_for_current_book(self):
        """Save current Q-values to library for current book state."""
        key = self._get_book_key(self.book)
        self.Q_library[key] = self.Q.copy()

    def act(self):
        """Choose action via softmax on Q-values."""
        probs = softmax_policy(self.Q, self.beta)
        action = np.random.choice(self.n_actions, p=probs)
        self.action_history.append(action)
        return action, probs

    def observe_other(self, other_action):
        """Record what the other did."""
        self.other_history.append(other_action)

    def compute_satisfaction(self, other_action):
        """
        How satisfied am I with what the other did?

        Payoff = how well other's action matches MY self-concept needs.
        """
        payoff = self.self_concept[other_action]
        self.payoff_history.append(payoff)
        return payoff

    def learn(self, my_action, other_action):
        """
        Q-learning update based on observed interaction.

        Key insight: I learn that certain actions of MINE tend to
        correlate with certain responses from the other.
        If I do X and they respond with something I like, X becomes valuable.
        """
        reward = self.self_concept[other_action]
        self.Q[my_action] += self.alpha * (reward - self.Q[my_action])

    def _find_shift_point(self, new_book, lookback=20):
        """
        Find when the other's behavior actually started shifting toward new_book.
        Simplified: just look at last N interactions in chunks.
        """
        if len(self.other_history) < 10:
            return len(self.other_history)

        # Simple heuristic: exclude last 1/3 of recent window
        exclude_n = min(lookback // 3, len(self.other_history) // 4)
        return max(0, len(self.other_history) - exclude_n)

    def _retrospective_correction(self, old_book, new_book, shift_idx):
        """
        Correct Q-values: restore old book's Q to before contamination,
        attribute recent interactions to new book.
        """
        n_to_correct = len(self.action_history) - shift_idx
        if n_to_correct <= 0:
            return

        # Restore old book Q-values from last logged clean state
        old_key = self._get_book_key(old_book)
        if old_key in self.Q_library and self.book_change_log:
            # Get Q from before shift
            for idx, _, q_snap in reversed(self.book_change_log):
                if idx <= shift_idx:
                    self.Q_library[old_key] = q_snap.copy()
                    break

        # Learn recent interactions under new book
        new_key = self._get_book_key(new_book)
        if new_key not in self.Q_library:
            self.Q_library[new_key] = self._interpolate_q(new_book)

        new_q = self.Q_library[new_key].copy()
        for i in range(shift_idx, len(self.action_history)):
            my_action = self.action_history[i]
            other_action = self.other_history[i]
            reward = self.self_concept[other_action]
            new_q[my_action] += self.alpha * (reward - new_q[my_action])
        self.Q_library[new_key] = new_q

    def update_book(self, window=20, prior_weight=0.3):
        """
        Update my belief about the other (book) based on their behavior.

        When book changes significantly:
        1. Find when the shift actually started (retrospectively)
        2. Unlearn contaminated interactions from old book
        3. Re-learn those interactions under new book
        4. Switch to Q-values for new context
        """
        if len(self.other_history) < 2:
            return

        current_idx = len(self.other_history)

        # Save Q-values for current book before updating
        self._save_q_for_current_book()

        recent = self.other_history[-window:]
        empirical = empirical_distribution(recent, self.n_actions)

        old_book = self.book.copy()
        self.book = prior_weight * self.book + (1 - prior_weight) * empirical
        self.book /= np.sum(self.book)
        self.book_history.append(self.book.copy())

        # If book changed significantly, do retrospective correction
        book_change = np.linalg.norm(self.book - old_book)
        if book_change > 0.05:
            # Find when the shift actually started
            shift_idx = self._find_shift_point(self.book, lookback=window * 2)

            # Retrospectively correct Q-values
            self._retrospective_correction(old_book, self.book, shift_idx)

            # Log this change for future retrospection
            self.book_change_log.append((current_idx, self.book.copy(), self.Q.copy()))
            self.last_book_update_idx = current_idx

            # Load Q-values for new book state
            self._init_q_for_current_book()

    def check_needs_met(self, window=50):
        """
        Check if other's behavior matches my self-concept needs.
        If mismatch too high, quit relationship.
        """
        if len(self.other_history) < window:
            return

        recent = self.other_history[-window:]
        other_behavior = empirical_distribution(recent, self.n_actions)

        mismatch = np.linalg.norm(self.self_concept - other_behavior)
        self.mismatch_history.append(mismatch)

        if mismatch > self.tolerance:
            self.active = False

    def on_relationship_end(self, success: bool):
        """
        Called when a relationship ends. Learn from the outcome.

        Adjust tolerance based on history to optimize for fewer breakups.
        """
        if success:
            self.successful_relationships += 1
        else:
            self.breakup_count += 1

        # Adapt tolerance based on history
        total = self.breakup_count + self.successful_relationships
        if total > 0:
            success_rate = self.successful_relationships / total
            # If many breakups, become more tolerant
            # If mostly successful, can afford to be pickier
            self.tolerance = self.base_tolerance * (0.5 + success_rate)

    def reset_for_new_relationship(self, preserve_book=True):
        """Reset state for a new relationship, keeping learned Q-values and book."""
        self.active = True
        self.action_history = []
        self.other_history = []
        self.payoff_history = []
        self.mismatch_history = []
        self.book_history = []
        # Reset retrospective correction tracking
        self.last_book_update_idx = 0
        self.book_change_log = [(0, self.book.copy(), self.Q.copy())]
        # Keep learned book from previous relationship unless explicitly reset
        if not preserve_book:
            self.book = get_distribution(self.book_name, self.n_actions)
        # Initialize Q-values for current book state (retrieves from library if known)
        self._init_q_for_current_book()


def run_simulation(
    agent1: Agent, agent2: Agent, rounds: int = 500, update_interval: int = 10
) -> dict:
    """
    Run single relationship simulation.

    Each round:
    1. Both agents act based on Q-values (specific to current book state)
    2. Each agent computes satisfaction based on their own self_concept
    3. Q-learning updates
    4. Periodically: update book, switch Q-context if needed, check needs
    """

    history = {
        "probs1": [],
        "probs2": [],
        "actions1": [],
        "actions2": [],
        "payoffs1": [],
        "payoffs2": [],
        "mismatch1": [],
        "mismatch2": [],
        "book1": [],
        "book2": [],
        "abort_round": None,
        "who_quit": None,
    }

    for t in range(rounds):
        if not agent1.active:
            history["abort_round"] = t
            history["who_quit"] = agent1.name
            break
        if not agent2.active:
            history["abort_round"] = t
            history["who_quit"] = agent2.name
            break

        # Both agents choose actions
        a1, p1 = agent1.act()
        a2, p2 = agent2.act()

        history["probs1"].append(p1.copy())
        history["probs2"].append(p2.copy())
        history["actions1"].append(a1)
        history["actions2"].append(a2)

        # Each agent observes what the other did
        agent1.observe_other(a2)
        agent2.observe_other(a1)

        # Satisfaction: how well did the other's action meet MY needs?
        pay1 = agent1.compute_satisfaction(a2)
        pay2 = agent2.compute_satisfaction(a1)

        history["payoffs1"].append(pay1)
        history["payoffs2"].append(pay2)

        # Learn: update Q-values based on what happened
        agent1.learn(a1, a2)
        agent2.learn(a2, a1)

        # Periodic updates
        if t > 0 and t % update_interval == 0:
            # Update beliefs about the other (may switch Q-context)
            agent1.update_book()
            agent2.update_book()

            # Check if my needs are being met
            agent1.check_needs_met()
            agent2.check_needs_met()

            # Track
            history["mismatch1"].append(
                agent1.mismatch_history[-1] if agent1.mismatch_history else 0
            )
            history["mismatch2"].append(
                agent2.mismatch_history[-1] if agent2.mismatch_history else 0
            )
            if agent1.book_history:
                history["book1"].append(agent1.book_history[-1])
            if agent2.book_history:
                history["book2"].append(agent2.book_history[-1])

    # Relationship ended - record outcome
    success = history["abort_round"] is None
    agent1.on_relationship_end(success)
    agent2.on_relationship_end(success)

    # Save final Q-values
    agent1._save_q_for_current_book()
    agent2._save_q_for_current_book()

    history["success"] = success
    history["agent1"] = agent1
    history["agent2"] = agent2
    return history


def run_multiple_relationships(
    agent1: Agent,
    agent2: Agent,
    n_relationships: int = 10,
    rounds_per_relationship: int = 500,
    update_interval: int = 10,
) -> List[dict]:
    """
    Run multiple relationships between the same two agents.

    Agents learn across relationships:
    - Q_library accumulates learned behaviors for different book states
    - Tolerance adapts based on breakup history
    """
    all_histories = []

    for r in range(n_relationships):
        # Reset for new relationship but keep learned Q-values and tolerance
        agent1.reset_for_new_relationship()
        agent2.reset_for_new_relationship()

        history = run_simulation(
            agent1,
            agent2,
            rounds=rounds_per_relationship,
            update_interval=update_interval,
        )
        history["relationship_num"] = r
        all_histories.append(history)

        status = "SUCCESS" if history["success"] else f"BREAKUP ({history['who_quit']})"
        print(
            f"  Relationship {r+1}: {status} "
            f"| tol1={agent1.tolerance:.2f} tol2={agent2.tolerance:.2f} "
            f"| Q-states1={len(agent1.Q_library)} Q-states2={len(agent2.Q_library)}"
        )

    return all_histories


def plot_results(history, agent1, agent2):
    """Visualize the simulation."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    rounds = len(history["actions1"])
    t = np.arange(rounds)

    # --- Row 1: Action distributions over time ---
    ax1, ax2 = axes[0]

    # Rolling action distribution
    window = 20
    dist1 = np.zeros((rounds, agent1.n_actions))
    dist2 = np.zeros((rounds, agent2.n_actions))
    for i in range(rounds):
        start = max(0, i - window)
        dist1[i] = empirical_distribution(
            history["actions1"][start : i + 1], agent1.n_actions
        )
        dist2[i] = empirical_distribution(
            history["actions2"][start : i + 1], agent2.n_actions
        )

    for j, name in enumerate(OCTANT_ORDER):
        ax1.plot(t, dist1[:, j], label=name, alpha=0.7)
        ax2.plot(t, dist2[:, j], label=name, alpha=0.7)

    ax1.set_title(f"{agent1.name} Behavior (rolling {window})")
    ax2.set_title(f"{agent2.name} Behavior (rolling {window})")
    ax1.set_ylabel("Probability")
    ax1.legend(loc="upper right", fontsize=8, ncol=2)
    ax2.legend(loc="upper right", fontsize=8, ncol=2)

    # Show what the OTHER needs as reference
    for j, name in enumerate(OCTANT_ORDER):
        ax1.axhline(agent2.self_concept[j], color=f"C{j}", linestyle=":", alpha=0.2)
        ax2.axhline(agent1.self_concept[j], color=f"C{j}", linestyle=":", alpha=0.2)

    # Mark abort
    if history["abort_round"]:
        for ax in [ax1, ax2]:
            ax.axvline(history["abort_round"], color="red", linestyle="--")

    # --- Row 2: Book evolution (belief about other's needs) ---
    ax3, ax4 = axes[1]

    if history["book1"]:
        book1 = np.array(history["book1"])
        book2 = np.array(history["book2"])
        t_book = np.arange(len(book1)) * 10  # update_interval

        for j, name in enumerate(OCTANT_ORDER):
            ax3.plot(t_book, book1[:, j], label=name, alpha=0.7)
            ax4.plot(t_book, book2[:, j], label=name, alpha=0.7)

    ax3.set_title(f"{agent1.name}'s book (belief: what does {agent2.name} need?)")
    ax4.set_title(f"{agent2.name}'s book (belief: what does {agent1.name} need?)")
    ax3.set_ylabel("Believed need")

    # Show actual other's self-concept as target (dotted)
    for j, name in enumerate(OCTANT_ORDER):
        ax3.axhline(agent2.self_concept[j], color=f"C{j}", linestyle=":", alpha=0.3)
        ax4.axhline(agent1.self_concept[j], color=f"C{j}", linestyle=":", alpha=0.3)

    # --- Row 3: Mismatch and payoffs ---
    ax5, ax6 = axes[2]

    if history["mismatch1"]:
        t_m = np.arange(len(history["mismatch1"])) * 10
        ax5.plot(t_m, history["mismatch1"], label=agent1.name, color="blue")
        ax5.plot(t_m, history["mismatch2"], label=agent2.name, color="orange")
        ax5.axhline(
            agent1.tolerance,
            color="blue",
            linestyle="--",
            alpha=0.5,
            label=f"{agent1.name} tolerance",
        )
        ax5.axhline(
            agent2.tolerance,
            color="orange",
            linestyle="--",
            alpha=0.5,
            label=f"{agent2.name} tolerance",
        )

    ax5.set_title("Mismatch (other's behavior vs my needs)")
    ax5.set_ylabel("Mismatch")
    ax5.set_xlabel("Round")
    ax5.legend()

    # Cumulative payoffs
    cum1 = np.cumsum(history["payoffs1"])
    cum2 = np.cumsum(history["payoffs2"])
    ax6.plot(t, cum1, label=agent1.name)
    ax6.plot(t, cum2, label=agent2.name)
    ax6.set_title("Cumulative Payoffs")
    ax6.set_ylabel("Total Payoff")
    ax6.set_xlabel("Round")
    ax6.legend()

    plt.tight_layout()

    # Summary text
    summary = f"""
    {agent1.name}: needs={agent1.self_concept_name}, tol={agent1.tolerance:.2f}
    {agent2.name}: needs={agent2.self_concept_name}, tol={agent2.tolerance:.2f}
    """
    if history.get("abort_round"):
        summary += f"\n    {history['who_quit']} quit at round {history['abort_round']}"
    else:
        summary += "\n    Relationship completed"

    fig.suptitle(summary, fontsize=10, y=1.02)
    plt.savefig("simulation_result.png", dpi=150, bbox_inches="tight")
    plt.show()

    return fig


# =============================================================================
# CONFIGURATION - MODIFY HERE
# =============================================================================

if __name__ == "__main__":
    n = 8
    n_relationships = 100
    rounds_per_rel = 1000

    # Agent1: Needs LM (warmth) from the other
    # Agent2: Needs NO (nurturing warmth) from the other
    # Both warm, but slightly different - can they learn each other's needs?

    agent1 = Agent(
        name="Agent1",
        n_actions=n,
        self_concept_name="LM",  # What I NEED from the other
        book_name="uniform",  # Initial belief about other
        tolerance=0.5,
        alpha=0.1,
        beta=5.0,
    )

    agent2 = Agent(
        name="Agent2",
        n_actions=n,
        self_concept_name="NO",  # What I NEED from the other
        book_name="uniform",
        tolerance=0.5,
        alpha=0.1,
        beta=5.0,
    )

    print("=" * 60)
    print("MULTI-RELATIONSHIP SIMULATION")
    print("=" * 60)
    print(
        f"\n{agent1.name}: needs {agent1.self_concept_name}, base_tol={agent1.tolerance}"
    )
    print(
        f"{agent2.name}: needs {agent2.self_concept_name}, base_tol={agent2.tolerance}"
    )
    print(f"\nRunning {n_relationships} relationships...")
    print("-" * 60)

    histories = run_multiple_relationships(
        agent1,
        agent2,
        n_relationships=n_relationships,
        rounds_per_relationship=rounds_per_rel,
        update_interval=10,
    )

    # Compute statistics for each relationship
    all_stats = []
    for i, h in enumerate(histories):
        stats = compute_relationship_statistics(h, agent1, agent2)
        stats["success"] = h["success"]
        all_stats.append(stats)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    successes = sum(1 for h in histories if h["success"])
    print(f"\nRelationships: {successes}/{n_relationships} successful")

    print(f"\n{agent1.name}:")
    print(f"  Final tolerance: {agent1.tolerance:.2f}")
    print(f"  Q-states learned: {len(agent1.Q_library)}")
    print(f"  Final book: {[f'{v:.2f}' for v in agent1.book]}")
    print(
        f"  Breakups: {agent1.breakup_count}, Successes: {agent1.successful_relationships}"
    )

    print(f"\n{agent2.name}:")
    print(f"  Final tolerance: {agent2.tolerance:.2f}")
    print(f"  Q-states learned: {len(agent2.Q_library)}")
    print(f"  Final book: {[f'{v:.2f}' for v in agent2.book]}")
    print(
        f"  Breakups: {agent2.breakup_count}, Successes: {agent2.successful_relationships}"
    )

    # Aggregate statistics across relationships
    print("\n" + "=" * 60)
    print("INTERPERSONAL DYNAMICS STATISTICS")
    print("=" * 60)

    # Average across all relationships
    avg_comp = np.mean([s["complementarity"] for s in all_stats])
    avg_recip = np.mean([s["reciprocity"] for s in all_stats])
    avg_warmth_recip = np.mean([s["warmth_reciprocity"] for s in all_stats])
    avg_dom_comp = np.mean([s["dominance_complementarity"] for s in all_stats])
    avg_need1 = np.mean([s["need_fulfillment1"] for s in all_stats])
    avg_need2 = np.mean([s["need_fulfillment2"] for s in all_stats])

    print(f"\nAveraged across {n_relationships} relationships:")
    print(f"  Complementarity:          {avg_comp:.3f} (circumplex proximity)")
    print(f"  Reciprocity:              {avg_recip:.3f} (exact matching)")
    print(f"  Warmth reciprocity:       {avg_warmth_recip:+.3f} (warm begets warm)")
    print(f"  Dominance complementarity:{avg_dom_comp:+.3f} (dom begets sub)")
    print(f"  {agent1.name} needs met:     {avg_need1:.3f}")
    print(f"  {agent2.name} needs met:     {avg_need2:.3f}")

    # Evolution across relationships
    print("\n  Evolution across relationships:")
    print("  Rel#  Comp   Recip  WarmR  DomC   Needs1 Needs2 Status")
    print("  " + "-" * 56)
    for i, s in enumerate(all_stats):
        status = "OK" if s["success"] else "QUIT"
        print(
            f"  {i+1:3d}   {s['complementarity']:.3f}  {s['reciprocity']:.3f}  "
            f"{s['warmth_reciprocity']:+.3f}  {s['dominance_complementarity']:+.3f}  "
            f"{s['need_fulfillment1']:.3f}  {s['need_fulfillment2']:.3f}  {status}"
        )

    # Detailed stats for last relationship
    print("\n" + "-" * 60)
    print(f"LAST RELATIONSHIP DETAILS (#{n_relationships})")
    print("-" * 60)
    print_relationship_stats(all_stats[-1], agent1.name, agent2.name)

    # Plot last relationship
    print("\nPlotting last relationship...")
    plot_results(histories[-1], agent1, agent2)
