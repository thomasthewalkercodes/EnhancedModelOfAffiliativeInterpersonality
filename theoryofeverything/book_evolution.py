"""
Book Evolution Tournament

Evolve Agent1's book (other-concept) to maximize COMMUNAL needs fulfillment
(Agent1's needs met + Agent2's needs met) against a fixed Agent2.

Only books evolve - optimizing for mutual satisfaction.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field

from two_agent_sim import (
    Agent,
    run_simulation,
    compute_relationship_statistics,
    get_distribution,
    OCTANT_ORDER,
)


@dataclass
class BookGenome:
    """Evolvable book (other-concept) for an agent."""

    # Book can be either a named distribution or a raw array
    book_name: str = "uniform"
    book_array: np.ndarray = field(default=None)
    n_actions: int = 8

    # Fitness tracking
    total_communal_payoff: float = 0.0
    relationships_played: int = 0

    def __post_init__(self):
        if self.book_array is None:
            self.book_array = get_distribution(self.book_name, self.n_actions)

    @property
    def fitness(self):
        if self.relationships_played == 0:
            return 0.0
        return self.total_communal_payoff / self.relationships_played

    def reset_fitness(self):
        self.total_communal_payoff = 0.0
        self.relationships_played = 0

    def copy(self):
        return BookGenome(
            book_name=self.book_name,
            book_array=self.book_array.copy(),
            n_actions=self.n_actions,
        )


def mutate_book(genome: BookGenome, mutation_rate: float = 0.3) -> BookGenome:
    """
    Mutate a book genome.

    Can either:
    1. Switch to a named book distribution
    2. Add noise to current distribution
    3. Shift peak slightly on circumplex
    """
    new_genome = genome.copy()

    if np.random.random() < mutation_rate:
        mutation_type = np.random.choice(["named", "noise", "shift"])

        if mutation_type == "named":
            # Switch to a random named book
            new_genome.book_name = np.random.choice(OCTANT_ORDER + ["uniform"])
            new_genome.book_array = get_distribution(
                new_genome.book_name, new_genome.n_actions
            )

        elif mutation_type == "noise":
            # Add Gaussian noise to current distribution
            noise = np.random.normal(0, 0.1, new_genome.n_actions)
            new_genome.book_array = new_genome.book_array + noise
            new_genome.book_array = np.maximum(new_genome.book_array, 0.01)
            new_genome.book_array /= np.sum(new_genome.book_array)
            new_genome.book_name = "evolved"

        elif mutation_type == "shift":
            # Shift distribution by rotating on circumplex
            shift = np.random.choice([-1, 1])
            new_genome.book_array = np.roll(new_genome.book_array, shift)
            new_genome.book_name = "evolved"

    return new_genome


def crossover_books(parent1: BookGenome, parent2: BookGenome) -> BookGenome:
    """Create offspring by averaging two parent book distributions."""
    weight = np.random.uniform(0.3, 0.7)
    child_array = weight * parent1.book_array + (1 - weight) * parent2.book_array
    child_array /= np.sum(child_array)

    return BookGenome(
        book_name="evolved",
        book_array=child_array,
        n_actions=parent1.n_actions,
    )


def evaluate_genome(
    genome: BookGenome,
    fixed_agent2: Agent,
    n_relationships: int = 10,
    rounds_per_rel: int = 300,
    update_interval: int = 10,
    agent1_needs: str = "LM",
    both_agents_learn: bool = False,
) -> float:
    """
    Evaluate a book genome by running relationships against fixed Agent2.

    Returns: communal fitness (sum of both agents' needs fulfillment)
    """
    n = genome.n_actions

    # Create Agent1 with this genome's book
    agent1 = Agent(
        name="Evolving",
        n_actions=n,
        self_concept_name=agent1_needs,  # Agent1's needs (configurable)
        book_name=genome.book_name,
        tolerance=1.0,  # High tolerance - focus on learning
        alpha=0.1,
        beta=5.0,
    )
    # Override book with genome's array
    agent1.book = genome.book_array.copy()

    total_communal = 0.0

    for _ in range(n_relationships):
        # Reset agents for new relationship
        agent1.reset_for_new_relationship(preserve_book=True)
        # Create fresh copy of agent2 for each relationship
        # If both_agents_learn=False, set alpha=0 so agent2 doesn't update Q-values
        agent2_copy = Agent(
            name=fixed_agent2.name,
            n_actions=fixed_agent2.n_actions,
            self_concept_name=fixed_agent2.self_concept_name,
            book_name=fixed_agent2.book_name,
            tolerance=fixed_agent2.tolerance,
            alpha=fixed_agent2.alpha if both_agents_learn else 0.0,
            beta=fixed_agent2.beta,
        )

        # Run simulation
        history = run_simulation(
            agent1, agent2_copy, rounds=rounds_per_rel, update_interval=update_interval
        )

        # Compute communal payoff
        stats = compute_relationship_statistics(history, agent1, agent2_copy)
        if stats:
            communal = stats["need_fulfillment1"] + stats["need_fulfillment2"]
            total_communal += communal

    genome.total_communal_payoff = total_communal
    genome.relationships_played = n_relationships

    return genome.fitness


def run_book_tournament(
    population_size: int = 20,
    generations: int = 50,
    n_relationships: int = 10,
    rounds_per_rel: int = 300,
    survival_ratio: float = 0.3,
    mutation_rate: float = 0.3,
    agent1_needs: str = "LM",
    fixed_agent2_needs: str = "NO",
    both_agents_learn: bool = False,
    verbose: bool = True,
):
    """
    Run evolutionary tournament to optimize Agent1's book.

    Fitness = communal needs fulfillment (Agent1 + Agent2)

    Args:
        agent1_needs: Self-concept (needs) for Agent1
        fixed_agent2_needs: Self-concept (needs) for Agent2
        both_agents_learn: If True, both agents update books during simulation
    """
    n_actions = 8

    # Create fixed Agent2
    fixed_agent2 = Agent(
        name="Fixed",
        n_actions=n_actions,
        self_concept_name=fixed_agent2_needs,
        book_name="uniform",
        tolerance=1.0,
        alpha=0.1,
        beta=5.0,
    )

    # Initialize population with diverse books
    population = []
    for book_name in OCTANT_ORDER + ["uniform"]:
        genome = BookGenome(book_name=book_name, n_actions=n_actions)
        population.append(genome)
    # Fill rest with random/evolved books
    while len(population) < population_size:
        base = np.random.choice(population)
        mutant = mutate_book(base, mutation_rate=1.0)
        population.append(mutant)

    history = []

    print("=" * 70)
    print("BOOK EVOLUTION TOURNAMENT")
    print("=" * 70)
    print(f"Agent1: needs {agent1_needs}, evolving book to optimize communal payoff")
    print(
        f"Agent2: needs {fixed_agent2_needs}"
        + (" (ALSO LEARNING)" if both_agents_learn else " (FIXED)")
    )
    print(f"Population: {population_size}, Generations: {generations}")
    print(
        f"Relationships per eval: {n_relationships}, Rounds per rel: {rounds_per_rel}"
    )
    print("=" * 70)

    for gen in range(generations):
        # Reset fitness
        for genome in population:
            genome.reset_fitness()

        # Evaluate each genome
        for genome in population:
            evaluate_genome(
                genome,
                fixed_agent2,
                n_relationships=n_relationships,
                rounds_per_rel=rounds_per_rel,
                agent1_needs=agent1_needs,
                both_agents_learn=both_agents_learn,
            )

        # Sort by fitness
        population.sort(key=lambda g: g.fitness, reverse=True)

        # Record stats
        fitnesses = [g.fitness for g in population]
        best = population[0]
        gen_stats = {
            "generation": gen,
            "mean_fitness": np.mean(fitnesses),
            "max_fitness": np.max(fitnesses),
            "min_fitness": np.min(fitnesses),
            "best_book_name": best.book_name,
            "best_book_array": best.book_array.copy(),
        }
        history.append(gen_stats)

        if verbose:
            print(
                f"Gen {gen:3d}: mean={gen_stats['mean_fitness']:.4f}, "
                f"best={gen_stats['max_fitness']:.4f} ({best.book_name})"
            )

        # Selection and reproduction
        n_survivors = max(2, int(population_size * survival_ratio))
        survivors = population[:n_survivors]

        # Create next generation
        new_population = []

        # Keep best (elitism)
        new_population.append(survivors[0].copy())

        while len(new_population) < population_size:
            if np.random.random() < 0.7:
                # Mutation
                parent = np.random.choice(survivors)
                offspring = mutate_book(parent, mutation_rate)
            else:
                # Crossover
                p1, p2 = np.random.choice(survivors, size=2, replace=False)
                offspring = crossover_books(p1, p2)
                offspring = mutate_book(offspring, mutation_rate * 0.5)

            new_population.append(offspring)

        population = new_population

    return history, population, fixed_agent2


def plot_evolution(history, best_genome, fixed_agent2, agent1_needs_name: str = "LM"):
    """Plot evolution results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    generations = [h["generation"] for h in history]

    # Fitness over time
    ax1 = axes[0, 0]
    ax1.plot(generations, [h["mean_fitness"] for h in history], label="Mean", alpha=0.7)
    ax1.plot(generations, [h["max_fitness"] for h in history], label="Best", alpha=0.7)
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Communal Fitness")
    ax1.set_title("Fitness Evolution")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Best book distribution
    ax2 = axes[0, 1]
    x = np.arange(8)
    ax2.bar(x - 0.2, best_genome.book_array, 0.4, label="Evolved Book", alpha=0.7)
    ax2.bar(
        x + 0.2,
        fixed_agent2.self_concept,
        0.4,
        label=f"Agent2 Needs ({fixed_agent2.self_concept_name})",
        alpha=0.7,
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(OCTANT_ORDER)
    ax2.set_ylabel("Probability")
    ax2.set_title("Best Evolved Book vs Agent2's Needs")
    ax2.legend()

    # Book evolution heatmap
    ax3 = axes[1, 0]
    book_evolution = np.array([h["best_book_array"] for h in history])
    im = ax3.imshow(book_evolution.T, aspect="auto", cmap="viridis")
    ax3.set_xlabel("Generation")
    ax3.set_ylabel("Octant")
    ax3.set_yticks(range(8))
    ax3.set_yticklabels(OCTANT_ORDER)
    ax3.set_title("Best Book Evolution Over Generations")
    plt.colorbar(im, ax=ax3)

    # Compare evolved vs uniform vs target
    ax4 = axes[1, 1]
    uniform = np.full(8, 1 / 8)
    agent1_needs_dist = get_distribution(agent1_needs_name, 8)

    x = np.arange(8)
    width = 0.2
    ax4.bar(x - 1.5 * width, best_genome.book_array, width, label="Evolved Book")
    ax4.bar(x - 0.5 * width, uniform, width, label="Uniform")
    ax4.bar(
        x + 0.5 * width,
        agent1_needs_dist,
        width,
        label=f"Agent1 Needs ({agent1_needs_name})",
    )
    ax4.bar(
        x + 1.5 * width,
        fixed_agent2.self_concept,
        width,
        label=f"Agent2 Needs ({fixed_agent2.self_concept_name})",
    )
    ax4.set_xticks(x)
    ax4.set_xticklabels(OCTANT_ORDER)
    ax4.set_ylabel("Probability")
    ax4.set_title("Distribution Comparison")
    ax4.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("book_evolution_results.png", dpi=150, bbox_inches="tight")
    plt.show()

    return fig


def analyze_behavior_patterns(
    best_genome: BookGenome,
    fixed_agent2: Agent,
    agent1_needs: str = "LM",
    both_agents_learn: bool = False,
    n_relationships: int = 20,
    rounds_per_rel: int = 400,
):
    """
    Run the best genome and collect detailed behavior statistics.
    Analyzes communion (warmth) and agency (dominance) matching.
    """
    from two_agent_sim import empirical_distribution

    n = best_genome.n_actions

    # Create Agent1 with best genome's book
    agent1 = Agent(
        name="BestEvolved",
        n_actions=n,
        self_concept_name=agent1_needs,
        book_name=best_genome.book_name,
        tolerance=1.0,
        alpha=0.1,
        beta=5.0,
    )
    agent1.book = best_genome.book_array.copy()

    # Collect all actions
    all_actions1 = []
    all_actions2 = []
    all_stats = []

    for _ in range(n_relationships):
        agent1.reset_for_new_relationship(preserve_book=True)
        agent2_copy = Agent(
            name=fixed_agent2.name,
            n_actions=fixed_agent2.n_actions,
            self_concept_name=fixed_agent2.self_concept_name,
            book_name=fixed_agent2.book_name,
            tolerance=fixed_agent2.tolerance,
            alpha=fixed_agent2.alpha if both_agents_learn else 0.0,
            beta=fixed_agent2.beta,
        )

        history = run_simulation(
            agent1, agent2_copy, rounds=rounds_per_rel, update_interval=10
        )
        all_actions1.extend(history["actions1"])
        all_actions2.extend(history["actions2"])

        stats = compute_relationship_statistics(history, agent1, agent2_copy)
        if stats:
            all_stats.append(stats)

    # Compute behavior distributions
    behavior_dist1 = empirical_distribution(all_actions1, n)
    behavior_dist2 = empirical_distribution(all_actions2, n)

    # Warmth scores per octant (OCTANT_ORDER = LM, NO, PA, BC, DE, FG, HI, JK)
    # Warm: LM=0, NO=1, JK=7
    # Cold: BC=3, DE=4, FG=5
    # Neutral: PA=2, HI=6
    warmth_scores = np.array([1.0, 1.0, 0.0, -1.0, -1.0, -1.0, 0.0, 1.0])

    # Agency scores (OCTANT_ORDER = LM, NO, PA, BC, DE, FG, HI, JK)
    # Dominant: NO=1, PA=2, BC=3
    # Submissive: FG=5, HI=6, JK=7
    # Neutral: LM=0, DE=4
    agency_scores = np.array([0.0, 1.0, 1.0, 1.0, 0.0, -1.0, -1.0, -1.0])

    # Compute average warmth and agency for each agent
    avg_warmth1 = np.sum(behavior_dist1 * warmth_scores)
    avg_warmth2 = np.sum(behavior_dist2 * warmth_scores)
    avg_agency1 = np.sum(behavior_dist1 * agency_scores)
    avg_agency2 = np.sum(behavior_dist2 * agency_scores)

    # Base rate (uniform distribution)
    base_warmth = np.sum(np.full(n, 1 / n) * warmth_scores)
    base_agency = np.sum(np.full(n, 1 / n) * agency_scores)

    # Communion matching: warmth-warmth reciprocity (both warm or both cold)
    # Compute per-interaction warmth correlation
    warmth1_per_round = [warmth_scores[a] for a in all_actions1]
    warmth2_per_round = [warmth_scores[a] for a in all_actions2]

    # Same-sign warmth (both warm or both cold)
    communion_matches = sum(
        1
        for w1, w2 in zip(warmth1_per_round, warmth2_per_round)
        if (w1 > 0 and w2 > 0) or (w1 < 0 and w2 < 0)
    )
    communion_match_pct = communion_matches / len(all_actions1) * 100

    # Base rate for communion: Given ACTUAL behavior distributions, expected random co-occurrence
    # P(both warm) + P(both cold) if behaviors were independent
    p_warm1 = np.sum(behavior_dist1[warmth_scores > 0])  # P(agent1 warm)
    p_cold1 = np.sum(behavior_dist1[warmth_scores < 0])  # P(agent1 cold)
    p_warm2 = np.sum(behavior_dist2[warmth_scores > 0])  # P(agent2 warm)
    p_cold2 = np.sum(behavior_dist2[warmth_scores < 0])  # P(agent2 cold)
    base_communion_pct = (p_warm1 * p_warm2 + p_cold1 * p_cold2) * 100

    # Agency matching: dominance-submissive complementarity
    agency1_per_round = [agency_scores[a] for a in all_actions1]
    agency2_per_round = [agency_scores[a] for a in all_actions2]

    # Opposite-sign agency (one dominant, one submissive)
    agency_matches = sum(
        1
        for a1, a2 in zip(agency1_per_round, agency2_per_round)
        if (a1 > 0 and a2 < 0) or (a1 < 0 and a2 > 0)
    )
    agency_match_pct = agency_matches / len(all_actions1) * 100

    # Base rate for agency complementarity: Given ACTUAL distributions, expected random complementarity
    # P(agent1 dom & agent2 sub) + P(agent1 sub & agent2 dom) if independent
    p_dom1 = np.sum(behavior_dist1[agency_scores > 0])  # P(agent1 dominant)
    p_sub1 = np.sum(behavior_dist1[agency_scores < 0])  # P(agent1 submissive)
    p_dom2 = np.sum(behavior_dist2[agency_scores > 0])  # P(agent2 dominant)
    p_sub2 = np.sum(behavior_dist2[agency_scores < 0])  # P(agent2 submissive)
    base_agency_pct = (p_dom1 * p_sub2 + p_sub1 * p_dom2) * 100

    results = {
        "behavior_dist1": behavior_dist1,
        "behavior_dist2": behavior_dist2,
        "avg_warmth1": avg_warmth1,
        "avg_warmth2": avg_warmth2,
        "avg_agency1": avg_agency1,
        "avg_agency2": avg_agency2,
        "base_warmth": base_warmth,
        "base_agency": base_agency,
        "communion_match_pct": communion_match_pct,
        "base_communion_pct": base_communion_pct,
        "communion_above_base": communion_match_pct - base_communion_pct,
        "agency_match_pct": agency_match_pct,
        "base_agency_pct": base_agency_pct,
        "agency_above_base": agency_match_pct - base_agency_pct,
        "all_stats": all_stats,
        "n_interactions": len(all_actions1),
    }

    return results


def plot_behavior_analysis(
    results: dict,
    agent1_needs_name: str,
    agent2_needs_name: str,
):
    """Plot behavior analysis results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Actual behavior distributions
    ax1 = axes[0, 0]
    x = np.arange(8)
    width = 0.35
    ax1.bar(
        x - width / 2,
        results["behavior_dist1"],
        width,
        label=f"Agent1 ({agent1_needs_name})",
        alpha=0.7,
    )
    ax1.bar(
        x + width / 2,
        results["behavior_dist2"],
        width,
        label=f"Agent2 ({agent2_needs_name})",
        alpha=0.7,
    )
    ax1.axhline(y=1 / 8, color="gray", linestyle="--", alpha=0.5, label="Uniform (1/8)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(OCTANT_ORDER)
    ax1.set_ylabel("Proportion of Actions")
    ax1.set_title("Actual Behavior Distributions")
    ax1.legend()
    ax1.set_ylim(
        0, max(results["behavior_dist1"].max(), results["behavior_dist2"].max()) * 1.2
    )

    # 2. Warmth and Agency comparison
    ax2 = axes[0, 1]
    categories = [
        "Warmth\n(Agent1)",
        "Warmth\n(Agent2)",
        "Agency\n(Agent1)",
        "Agency\n(Agent2)",
    ]
    values = [
        results["avg_warmth1"],
        results["avg_warmth2"],
        results["avg_agency1"],
        results["avg_agency2"],
    ]
    base_values = [
        results["base_warmth"],
        results["base_warmth"],
        results["base_agency"],
        results["base_agency"],
    ]
    colors = ["green" if v > 0 else "red" for v in values]

    bars = ax2.bar(categories, values, color=colors, alpha=0.7, edgecolor="black")
    ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax2.scatter(
        categories,
        base_values,
        color="gray",
        marker="_",
        s=200,
        linewidths=3,
        label="Base rate",
        zorder=5,
    )
    ax2.set_ylabel("Score (-1 to +1)")
    ax2.set_title("Interpersonal Style\n(Green=Warm/Dominant, Red=Cold/Submissive)")
    ax2.legend()

    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax2.annotate(
            f"{val:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3 if height >= 0 else -12),
            textcoords="offset points",
            ha="center",
            va="bottom" if height >= 0 else "top",
            fontsize=10,
            fontweight="bold",
        )

    # 3. Circumplex matching percentages
    ax3 = axes[1, 0]
    match_types = ["Communion\n(Warmth↔Warmth)", "Agency\n(Dom↔Sub)"]
    observed = [results["communion_match_pct"], results["agency_match_pct"]]
    baselines = [results["base_communion_pct"], results["base_agency_pct"]]
    above_base = [results["communion_above_base"], results["agency_above_base"]]

    x_pos = np.arange(len(match_types))
    width = 0.35

    bars1 = ax3.bar(
        x_pos - width / 2,
        observed,
        width,
        label="Observed",
        color="steelblue",
        alpha=0.8,
    )
    bars2 = ax3.bar(
        x_pos + width / 2, baselines, width, label="Base rate", color="gray", alpha=0.6
    )

    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(match_types)
    ax3.set_ylabel("Match Percentage (%)")
    ax3.set_title("Circumplex Pattern Matching")
    ax3.legend()

    # Add value labels
    for bar, val, ab in zip(bars1, observed, above_base):
        height = bar.get_height()
        sign = "+" if ab >= 0 else ""
        ax3.annotate(
            f"{val:.1f}%\n({sign}{ab:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # 4. Summary text
    ax4 = axes[1, 1]
    ax4.axis("off")

    summary_text = f"""
BEHAVIOR ANALYSIS SUMMARY
{'='*40}

Total Interactions: {results['n_interactions']:,}

INTERPERSONAL STYLE (relative to base rate 0.0):
  Agent1 ({agent1_needs_name}):
    Warmth:  {results['avg_warmth1']:+.3f}  ({"warm" if results['avg_warmth1'] > 0 else "cold"})
    Agency:  {results['avg_agency1']:+.3f}  ({"dominant" if results['avg_agency1'] > 0 else "submissive"})

  Agent2 ({agent2_needs_name}):
    Warmth:  {results['avg_warmth2']:+.3f}  ({"warm" if results['avg_warmth2'] > 0 else "cold"})
    Agency:  {results['avg_agency2']:+.3f}  ({"dominant" if results['avg_agency2'] > 0 else "submissive"})

CIRCUMPLEX MATCHING (vs. base rate):
  Communion (warmth↔warmth):
    Observed: {results['communion_match_pct']:.1f}%
    Base:     {results['base_communion_pct']:.1f}%
    Above base: {results['communion_above_base']:+.1f}%

  Agency (dominant↔submissive):
    Observed: {results['agency_match_pct']:.1f}%
    Base:     {results['base_agency_pct']:.1f}%
    Above base: {results['agency_above_base']:+.1f}%

INTERPRETATION:
  {"✓ Warmth reciprocity above chance" if results['communion_above_base'] > 0 else "✗ Warmth reciprocity below chance"}
  {"✓ Dominance complementarity above chance" if results['agency_above_base'] > 0 else "✗ Dominance complementarity below chance"}
"""

    ax4.text(
        0.05,
        0.95,
        summary_text,
        transform=ax4.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig("behavior_analysis_results.png", dpi=150, bbox_inches="tight")
    plt.show()

    return fig


if __name__ == "__main__":
    # Configuration - set agent needs here
    AGENT1_NEEDS = "LM"  # Agent1's needs (what they want from others)
    AGENT2_NEEDS = "LM"  # Agent2's needs (what they want from others)
    BOTH_AGENTS_LEARN = (
        False  # If True, both agents update their books during simulation
    )

    # Run tournament
    history, final_population, fixed_agent2 = run_book_tournament(
        population_size=10,
        generations=8,
        n_relationships=5,
        rounds_per_rel=300,
        survival_ratio=0.4,
        mutation_rate=0.01,
        agent1_needs=AGENT1_NEEDS,
        fixed_agent2_needs=AGENT2_NEEDS,
        both_agents_learn=BOTH_AGENTS_LEARN,
        verbose=True,
    )

    # Re-evaluate final population to get accurate fitness for display
    for genome in final_population:
        evaluate_genome(
            genome,
            fixed_agent2,
            n_relationships=10,
            rounds_per_rel=300,
            agent1_needs=AGENT1_NEEDS,
            both_agents_learn=BOTH_AGENTS_LEARN,
        )

    # Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    best = max(final_population, key=lambda g: g.fitness)
    print(f"\nBest evolved book: {best.book_name}")
    print(f"  Distribution: {[f'{v:.3f}' for v in best.book_array]}")
    print(f"  Fitness (communal): {best.fitness:.4f}")

    print(f"\nAgent2's needs ({fixed_agent2.self_concept_name}):")
    print(f"  {[f'{v:.3f}' for v in fixed_agent2.self_concept]}")

    # How well does evolved book match Agent2's needs?
    match = np.sum(best.book_array * fixed_agent2.self_concept)
    print(f"\nEvolved book alignment with Agent2 needs: {match:.4f}")

    # Compare to uniform
    uniform_match = np.sum(np.full(8, 1 / 8) * fixed_agent2.self_concept)
    print(f"Uniform book alignment with Agent2 needs: {uniform_match:.4f}")
    print(f"Improvement: {(match - uniform_match) / uniform_match * 100:.1f}%")

    print("\nTop 5 books in final population:")
    top5 = sorted(final_population, key=lambda g: g.fitness, reverse=True)[:5]
    for i, g in enumerate(top5, 1):
        peak_idx = np.argmax(g.book_array)
        print(
            f"  {i}. {g.book_name:10s} | peak={OCTANT_ORDER[peak_idx]} | "
            f"fitness={g.fitness:.4f}"
        )

    # Plot evolution
    print("\nPlotting evolution...")
    plot_evolution(history, best, fixed_agent2, agent1_needs_name=AGENT1_NEEDS)

    # Analyze and plot actual behavior patterns
    print("\nAnalyzing behavior patterns...")
    behavior_results = analyze_behavior_patterns(
        best,
        fixed_agent2,
        agent1_needs=AGENT1_NEEDS,
        both_agents_learn=BOTH_AGENTS_LEARN,
        n_relationships=20,
        rounds_per_rel=400,
    )
    plot_behavior_analysis(behavior_results, AGENT1_NEEDS, AGENT2_NEEDS)
