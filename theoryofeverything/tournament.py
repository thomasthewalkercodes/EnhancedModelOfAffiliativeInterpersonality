"""
Evolutionary tournament for interpersonal agents.

Agents evolve their:
- Self-concept (book)
- Tolerance for mismatch
- Sensitivity to deviations

Fitness = weighted combination of payoff + relationship survival.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple
import random

from better_game_engine import simulation, get_current_book
from motive_system import create_motive_matrices, OCTANT_ORDER


@dataclass
class AgentGenome:
    """Evolvable traits of an agent."""
    book_name: str = "uniform"
    amplitude: float = 1.0
    tolerance: float = 1.0
    sensitivity: float = 1.0

    # Fitness tracking (not inherited)
    total_payoff: float = field(default=0.0, repr=False)
    total_rounds: int = field(default=0, repr=False)
    relationships_completed: int = field(default=0, repr=False)
    relationships_aborted: int = field(default=0, repr=False)

    @property
    def fitness(self):
        """Combined fitness: payoff + survival."""
        if self.total_rounds == 0:
            return 0.0
        avg_payoff = self.total_payoff / self.total_rounds
        survival_rate = (
            self.relationships_completed /
            max(1, self.relationships_completed + self.relationships_aborted)
        )
        return avg_payoff + survival_rate

    def reset_fitness(self):
        self.total_payoff = 0.0
        self.total_rounds = 0
        self.relationships_completed = 0
        self.relationships_aborted = 0


def mutate_genome(genome: AgentGenome, mutation_rate: float = 0.1) -> AgentGenome:
    """Create mutated offspring from a genome."""
    new_genome = AgentGenome(
        book_name=genome.book_name,
        amplitude=genome.amplitude,
        tolerance=genome.tolerance,
        sensitivity=genome.sensitivity,
    )

    # Mutate book_name
    if random.random() < mutation_rate:
        new_genome.book_name = random.choice(OCTANT_ORDER + ["uniform"])

    # Mutate continuous parameters with Gaussian noise
    if random.random() < mutation_rate:
        new_genome.amplitude = max(0.1, genome.amplitude + np.random.normal(0, 0.2))

    if random.random() < mutation_rate:
        new_genome.tolerance = max(0.1, genome.tolerance + np.random.normal(0, 0.2))

    if random.random() < mutation_rate:
        new_genome.sensitivity = max(0.1, genome.sensitivity + np.random.normal(0, 0.3))

    return new_genome


def run_interaction(
    genome1: AgentGenome, genome2: AgentGenome,
    base_payoff: np.ndarray, rounds: int,
    l_rate: float, e_rate: float, update_interval: int
) -> Tuple[float, float, bool]:
    """
    Run single interaction between two genomes.
    Returns: (payoff1, payoff2, completed)
    """
    n_actions = base_payoff.shape[0]
    motive_matrices = create_motive_matrices(base_payoff, n_actions)

    result = simulation(
        P1=base_payoff, P2=base_payoff.T,
        l_rate=l_rate, e_rate=e_rate, rounds=rounds,
        book1=genome1.book_name, book2=genome2.book_name,
        tolerance1=genome1.tolerance, tolerance2=genome2.tolerance,
        motive_matrices1=motive_matrices, motive_matrices2=motive_matrices,
        update_interval=update_interval, sensitivity=genome1.sensitivity
    )

    abort = result["abort_round"]
    actual_rounds = abort if abort else rounds

    payoff1 = np.sum(result["payoff_history1"][:actual_rounds])
    payoff2 = np.sum(result["payoff_history2"][:actual_rounds])
    completed = abort is None

    return payoff1, payoff2, completed, actual_rounds


def run_generation(
    population: List[AgentGenome],
    base_payoff: np.ndarray,
    interactions_per_agent: int,
    rounds_per_interaction: int,
    l_rate: float,
    e_rate: float,
    update_interval: int
):
    """Run one generation: random pairings, accumulate fitness."""
    # Reset fitness
    for genome in population:
        genome.reset_fitness()

    # Random pairings
    for _ in range(interactions_per_agent):
        # Shuffle and pair
        indices = list(range(len(population)))
        random.shuffle(indices)

        for i in range(0, len(indices) - 1, 2):
            g1 = population[indices[i]]
            g2 = population[indices[i + 1]]

            p1, p2, completed, actual_rounds = run_interaction(
                g1, g2, base_payoff, rounds_per_interaction,
                l_rate, e_rate, update_interval
            )

            # Update fitness
            g1.total_payoff += p1
            g2.total_payoff += p2
            g1.total_rounds += actual_rounds
            g2.total_rounds += actual_rounds

            if completed:
                g1.relationships_completed += 1
                g2.relationships_completed += 1
            else:
                g1.relationships_aborted += 1
                g2.relationships_aborted += 1


def evolve_population(
    population: List[AgentGenome],
    survival_ratio: float,
    mutation_rate: float
) -> List[AgentGenome]:
    """Select top agents, create offspring with mutations."""
    # Sort by fitness
    population.sort(key=lambda g: g.fitness, reverse=True)

    # Survivors
    n_survivors = max(2, int(len(population) * survival_ratio))
    survivors = population[:n_survivors]

    # Create new population
    new_population = []
    while len(new_population) < len(population):
        parent = random.choice(survivors)
        offspring = mutate_genome(parent, mutation_rate)
        new_population.append(offspring)

    return new_population


def run_tournament(config: dict) -> List[dict]:
    """
    Run full evolutionary tournament.

    Returns history of each generation's stats.
    """
    # Initialize population
    population = []
    for _ in range(config["population_size"]):
        genome = AgentGenome(
            book_name=random.choice(OCTANT_ORDER + ["uniform"]),
            amplitude=random.uniform(0.5, 1.5),
            tolerance=random.uniform(0.5, 2.0),
            sensitivity=random.uniform(0.5, 2.0),
        )
        population.append(genome)

    history = []

    for gen in range(config["generations"]):
        # Run generation
        run_generation(
            population,
            config["base_payoff"],
            config["interactions_per_agent"],
            config["rounds_per_interaction"],
            config["l_rate"],
            config["e_rate"],
            config["update_interval"]
        )

        # Record stats
        fitnesses = [g.fitness for g in population]
        books = [g.book_name for g in population]
        book_counts = {b: books.count(b) for b in set(books)}

        gen_stats = {
            "generation": gen,
            "mean_fitness": np.mean(fitnesses),
            "max_fitness": np.max(fitnesses),
            "min_fitness": np.min(fitnesses),
            "book_distribution": book_counts,
            "mean_tolerance": np.mean([g.tolerance for g in population]),
            "mean_sensitivity": np.mean([g.sensitivity for g in population]),
            "best_genome": max(population, key=lambda g: g.fitness),
        }
        history.append(gen_stats)

        if config.get("verbose", False):
            print(f"Gen {gen}: fitness={gen_stats['mean_fitness']:.3f}, "
                  f"books={book_counts}")

        # Evolve
        population = evolve_population(
            population,
            config["survival_ratio"],
            config["mutation_rate"]
        )

    return history, population


# Default configuration
DEFAULT_CONFIG = {
    "population_size": 20,
    "generations": 50,
    "interactions_per_agent": 5,
    "rounds_per_interaction": 100,
    "l_rate": 0.1,
    "e_rate": 5.0,
    "update_interval": 10,
    "survival_ratio": 0.3,
    "mutation_rate": 0.2,
    "base_payoff": None,  # Must be set
    "verbose": True,
}


if __name__ == "__main__":
    # Example: run tournament with simple payoff matrix
    n = 8
    # Complementarity matrix: reward when actions match circumplex neighbors
    base = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist = min(abs(i - j), n - abs(i - j))
            base[i, j] = 1.0 - dist / (n / 2)

    config = DEFAULT_CONFIG.copy()
    config["base_payoff"] = base

    history, final_pop = run_tournament(config)

    print("\n=== Final Population ===")
    for g in sorted(final_pop, key=lambda x: x.fitness, reverse=True)[:5]:
        print(f"  {g.book_name}: tol={g.tolerance:.2f}, "
              f"sens={g.sensitivity:.2f}, fit={g.fitness:.3f}")
