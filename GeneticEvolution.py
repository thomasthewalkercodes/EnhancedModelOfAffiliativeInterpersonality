"""
Genetic Algorithm for evolving agent parameters.

This module implements a genetic algorithm that evolves populations of agents
over multiple generations to find optimal parameter combinations.
"""

import numpy as np
import csv
import os
from pathlib import Path
from EvolutionaryAgent import EvolutionaryAgent
from AmbiguousQLearning import simulate_ambiguous


class GeneticEvolution:
    """
    Genetic algorithm for evolving agent parameters.

    Evolution process:
    1. Create initial population with diverse parameters
    2. Each agent plays against the opponent agent
    3. Agents accumulate fitness (total points)
    4. Select top performers
    5. Create next generation from top performers with mutations
    6. Repeat for N generations
    """

    def __init__(
        self,
        population_size,
        num_generations,
        selection_rate,
        mutation_rate,
        rounds_per_match,
        ambiguous_freq,
        matches_per_evaluation,
        A1, A2,
        p_init, q_init,
        opponent_alpha,
        opponent_beta,
        opponent_trust,
        results_folder="evolution_results"
    ):
        """
        Initialize the genetic evolution system.

        Args:
            population_size: Number of agents per generation
            num_generations: How many generations to evolve
            selection_rate: Fraction of top agents to keep (e.g., 0.2 = top 20%)
            mutation_rate: How much parameters mutate (0 to 1)
            rounds_per_match: How many rounds each agent plays against opponent
            ambiguous_freq: Frequency of ambiguous situations
            matches_per_evaluation: How many matches each agent plays (mean is used for fitness)
            A1, A2: Payoff matrices
            p_init, q_init: Initial probabilities from Nash equilibrium
            opponent_alpha, opponent_beta, opponent_trust: Fixed opponent parameters
            results_folder: Where to save results
        """
        self.population_size = population_size
        self.num_generations = num_generations
        self.selection_rate = selection_rate
        self.mutation_rate = mutation_rate
        self.rounds_per_match = rounds_per_match
        self.ambiguous_freq = ambiguous_freq
        self.matches_per_evaluation = matches_per_evaluation

        self.A1 = A1
        self.A2 = A2
        self.p_init = p_init
        self.q_init = q_init

        self.opponent_alpha = opponent_alpha
        self.opponent_beta = opponent_beta
        self.opponent_trust = opponent_trust

        self.results_folder = results_folder
        self.num_survivors = max(1, int(population_size * selection_rate))

        # Storage
        self.population = []
        self.all_agents_history = []  # All agents across all generations
        self.generation_stats = []  # Stats per generation

        # Create results folder
        Path(self.results_folder).mkdir(exist_ok=True)

    def initialize_population(self):
        """Create initial generation with random, diverse parameters."""
        print(f"Initializing population of {self.population_size} agents...")
        EvolutionaryAgent.reset_id_counter()
        self.population = [
            EvolutionaryAgent.create_random(generation=0)
            for _ in range(self.population_size)
        ]

    def evaluate_agent(self, agent):
        """
        Evaluate an agent by having it play multiple matches against the fixed opponent.

        The agent plays as Agent1, opponent plays as Agent2.
        Each agent plays multiple matches and the mean score is used for fitness.
        This reduces variance and provides more robust evaluation.
        """
        match_scores = []

        for _ in range(self.matches_per_evaluation):
            result = simulate_ambiguous(
                self.A1, self.A2,
                self.p_init, self.q_init,
                agent.alpha, agent.beta,
                self.opponent_alpha, self.opponent_beta,
                agent.trust, self.opponent_trust,
                self.ambiguous_freq,
                self.rounds_per_match
            )

            # Total points from this match
            total_points = np.sum(result["payoff_history1"])
            match_scores.append(total_points)

        # Fitness = mean of all matches
        mean_fitness = np.mean(match_scores)
        agent.add_fitness(mean_fitness)

        return mean_fitness

    def evaluate_generation(self, generation_num):
        """Evaluate all agents in the current generation."""
        print(f"\nGeneration {generation_num}: Evaluating {len(self.population)} agents...")

        for i, agent in enumerate(self.population):
            points = self.evaluate_agent(agent)
            if (i + 1) % 10 == 0 or (i + 1) == len(self.population):
                print(f"  Evaluated {i + 1}/{len(self.population)} agents...", end='\r')

        print(f"  Evaluated {len(self.population)}/{len(self.population)} agents [DONE]")

        # Sort by fitness (descending)
        self.population.sort(key=lambda a: a.fitness, reverse=True)

        # Calculate statistics
        fitnesses = [a.fitness for a in self.population]
        stats = {
            'generation': generation_num,
            'mean_fitness': np.mean(fitnesses),
            'max_fitness': np.max(fitnesses),
            'min_fitness': np.min(fitnesses),
            'std_fitness': np.std(fitnesses),
            'best_alpha': self.population[0].alpha,
            'best_beta': self.population[0].beta,
            'best_trust': self.population[0].trust
        }
        self.generation_stats.append(stats)

        # Print generation summary
        print(f"\n  Generation {generation_num} Results:")
        print(f"    Best fitness:  {stats['max_fitness']:.1f}")
        print(f"    Mean fitness:  {stats['mean_fitness']:.1f}")
        print(f"    Worst fitness: {stats['min_fitness']:.1f}")
        print(f"    Best agent: alpha={stats['best_alpha']:.3f}, "
              f"beta={stats['best_beta']:.3f}, trust={stats['best_trust']:.3f}")

        # Save all agents to history
        self.all_agents_history.extend([a.to_dict() for a in self.population])

    def select_and_reproduce(self, generation_num):
        """
        Select top performers and create next generation.

        Selection: Keep top selection_rate agents
        Reproduction: Each survivor creates multiple offspring to fill population
        """
        # Select survivors
        survivors = self.population[:self.num_survivors]

        print(f"\n  Selecting top {self.num_survivors} agents for reproduction...")

        # Create next generation
        next_generation = []

        # Calculate how many offspring each survivor should have
        offspring_per_survivor = self.population_size // self.num_survivors
        remainder = self.population_size % self.num_survivors

        for i, parent in enumerate(survivors):
            # Number of children for this parent
            num_children = offspring_per_survivor
            if i < remainder:  # Distribute remainder
                num_children += 1

            # Create offspring
            for _ in range(num_children):
                child = parent.create_offspring(
                    generation=generation_num + 1,
                    mutation_rate=self.mutation_rate
                )
                next_generation.append(child)

        self.population = next_generation

    def run_evolution(self):
        """Run the complete evolutionary process."""
        print("="*70)
        print("STARTING GENETIC EVOLUTION")
        print("="*70)
        print(f"\nParameters:")
        print(f"  Population size:      {self.population_size}")
        print(f"  Generations:          {self.num_generations}")
        print(f"  Selection rate:       {self.selection_rate} ({self.num_survivors} survivors)")
        print(f"  Mutation rate:        {self.mutation_rate}")
        print(f"  Matches per agent:    {self.matches_per_evaluation}")
        print(f"  Rounds per match:     {self.rounds_per_match}")
        print(f"  Ambiguous freq:       {self.ambiguous_freq}")
        print(f"\nOpponent parameters:")
        print(f"  Alpha: {self.opponent_alpha}")
        print(f"  Beta:  {self.opponent_beta}")
        print(f"  Trust: {self.opponent_trust}")

        # Initialize population
        self.initialize_population()

        # Evolution loop
        for generation in range(self.num_generations):
            # Evaluate current generation
            self.evaluate_generation(generation)

            # Create next generation (except for last generation)
            if generation < self.num_generations - 1:
                self.select_and_reproduce(generation)

        print("\n" + "="*70)
        print("EVOLUTION COMPLETE")
        print("="*70)

        # Save results
        self.save_results()

    def save_results(self):
        """Save all results to CSV files."""
        print(f"\nSaving results to '{self.results_folder}/'...")

        # 1. Save all agents history
        agents_file = os.path.join(self.results_folder, "all_agents.csv")
        with open(agents_file, 'w', newline='') as f:
            fieldnames = ['agent_id', 'generation', 'parent_id', 'alpha', 'beta', 'trust', 'fitness']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.all_agents_history)
        print(f"  [OK] All agents saved to: {agents_file}")

        # 2. Save generation statistics
        stats_file = os.path.join(self.results_folder, "generation_stats.csv")
        with open(stats_file, 'w', newline='') as f:
            fieldnames = ['generation', 'mean_fitness', 'max_fitness', 'min_fitness',
                         'std_fitness', 'best_alpha', 'best_beta', 'best_trust']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.generation_stats)
        print(f"  [OK] Generation stats saved to: {stats_file}")

        # 3. Save top performers from final generation
        final_generation = [a for a in self.all_agents_history
                          if a['generation'] == self.num_generations - 1]
        final_generation.sort(key=lambda x: x['fitness'], reverse=True)

        top_performers_file = os.path.join(self.results_folder, "top_performers.csv")
        with open(top_performers_file, 'w', newline='') as f:
            fieldnames = ['rank', 'agent_id', 'alpha', 'beta', 'trust', 'fitness']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rank, agent in enumerate(final_generation[:20], 1):  # Top 20
                agent['rank'] = rank
                writer.writerow({k: agent[k] for k in fieldnames})
        print(f"  [OK] Top 20 performers saved to: {top_performers_file}")

        # 4. Print summary
        print(f"\n{'='*70}")
        print("TOP 5 BEST PERFORMING PARAMETER COMBINATIONS:")
        print(f"{'='*70}")
        print(f"{'Rank':<6} {'Alpha':<8} {'Beta':<8} {'Trust':<8} {'Fitness':<10}")
        print("-"*70)
        for rank, agent in enumerate(final_generation[:5], 1):
            print(f"{rank:<6} {agent['alpha']:<8.3f} {agent['beta']:<8.3f} "
                  f"{agent['trust']:<8.3f} {agent['fitness']:<10.1f}")
        print("="*70)
