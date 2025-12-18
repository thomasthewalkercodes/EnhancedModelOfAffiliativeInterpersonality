"""
Evolutionary Agent class for genetic algorithm-based parameter optimization.

Each agent has parameters (alpha, beta, trust) that evolve over generations.
"""

import numpy as np


class EvolutionaryAgent:
    """
    Represents an agent with evolvable parameters.

    Attributes:
        alpha: Learning rate (0.01 to 4.0)
        beta: Softmax temperature (1.0 to 8.0)
        trust: Trust score (0.0 to 1.0)
        fitness: Total points accumulated in the current generation
        generation: Which generation this agent was born in
        parent_id: ID of parent agent (None for generation 0)
        agent_id: Unique identifier
    """

    # Parameter bounds
    ALPHA_MIN, ALPHA_MAX = 0.01, 4.0
    BETA_MIN, BETA_MAX = 1.0, 8.0
    TRUST_MIN, TRUST_MAX = 0.0, 1.0

    _id_counter = 0  # Class variable for unique IDs

    def __init__(self, alpha, beta, trust, generation=0, parent_id=None):
        """Initialize an agent with specific parameters."""
        self.alpha = np.clip(alpha, self.ALPHA_MIN, self.ALPHA_MAX)
        self.beta = np.clip(beta, self.BETA_MIN, self.BETA_MAX)
        self.trust = np.clip(trust, self.TRUST_MIN, self.TRUST_MAX)
        self.fitness = 0.0
        self.generation = generation
        self.parent_id = parent_id

        # Assign unique ID
        self.agent_id = EvolutionaryAgent._id_counter
        EvolutionaryAgent._id_counter += 1

    @classmethod
    def create_random(cls, generation=0):
        """Create an agent with random parameters."""
        alpha = np.random.uniform(cls.ALPHA_MIN, cls.ALPHA_MAX)
        beta = np.random.uniform(cls.BETA_MIN, cls.BETA_MAX)
        trust = np.random.uniform(cls.TRUST_MIN, cls.TRUST_MAX)
        return cls(alpha, beta, trust, generation=generation)

    def create_offspring(self, generation, mutation_rate=0.1):
        """
        Create a child agent with slightly mutated parameters.

        Args:
            generation: Generation number of the offspring
            mutation_rate: How much parameters can change (0 to 1)
                          Lower = smaller mutations, Higher = larger mutations

        Returns:
            New EvolutionaryAgent with mutated parameters
        """
        # Mutation ranges (as fraction of parameter range)
        alpha_range = self.ALPHA_MAX - self.ALPHA_MIN
        beta_range = self.BETA_MAX - self.BETA_MIN
        trust_range = self.TRUST_MAX - self.TRUST_MIN

        # Add gaussian noise proportional to mutation rate
        new_alpha = self.alpha + np.random.normal(0, mutation_rate * alpha_range * 0.1)
        new_beta = self.beta + np.random.normal(0, mutation_rate * beta_range * 0.1)
        new_trust = self.trust + np.random.normal(0, mutation_rate * trust_range * 0.1)

        return EvolutionaryAgent(
            new_alpha, new_beta, new_trust,
            generation=generation,
            parent_id=self.agent_id
        )

    def add_fitness(self, points):
        """Add points to this agent's fitness score."""
        self.fitness += points

    def reset_fitness(self):
        """Reset fitness to 0 (for new generation)."""
        self.fitness = 0.0

    def to_dict(self):
        """Convert agent to dictionary for CSV export."""
        return {
            'agent_id': self.agent_id,
            'generation': self.generation,
            'parent_id': self.parent_id,
            'alpha': self.alpha,
            'beta': self.beta,
            'trust': self.trust,
            'fitness': self.fitness
        }

    def __repr__(self):
        return (f"Agent(id={self.agent_id}, gen={self.generation}, "
                f"alpha={self.alpha:.3f}, beta={self.beta:.3f}, "
                f"trust={self.trust:.3f}, fitness={self.fitness:.1f})")

    @classmethod
    def reset_id_counter(cls):
        """Reset the ID counter (useful for new evolution runs)."""
        cls._id_counter = 0
