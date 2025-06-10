from typing import Callable

import numpy as np
from sklearn.base import BaseEstimator

from feature_selection.core.base import BaseFeatureSelector
from feature_selection.schemas.history import FeatureSet, Iteration, History


def average_hamming_distance(population: np.ndarray) -> float:
    n = len(population)
    if n < 2:
        return 0.0
    total_distance = 0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total_distance += np.sum(population[i] != population[j])
            count += 1
    return total_distance / count if count > 0 else 0.0


class GeneticFeatureSelector(BaseFeatureSelector):
    def __init__(
            self,
            estimator: BaseEstimator,
            scoring: str | Callable,
            fitness_function: Callable,
            population_size: int = 16,
            max_iterations: int = 25,
            mutation_probability: float = 0.05,
            crossover_method: str = 'random',
            selection_method: str = 'elitism',
            cv: int = 5,
            tournament_size: int = 3,
            max_iteration_without_improvement: int = 10,
    ):
        super().__init__(
            estimator=estimator,
            scoring=scoring,
            cv=cv
        )
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.crossover_method = crossover_method
        self.mutation_probability = mutation_probability
        self.fitness_function = fitness_function
        self.selection_method = selection_method
        self.tournament_size = tournament_size
        self.max_iteration_without_improvement = max_iteration_without_improvement

    def _generate_population(self, n_features: int) -> np.ndarray:
        return np.random.random((self.population_size, n_features)) > 0.5

    def _elitism_selection(self, population: np.ndarray,
                           fitness: np.ndarray) -> tuple[
        np.ndarray, np.ndarray]:
        best_indexes = np.argsort(fitness)[::-1][:self.population_size]
        return population[best_indexes], fitness[best_indexes]

    def _tournament_selection(self, population: np.ndarray,
                              fitness: np.ndarray) -> tuple[
        np.ndarray, np.ndarray]:
        new_population = []
        new_fitness = []
        population_count = len(population)

        for _ in range(self.population_size):
            replace = self.tournament_size > population_count
            contenders = np.random.choice(
                population_count,
                self.tournament_size,
                replace=replace
            )
            best_idx = contenders[np.argmax(fitness[contenders])]
            new_population.append(population[best_idx])
            new_fitness.append(fitness[best_idx])

        return np.array(new_population), np.array(new_fitness)

    def _selection(self, population: np.ndarray, fitness: np.ndarray) -> tuple[
        np.ndarray, np.ndarray]:
        if self.selection_method == "elitism":
            return self._elitism_selection(population, fitness)
        elif self.selection_method == "tournament":
            return self._tournament_selection(population, fitness)
        return population, fitness

    def _mutation(self, feature_mask: np.ndarray) -> np.ndarray:
        mutated_mask = feature_mask.copy()
        mutation_mask = np.random.random(
            size=feature_mask.shape) < self.mutation_probability
        mutated_mask[mutation_mask] = ~mutated_mask[mutation_mask]
        return mutated_mask

    def _crossover(self, parent1: np.ndarray,
                   parent2: np.ndarray) -> np.ndarray:
        if self.crossover_method == 'one_point':
            crossover_point = np.random.randint(1, len(parent1) - 1)
            child = np.concatenate(
                [parent1[:crossover_point], parent2[crossover_point:]])
        else:  # random
            mask = np.random.random(size=len(parent1)) > 0.5
            child = np.where(mask, parent1, parent2)
        return self._mutation(child)

    def _calculate_fitness(
            self,
            population: np.ndarray,
            X: np.ndarray,
            y: np.ndarray,
            cache: dict
    ) -> np.ndarray:
        fitness_values = []
        for individual in population:
            score = self._estimate_algorithm_on_subset(
                features_subset_mask=individual,
                X=X,
                y=y,
                cache=cache
            )
            fitness_values.append(self.fitness_function(score, individual))
        return np.array(fitness_values)

    def _store_iteration(
            self,
            iteration: int,
            population: np.ndarray,
            fitness: np.ndarray
    ) -> None:
        # Create FeatureSet objects for all candidates
        candidate_features = []
        for mask, fitness_val in zip(population, fitness):
            features = set(np.where(mask)[0])
            candidate_features.append(
                FeatureSet(
                    features=features,
                    score=fitness_val,
                    original_feature_num=len(mask)
                )
            )

        # Find best candidate for this iteration
        best_idx = np.argmax(fitness)
        best_mask = population[best_idx]
        best_features = set(np.where(best_mask)[0])
        selected_features = FeatureSet(
            features=best_features,
            score=fitness[best_idx],
            original_feature_num=len(best_mask)
        )

        # Prepare metadata
        metadata = {
            'avg_fitness': np.mean(fitness),
            'avg_hamming': average_hamming_distance(population),
            'selected_features_count': np.sum(best_mask),
        }

        # Create and store iteration
        iteration_data = Iteration(
            iteration=iteration,
            candidate_features=candidate_features,
            selected_features=selected_features,
            metadata=metadata
        )
        self.history.add_iteration(iteration_data)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.history = History()
        n, n_features = X.shape
        cache = {}
        best_global_score = -np.inf
        best_global_features = None

        # Generate initial population
        population = self._generate_population(n_features)
        fitness = self._calculate_fitness(population, X, y, cache)
        self._store_iteration(0, population, fitness)

        # Track best global solution
        current_best_idx = np.argmax(fitness)
        if fitness[current_best_idx] > best_global_score:
            best_global_score = fitness[current_best_idx]
            best_global_features = population[current_best_idx]

        # Evolution loop
        iteration_without_improvement = 0
        early_stopping = False

        for iteration in range(1, self.max_iterations + 1):
            # Create new population
            new_population = []
            for _ in range(self.population_size):
                idx1, idx2 = np.random.choice(len(population), 2,
                                              replace=False)
                child = self._crossover(population[idx1], population[idx2])
                new_population.append(child)

            # Evaluate new individuals
            new_population = np.array(new_population)
            new_fitness = self._calculate_fitness(new_population, X, y, cache)

            # Combine populations
            combined_population = np.vstack([population, new_population])
            combined_fitness = np.concatenate([fitness, new_fitness])

            # Selection
            population, fitness = self._selection(combined_population,
                                                  combined_fitness)
            self._store_iteration(iteration, population, fitness)

            # Update best solution
            current_best_idx = np.argmax(fitness)
            current_best_score = fitness[current_best_idx]

            if current_best_score > best_global_score:
                best_global_score = current_best_score
                best_global_features = population[current_best_idx]
                iteration_without_improvement = 0
            else:
                iteration_without_improvement += 1
                if iteration_without_improvement >= self.max_iteration_without_improvement:
                    early_stopping = True
                    break

        # Save final feature set
        best_features = set(np.where(best_global_features)[0])
        self.selected_features = FeatureSet(
            features=best_features,
            score=best_global_score,
            original_feature_num=len(best_global_features)
        )

        # Store early stopping info in metadata
        if early_stopping:
            self.history.iterations[-1].metadata['early_stopping'] = True

        return self
