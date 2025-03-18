from functools import partial
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

from feature_selection import BaseFeatureSelector


class GeneticFeatureSelector(BaseFeatureSelector):

    def __init__(
            self,
            population_size: int,
            max_iterations: int,
            crossover_method: str,
            mutation_probability: float,
            algorithm: Any,
            algorithm_metric: Any,
            fitness_function: Any,
    ):
        super().__init__()
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.crossover_method = crossover_method
        self.mutation_probability = mutation_probability
        self.algorithm = algorithm
        self.algorithm_metric = algorithm_metric
        self.fitness_function = fitness_function
        self.feature_information = None

        self.population = []
        self.history = []
        self.fitness = []

        self.X = None
        self.y = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.feature_dim = None

    def _generate_population(self):
        self.population = []
        for _ in range(self.population_size):
            self.population.append(
                np.random.random(size=self.feature_dim) > 0.5
            )
        self.population = np.array(self.population)

    def _selection(self):
        best_indexes = np.argsort(self.fitness)
        best_indexes = best_indexes[:self.population_size]
        self.population = self.population[best_indexes]
        self.fitness = self.fitness[best_indexes]

    def _mutation(self, feature_mask):
        mutated_mask = feature_mask.copy()
        for i in range(len(mutated_mask)):
            if np.random.random() < self.mutation_probability:
                mutated_mask[i] = ~ mutated_mask[i]
        return mutated_mask

    def _crossover(self, parent1, parent2, method='random'):
        children = None
        match method:
            case 'random':
                mask = np.random.random(size=self.feature_dim) > 0.5
                children = np.where(mask, parent1, parent2)

        return self._mutation(children)

    def _estimate_algorithm_on_subset(self, individual):
        X_train_sub = self.X_train[:, individual]
        X_test_sub = self.X_test[:, individual]

        self.algorithm.fit(X_train_sub, self.y_train)
        score = self.algorithm_metric(
            self.y_test,
            self.algorithm.predict(X_test_sub))
        return score



    def select(self, X, y):
        n, m = X.shape
        self.feature_dim = m
        self.X = X
        self.y = y
        self.history = []

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self._generate_population()
        self._calculate_fitness()
        self._store_population(0)

        for iteration in range(1, self.max_iterations):
            new_population = []
            for _ in range(self.population_size):
                parent1_idx = np.random.randint(0, len(self.population))
                parent2_idx = np.random.randint(0, len(self.population))

                while parent1_idx == parent2_idx:
                    parent2_idx = np.random.randint(0, len(self.population))

                child = self._crossover(
                    self.population[parent1_idx],
                    self.population[parent2_idx],
                    self.crossover_method
                )

                new_population.append(child)

            self.population = np.vstack(
                (self.population, np.array(new_population))
            )
            self._calculate_fitness()
            self._selection()
            self._store_population(iteration)

        best_idx = np.argmax(self.fitness)
        best_feature_mask = self.population[best_idx]
        best_fitness = self.fitness[best_idx]

        self.feature_information = {
            'selected_features': best_feature_mask,
            'num_selected': np.sum(best_feature_mask),
            'fitness_score': best_fitness
        }

        return best_feature_mask, best_fitness

    def _store_population(self, iteration):
        population_copy = self.population.copy()
        fitness_copy = self.fitness.copy()

        # Find the best individual in current population
        best_idx = np.argmax(fitness_copy)

        best_mask = population_copy[best_idx]
        best_fitness = fitness_copy[best_idx]

        # Calculate average fitness
        avg_fitness = np.mean(fitness_copy)

        # Calculate diversity (number of unique individuals)
        unique_individuals = np.unique(population_copy, axis=0).shape[0]

        # Store data for this iteration
        self.history.append({
            'iteration': iteration,
            'population': population_copy,
            'fitness': fitness_copy,
            'best_mask': best_mask,
            'best_fitness': best_fitness,
            'avg_fitness': avg_fitness,
            'unique_individuals': unique_individuals,
            'selected_features_count': np.sum(best_mask)
        })


    def _calculate_fitness(self):
        self.fitness = []
        for individual in self.population:
            self.fitness.append(
                self.fitness_function(
                    self._estimate_algorithm_on_subset(individual),
                    individual
                ))
        self.fitness = np.array(self.fitness)