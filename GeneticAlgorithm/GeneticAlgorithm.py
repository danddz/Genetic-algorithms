import random
from typing import Tuple, List
import numpy as np

class Genome:
    values = np.array([])
    fitness = None

    def __init__(self, low: int, high: int):
        self.low = low
        self.high = high

    @classmethod
    def random_init(cls, size: int, low: int, high: int):
        self = cls(low, high)
        self.values = self.random_values(size)
        return self

    def random_values(self, size: int) -> np.array:
        return np.random.uniform(self.low, self.high, size)

    def __eq__(self, other):
        return np.array_equal(self.values, other.values)

    def __lt__(self, other):
        return self.fitness < other.fitness

    def __str__(self):
        return f'({self.values}, fitness={self.fitness}'
    
    def __repr__(self): 
        return str(self)
        
class GeneticAlgorithm:
    def __init__(self,
                 population_size: int,
                 genome_size: int,
                 value_range: Tuple[int, int],
                 approximation_deg: int,
                 mutation_probability: float,
                 max_iterations: int):
        self.approximation_deg = approximation_deg
        self.population = self.create_population(population_size, genome_size, value_range)
        self.calc_fitness()
        print('The first fitness =', self.population[0].fitness)
        self.mutation_probability = mutation_probability
        self.max_iterations = max_iterations
        self.best_gene = self.population[0]

    @staticmethod
    def create_population(population_size: int, genome_size: int, value_range: Tuple[int, int]) -> list:
        return [Genome.random_init(genome_size, *value_range) for _ in range(population_size)]

    def solve(self) -> np.array:
        for iteration in range(self.max_iterations):
            for genome in self.population:
                if genome.fitness > self.best_gene.fitness:
                    self.best_gene = genome

            self.population = self.get_new_population()
            self.calc_fitness()
        return self.best_gene

    def get_new_population(self) -> List[Genome]:
        genome = self.population[0] 
        if False not in list(map(lambda g: g == genome, self.population)):
            return self.create_population(len(self.population), len(genome.values), (genome.low, genome.high))

        new_population = []
        fittest_half = self.select_fittest_half()

        for _ in range(len(self.population) - 1):
            parents = np.random.choice(fittest_half, 2)
            new_genome = self.mating(parents)
            self.mutate(new_genome, self.mutation_probability)
            new_population.append(new_genome)
            
        new_population.append(self.best_gene)
        
        return new_population

    def calc_fitness(self):
        for genome in self.population:
            u = self.get_u(genome, self.approximation_deg)
            x = self.get_x(u)
            genome.fitness = self.get_fitness(x)

    @staticmethod
    def get_u(genome: Genome, deg: int) -> np.array:
        xs = genome.values
        ys = np.linspace(0, 1, len(genome.values))
        u = np.array(np.polyfit(xs, ys, deg))
        return u

    @staticmethod
    def get_x(u: np.array) -> np.array:
        x = u / np.arange(len(u), 0, -1)
        x = np.append(x, 0)
        return x

    def get_fitness(self, x) -> int:
        return self.get_integral(x) - self.get_integral(np.convolve(x, x))

    @staticmethod
    def get_integral(x: np.array) -> int:
        return (x / np.arange(len(x), 0, -1)).sum()

    def select_fittest_half(self) -> list:
        return sorted(self.population)[(len(self.population) // 2):]

    @staticmethod
    def mutate(genome: Genome, mutation_probability: float):
        def random_value_mutation(value):
            return genome.random_values(1)[0] if random.random() <= mutation_probability else value

        genome.values = np.array(list(map(random_value_mutation, genome.values)))

    @staticmethod
    def mating(parents: List[Genome]) -> Genome:
        if np.random.randint(0, 1) == 1:
            parents = (parents[1], parents[0])

        parent = parents[0]
        genome_size = len(parent.values)

        pivot_1 = np.random.randint(1, genome_size - 1)
        pivot_2 = np.random.randint(pivot_1 + 1, genome_size)

        new_genome = Genome(parent.low, parent.high)
        new_genome.values = np.concatenate((
            parents[0].values[:pivot_1],
            parents[1].values[pivot_1:pivot_2],
            parents[0].values[pivot_2:]
        ))
        return new_genome


def str_function(approximation: np.array, func_name: str) -> str:
    function = f'{func_name} ='
    function += f' {approximation[-1]:0.3f}'
    for i, value in enumerate(reversed(approximation[:-1]), 1):
        function += f' {"+" if value >= 0.0 else "-"} {np.abs(value):0.3f} * t^{i}'
    return function


def main():
    deg = 3
    genetic_algo = GeneticAlgorithm(
        population_size=5,
        genome_size=10,
        value_range=(-1, 1),
        approximation_deg=deg,
        mutation_probability=0.05,
        max_iterations=1000,
    )

    genome = genetic_algo.solve()

    u = GeneticAlgorithm.get_u(genome, deg)
    x = GeneticAlgorithm.get_x(u)

    print(str_function(u, 'u(t)'))
    print(str_function(x, 'x(t)'))
    print(f'Max integral value: {genome.fitness:0.7f}')


if __name__ == '__main__':
    main()
