import random
from typing import Tuple, List
import numpy as np
import matplotlib.pyplot as plt

class Genome:
    values = np.array([])
    fitness = None
    all_ = list()
    X_ = list()

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
    
    @classmethod
    def show_individ(cls, index):
        size = 8
        plt.figure(figsize=(size,size))
        plt.xlabel('t', fontsize=16)
        plt.ylabel('U', fontsize=16)
        plt.xlim(0, 1)
        plt.ylim(-1, 4)
        plt.title(f'I = {cls.all_[index].fitness}', size=16)
        plt.grid()

        values = cls.all_[index].values
        plt.plot(np.linspace(0, 1, len(values)), values)
        
    @classmethod
    def show_X(cls):
        size = 8
        plt.figure(figsize=(size,size))
        plt.xlabel('t', fontsize=16)
        plt.ylabel('X', fontsize=16)
        plt.xlim(0, 1)
        plt.ylim(-1, 4)
        plt.grid()

        values = cls.X_
        plt.plot(np.linspace(0, 1, len(values)), values)

    def __eq__(self, other):
        return np.array_equal(self.values, other.values)

    def __lt__(self, other):
        return self.fitness < other.fitness

    def __str__(self):
        return f'({self.values}, fitness = {self.fitness}'
    
    def __repr__(self): 
        return str(self)

    def __len__(self):
        return len(self.values)
      
      
class GradientAlgorithm:
    def __init__(self,
                 genome_size: int,
                 value_range: Tuple[int, int],
                 max_iterations: int):
        self.population = self.create_population(genome_size, value_range)
        self.population.fitness = self.get_integral(self.population.values)
        self.max_iterations = max_iterations
        Genome.all_.append(self.population)

    @staticmethod
    def create_population(genome_size: int, value_range: Tuple[int, int]) -> list:
        return Genome.random_init(genome_size, *value_range)

    def solve(self) -> Genome:
        for iteration in range(self.max_iterations):
            #print(self.population.fitness)
            self.population = self.new_population()
            Genome.all_.append(self.population)
            Genome.X_.append(self.get_x(len(self.population.values) - 1, self.population.values))

        return self.population
    
    def new_population(self, h=1) -> Genome:
        grad = self.nabla_operator(self.population)
        new_population = Genome(self.population.low, self.population.high)
        new_population.values = self.population.values + h * grad

        for i in range(len(new_population.values)):
            if new_population.values[i] >= 1:
                new_population.values[i] = 1
            elif new_population.values[i] <= -1:
                new_population.values[i] = -1
        
        new_population.fitness = self.get_integral(new_population.values)
        
        return new_population
    
    @staticmethod
    def nabla_operator(genome: Genome, h=0.1) -> np.array:
        gradient = np.zeros(len(genome.values))
    
        for i in np.arange(len(gradient)):
            genome.values[i] += h
            gradient[i] = (GradientAlgorithm.get_integral(genome.values) - genome.fitness) / h
            genome.values[i] -= h
            
        return gradient
    
    @staticmethod
    def sum_(func: List) -> float:
        return (sum(func) * 2 - func[0] - func[-1]) / len(func) / 2.0
    
    @staticmethod
    def get_x(k: int, func: Genome) -> float:
        return (sum(func[:k + 1]) * 2 - func[0] - func[k]) / len(func) / 2.0
    
    @staticmethod
    def get_integral(func: Genome) -> float:
        return GradientAlgorithm.sum_([x - x ** 2 for x in [GradientAlgorithm.get_x(i, func) for i in range(len(func))]])
      
      
def main():
    gradient_algo = GradientAlgorithm(
        genome_size=10 * 3,
        value_range=(-1, 1),
        max_iterations=100,
    )

    genome = gradient_algo.solve()

    print(f'Max integral value: {genome.fitness:0.2f}')
    
    Genome.show_individ(0)
    Genome.show_individ(-1)
    Genome.show_X()
    
    
if __name__ == '__main__':
    main()
