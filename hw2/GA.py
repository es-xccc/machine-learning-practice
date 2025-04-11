import json
import random

class Problem:
    def __init__(self, input):
        self.input = input
        self.numTasks = len(input)

    def cost(self, ans):
        totalTime = 0
        for task, agent in enumerate(ans):
            totalTime += self.input[task][agent]
        return totalTime
    
    def fitness(self, ans):
        return 1 / self.cost(ans)


def initial_population(num_tasks):
    population = []
    for _ in range(100 * num_tasks):
        individual = list(range(num_tasks))
        random.shuffle(individual)
        population.append(individual)
    return population

def selection(population, fitnesses):
    selected = random.choices(population, weights = fitnesses, k = len(population) // 2)
    return selected

def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

def mutation(individual, numTasks):
    task = [j for j in range(numTasks) if j not in individual]
    random.shuffle(task)
    for i in range(numTasks):
        if individual.count(i) > 1:
            individual[individual.index(i)] = task.pop(0)

def solve(problem):
    population = initial_population(problem.numTasks)
    best_individual = None
    best_fitness = 0

    for _ in range(100):
        fitnesses = [problem.fitness(individual) for individual in population]
        best_gen_fitness = max(fitnesses)
        best_gen_individual = population[fitnesses.index(best_gen_fitness)]

        if best_gen_fitness > best_fitness:
            best_fitness = best_gen_fitness
            best_individual = best_gen_individual

        parents = selection(population, fitnesses)
        next_population = []

        for i in range(0, len(parents), 2):
            parent1 = parents[i]
            parent2 = parents[i + 1] if i + 1 < len(parents) else parents[0]
            child1, child2 = crossover(parent1, parent2)
            mutation(child1, problem.numTasks)
            mutation(child2, problem.numTasks)
            next_population.extend([child1, child2])

        population = next_population

    return best_individual

if __name__ == '__main__':
    with open('input.json', 'r') as inputFile:
        data = json.load(inputFile)
        for key in data:
            input = data[key]
            problem = Problem(input)
            bestAssignment = solve(problem)  # 用基因演算法得出的答案
            print('Assignment:', bestAssignment)  # print 出分配結果
            print('Cost:', problem.cost(bestAssignment))  # print 出 cost 是多少