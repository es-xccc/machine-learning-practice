import json
import itertools

class Problem:
    def __init__(self, input):
        self.input = input
        self.numTasks = len(input)

    def cost(self, ans):
        totalTime = 0
        for task, agent in enumerate(ans):
            totalTime += self.input[task][agent]
        return totalTime

def solve(problem):
    bestCost = float('inf')
    bestAssignment = None
    for assignment in itertools.permutations(range(len(problem.input))): # 自動排列組合 task 對應第幾個 agent
        cost = problem.cost(assignment)
        if cost < bestCost:
            bestCost = cost
            bestAssignment = assignment
    return bestAssignment

if __name__ == '__main__':
    with open('input.json', 'r') as inputFile:
        data = json.load(inputFile)
        for key in data:
            input = data[key]
            problem = Problem(input)
            bestAssignment = solve(problem)  # 用演算法得出的答案
            print('Assignment:', bestAssignment)  # print 出分配結果
            print('Cost:', problem.cost(bestAssignment))  # print 出 cost 是多少