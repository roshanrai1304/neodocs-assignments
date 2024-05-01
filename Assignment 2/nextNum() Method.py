import random

class NextNumberGenerator:
    def __init__(self, numbers, probabilities):
        self.numbers = numbers
        self.probabilities = probabilities
        self.cumulative_probabilities = self.calculate_cumulative_probabilities()

    def calculate_cumulative_probabilities(self):
        cumulative_probabilities = []
        cumulative_prob = 0
        for prob in self.probabilities:
            cumulative_prob += prob
            cumulative_probabilities.append(cumulative_prob)
        return cumulative_probabilities

    def nextNum(self):
        rand = random.random()
        for i, cumulative_prob in enumerate(self.cumulative_probabilities):
            if rand <= cumulative_prob:
                return self.numbers[i]
        
        return self.numbers[-1]


numbers = [-1, 0, 1, 2, 3]
probabilities = [0.01, 0.3, 0.58, 0.1, 0.01]
generator = NextNumberGenerator(numbers, probabilities)


result_counts = {num: 0 for num in numbers}
num_tests = 100
for _ in range(num_tests):
    result = generator.nextNum()
    result_counts[result] += 1


for num, count in result_counts.items():
    print(f"{num}: {count} times ({count / num_tests * 100:.2f}%)")
