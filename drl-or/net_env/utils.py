#coding=utf-8
import random
import numpy as np

'''
@param: 
    weight: an int list
'''
def weight_choice(weight):
    t = random.randint(0, sum(weight) - 1)
    for i, val in enumerate(weight):
        t -= val
        if t < 0:
            return i


def random_weight_choice(weight):
    return random.choices(range(len(weight)), weights=weight, k=1)[0]

if __name__ == "__main__":

    traffic_matrix = [
        [0, 236, 58, 27, 203, 89, 132, 141, 18, 57, 489],
        [129, 0, 77, 73, 337, 61, 104, 34, 36, 45, 89],
        [76, 646, 0, 93, 238, 167, 95, 241, 96, 151, 421],
        [42, 182, 14, 0, 72, 23, 608, 146, 11, 22, 90],
        [191, 887, 131, 46, 0, 165, 537, 246, 51, 119, 449],
        [35, 332, 84, 25, 108, 0, 20, 46, 15, 18, 100],
        [121, 820, 64, 72, 302, 75, 0, 364, 52, 187, 470],
        [450, 452, 232, 127, 689, 226, 354, 0, 84, 151, 1181],
        [12, 36, 113, 3, 16, 11, 119, 26, 0, 418, 13],
        [89, 559, 44, 36, 164, 41, 242, 153, 64, 0, 137],
        [800, 943, 501, 213, 590, 262, 458, 1588, 66, 286, 0]
    ]

    m = list(np.array(traffic_matrix).flatten())
    print(m)
    index = weight_choice(m)
    print(index // 11, index % 11)


