def hardlim(n):
    if n > 0:
        return 1
    else:
        return 0


def update_weights(a, b, sign=0):
    for i in range(0, len(a)):
        a[i] += sign * b[i]


def product_vector(a, b):
    prod = 0
    for i in range(0, len(a)):
        prod += a[i] * b[i]
    return prod


class Perceptron:

    def __init__(self):
        self.weights = [[1, 1], [1, -1]]

    def train(self, data, target):
        error = 1
        while error != 0:
            error = 0
            for i in range(0, len(data)):
                for j in range(0, len(data[i])):
                    out = hardlim(product_vector(self.weights[j], data[i][j]))
                    error += abs(target[i][j] - out)
                    update_weights(self.weights[j], data[i][j], target[i][j] - out)

    def predict(self, data):
        classes = []
        for i in range(0, len(data)):
            temp = [hardlim(product_vector(data[i][0], self.weights[0])),
                    hardlim(product_vector(data[i][1], self.weights[1]))]
            classes.append(temp)
        return classes


if __name__ == '__main__':
    data = [[[1, 1], [1, 3]], [[2, -1], [2, 0]], [[-1, 2], [-2, 1]], [[-1, -1], [-2, -2]]]
    target = [[0, 0], [0, 1], [1, 0], [1, 1]]
    model = Perceptron()
    model.train(data, target)
    classes = model.predict(data)
    print(classes)

