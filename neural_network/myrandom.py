import numpy

def random_arr(x, y):
    return numpy.random.rand(x, y)

if __name__ == '__main__':
    print(random_arr(3, 3))