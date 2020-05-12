import numpy as np


def calc_logits(matrix):
    """
    Self-attention Logits
    :param matrix: numpy matrix
    """
    return np.dot(matrix, np.transpose(matrix))


if __name__ == '__main__':
    Input = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
    Logits = calc_logits(Input)
    print(Logits)
