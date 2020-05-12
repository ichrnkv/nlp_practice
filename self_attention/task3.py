import numpy as np


def calc_logits(queries, keys):
    """
    Self-attention logits
    :param queries: numpy matrix
    :param keys: numpy matrix
    """
    return np.dot(queries, np.transpose(keys))


def rows_softmax(matrix):
    """
    Softmax by rows
    :param matrix: numpy matrix
    """
    att_scores = np.zeros((matrix.shape))
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            att_scores[i][j] = np.exp(value)/sum(np.exp(row))

    return att_scores


def matmul(matrix1, matrix2, bias=None):
    """
    Matrix multiplication with bias
    :param matrix1: numpy matrix size any x N
    :param matrix2: numpy matrix size N x any
    :param bias: numpy array
    """
    if bias is not None:
        result = np.dot(matrix1, matrix2) + bias
    else:
        result = np.dot(matrix1, matrix2)
    return result


if __name__ == '__main__':
    Input = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
    ProjK = np.array([[1, 0], [0, 0]])
    ProjQ = np.array([[0, 0], [1, 0]])
    ProjV = np.array([[1, 0], [0, 1]])
    BiasK = np.array([0, 0])
    BiasQ = np.array([0, 0])
    BiasV = np.array([0, 0])

    Keys = matmul(Input, ProjK, BiasK)
    Queries = matmul(Input, ProjQ, BiasQ)
    Values = matmul(Input, ProjV, BiasV)

    Logits = calc_logits(Queries, Keys)
    AttScores = rows_softmax(Logits)
    Result = np.dot(AttScores, Values)

    print(Result)
