import numpy as np


def calc_logits(matrix):
    """
    Self-attention Logits
    :param matrix: numpy matrix
    :return:
    """
    return np.dot(matrix, np.transpose(matrix))


def rows_softmax(matrix):
    """
    Softmax by rows
    :param matrix: numpy matrix
    :return:
    """
    att_scores = np.zeros((matrix.shape))
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            att_scores[i][j] = np.exp(value)/sum(np.exp(row))

    return att_scores


if __name__ == '__main__':
    Input = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
    Logits = calc_logits(Input)
    AttScores = rows_softmax(Logits)
    Result = np.dot(AttScores, Input)
    print(Result)
