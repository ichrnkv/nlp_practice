# multihead self-attention
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


def self_attention(input_matrix,
                   proj_k, proj_q, proj_v,
                   bias_k, bias_q, bias_v):
    """
    Simple self-attention
    :param input_matrix: numpy matrix
    :param proj_k: numpy matrix
    :param proj_q: numpy matrix
    :param bias_k: numpy array
    :param bias_q: numpy array
    :param bias_v: numpy array
    """
    keys = matmul(input_matrix, proj_k, bias_k)
    queries = matmul(input_matrix, proj_q, bias_q)
    values = matmul(input_matrix, proj_v, bias_v)
    logits = calc_logits(queries, keys)
    att_scores = rows_softmax(logits)
    result = np.dot(att_scores, values)

    return result


if __name__ == '__main__':
    Input = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
    ProjK1 = np.array([[1, 0], [0, 0]])
    ProjK2 = np.array([[0, 0], [1, 0]])
    ProjQ1 = np.array([[0, 1], [1, 0]])
    ProjQ2 = np.array([[1, 1], [1, 1]])
    ProjV1 = np.array([[1], [0]])
    ProjV2 = np.array([[0], [1]])
    BiasK = np.array([0, 0])
    BiasQ = np.array([0, 0])
    BiasV = 0

    Result1 = self_attention(Input, ProjK1, ProjQ1, ProjV1, BiasK, BiasQ, BiasV)
    Result2 = self_attention(Input, ProjK2, ProjQ2, ProjV2, BiasK, BiasQ, BiasV)

    MHResult = np.concatenate((Result1, Result2), axis=1)

    print(MHResult)
