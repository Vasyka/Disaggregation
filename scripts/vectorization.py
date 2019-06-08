import logging
import traceback
import numpy as np
from scipy.sparse import coo_matrix
import scipy.sparse as sps

def addG(indexes, size, G = None):
    """
    Добавение нового линейного ограничения к матрице G

    Parameters
    ----------
    G: np.ndarray
        матрица линейных ограничений
    indexes: np.ndarray
        вектор индексов, участвующих в новом ограничении
    size: tuple
        размерность исходной таблицы

    Returns
    -------
    G: np.ndarray
        матрица линейных ограничений с добавлением нового
    """
    try:
        g = np.zeros(size[0] * size[1])
        for ind in indexes:
            g[ind[1] * size[0] + ind[0]] = 1
        if G is not None:
            return np.vstack((G,g))
        return g
    except Exception as e:
        logging.error(traceback.format_exc())
        return -1


def addrows(shape, G = None, sparsed = False):
    """
    Добавение линейных ограничений для сумм всех строк к матрице G

    Parameters
    ----------
    G: np.ndarray
        матрица линейных ограничений
    shape: tuple
        размерность исходной таблицы

    Returns
    -------
    G: np.ndarray
        матрица линейных ограничений с добавлением новых
    """
    try:
        if sparsed:
            row = np.tile(range(shape[0]),shape[1])
            col = np.arange(shape[0]*shape[1])
            data = np.ones(shape[0]*shape[1])
            g = coo_matrix((data, (row, col)), shape=(shape[0], shape[0]*shape[1])).tocsr()
            if G is not None:
                return sps.vstack((G,g))
        else:
            g = np.zeros(shape=[shape[0], shape[0] * shape[1]])
            g = np.tile(np.eye(shape[0]),shape[1])
            if G is not None:
                return np.vstack((G,g))
        return g
    except Exception as e:
        logging.error(traceback.format_exc())
        return -1


def addcolumns(shape, G = None, sparsed = False):
    """
    Добавение линейных ограничений для сумм всех столбцов к матрице G

    Parameters
    ----------
    G: np.ndarray
        матрица линейных ограничений
    shape: tuple
        размерность исходной таблицы

    Returns
    -------
    G: np.ndarray
        матрица линейных ограничений с добавлением новых
    """
    try:
        if sparsed:
            row = np.repeat(range(shape[1]),shape[0])
            col = np.arange(shape[0]*shape[1])
            data = np.ones(shape[0]*shape[1])
            g = coo_matrix((data, (row, col)), shape=(shape[1], shape[0]*shape[1])).tocsr()
            if G is not None:
                return sps.vstack((G,g))
        else:
            g = np.zeros(shape=[shape[1], shape[0] * shape[1]])
            for i in range(0, shape[1]):
                for j in range(shape[0]):
                    g[i][i * shape[0] + j] = 1
            if G is not None:
                return np.vstack((G,g))
        return g
    except Exception as e:
        logging.error(traceback.format_exc())
        return -1


def tovector(a):
    """
    Векторизация матрицы а

    Parameters
    ----------
    a: np.ndarray
        исходная матрица

    Returns
    -------
    a: np.ndarray
        векторизованная матрица а
    """
    try:
        return a.T.flatten()[np.newaxis].T
    except Exception as e:
        logging.error(traceback.format_exc())
        return -1


def tomatrix(a, shape = None):
    """
    Перевод вектора а в матричную форму размерности shape

    Parameters
    ----------
    a: np.ndarray
        исходный вектор
    shape: tuple
        размерность матрицы, если не задана, то предполагается, что матрица квадратная

    Returns
    -------
    a: np.ndarray
        матричная форма вектора a
    """
    try:
        if not shape:
            shape = int(np.sqrt(len(a))), -1
        return a.reshape((shape[1], shape[0])).T
    except Exception as e:
        logging.error(traceback.format_exc())
        return -1
