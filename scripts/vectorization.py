import logging
import traceback
import numpy as np


def addG(G, indexes, size):
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
        G = np.append(G, [g], axis=0)
        return G
    except Exception as e:
        logging.error(traceback.format_exc())
        return -1


def addrows(G, shape):
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
        g = np.zeros(shape=[shape[0], shape[0] * shape[1]])
        for i in range(0, shape[0]):
            for j in range(shape[1]):
                g[i][j * shape[0] + i] = 1
        G = np.append(G, g, axis=0)
        return G
    except Exception as e:
        logging.error(traceback.format_exc())
        return -1


def addcolumns(G, shape):
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
        g = np.zeros(shape=[shape[1], shape[0] * shape[1]])
        for i in range(0, shape[1]):
            for j in range(shape[0]):
                g[i][i * shape[0] + j] = 1
        G = np.append(G, g, axis=0)
        return G
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


def tomatrix(a, shape):
    """
    Перевод вектора а в матричную форму размерности shape

    Parameters
    ----------
    a: np.ndarray
        исходный вектор
    shape: tuple
        размерность матрицы

    Returns
    -------
    a: np.ndarray
        матричная форма вектора a
    """
    try:
        return a.reshape((shape[1], shape[0])).T
    except Exception as e:
        logging.error(traceback.format_exc())
        return -1
