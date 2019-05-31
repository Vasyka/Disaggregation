import logging
import traceback
import numpy as np


def mape(x, x_true):
    """
    Получение значения метрики MAPE для таблицы x на основании таблицы x_true

    Parameters
    ----------
    x_true: np.ndarray
        векторизованная базовая матрица
    x: np.ndarray
        векторизованная матрица для сравнения ограничений

    Returns
    -------
    a: float
        значение метрики
    """
    try:
        x = x.flatten()
        x_true = x_true.flatten()
        return (np.sum(abs(x - x_true) / abs(x_true)) / len(x)) * 100
    except Exception as e:
        logging.error(traceback.format_exc())
        return -1


def wape(x, x_true):
    """
    Получение значения метрики WAPE для таблицы x на основании таблицы x_true

    Parameters
    ----------
    x_true: np.ndarray
        векторизованная базовая матрица
    x: np.ndarray
        векторизованная матрица для сравнения ограничений

    Returns
    -------
    a: float
        значение метрики
    """
    try:
        x = x.flatten()
        x_true = x_true.flatten()
        return 100 * 1 / (np.sum(x_true)) * np.sum(abs(x - x_true))
    except Exception as e:
        logging.error(traceback.format_exc())
        return -1


def swad(x, x_true):
    """
    Получение значения метрики SWAD для таблицы x на основании таблицы x_true

    Parameters
    ----------
    x_true: np.ndarray
        векторизованная базовая матрица
    x: np.ndarray
        векторизованная матрица для сравнения ограничений

    Returns
    -------
    a: float
        значение метрики
    """
    try:
        x = x.flatten()
        x_true = x_true.flatten()
        return sum(abs(x_true) * abs(x - x_true)) / np.sum(x_true ** 2)
    except Exception as e:
        logging.error(traceback.format_exc())
        return -1
