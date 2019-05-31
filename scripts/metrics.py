import logging
import traceback
import numpy as np
from inspect import getmembers, isfunction
import sys

def get_values(x, x_true, save = False):
    """
    Получение значения всех метрик для таблицы x на основании таблицы x_true

    Parameters
    ----------
    x_true: np.ndarray
        векторизованная базовая матрица
    x: np.ndarray
        векторизованная матрица для сравнения ограничений
    save: boolean, default False
        по умолчанию - печать значений метрик, если необходимо сохранить значения (True) - возвращает в виде словаря
    Returns
    -------
    results: dict
        значения метрик - {название метрики: значение метрики}
    """
    results = {}
    spec_functions = ['get_values','getmembers','isfunction']
    for metric in getmembers(sys.modules[__name__], isfunction):
        if (metric[0] not in spec_functions):
            metric_name = metric[0].upper()
            res = metric[1](x, x_true)
            if save:
                results[metric_name] = round(res,3)
            else:
                print(metric_name, round(res,3))
    if save:
        return results


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
        x_true[x_true == 0] = 1e+20
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
