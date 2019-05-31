import logging
import traceback
import numpy as np



def nras(G, aa, c, accuracy, limit):
    """
    Построение таблицы с помощью метода non_sign_RAS
    Решает систему G*x = c, где G - матрица and c - вектор ограничений.

    Parameters
    ----------
    G : np.ndarray
        матрица коэффициентов для линейных ограничений
    aa: np.ndarray
        векторизованная базовая матрица
    c: np.ndarray
        вектор ограничений
    accuracy: float
        допустимая погрешность
    limit: float
        максимальное количество итераций

    Returns
    -------
    a: np.ndarray
        векторизованная матрица результата
    """
    counter = 0

    try:
        a = aa.astype(float)
        a[a == 0] = 1e-2
        (quantity_c, n) = G.shape

        while (max(abs(G @ a - c).flatten()) > accuracy) and (counter < limit):

            for i in range(quantity_c):
                inn = (a.T * G[i, :]).flatten()
                pos = sum(inn[inn > 0])
                neg = -sum(inn[inn < 0])
                su = sum(inn)

                if (pos + neg) != 0:
                    if pos * neg == 0:
                        r_nom = ((np.sign(su) * c[i]) / (pos + neg))[0]
                        r = r_nom * np.ones([n, 1])
                        a_sign = (np.sign(inn))
                        r = r ** (np.sign(su) * a_sign.T)[np.newaxis].T
                        a = a * r
                    else:
                        if pos > 0:
                            r_nom = (c[i] + np.sqrt(c[i] ** 2 + 4. * pos * neg)) / (2. * pos)
                            r = r_nom * np.ones([n, 1])
                            a_sign = (np.sign(inn))
                            r = r ** a_sign[np.newaxis].T
                            a = a * r
                        else:
                            if c[i] != 0:
                                r = (-neg / c[i]) * np.ones([n, 1])
                                a_sign = (np.sign(inn))
                                r = r ** a_sign[np.newaxis].T
                                a = a * r
            counter = counter + 1
        print(counter)
        return a
    except Exception as e:
        logging.error(traceback.format_exc())
        return -1
