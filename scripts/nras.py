import logging
import traceback
import numpy as np
from scipy.sparse import csr_matrix

def nras(G, aa, c, accuracy, limit, sparsed=False):
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
        if sparsed:
            cover_rows = [G.indices[G.indptr[i]:G.indptr[i + 1]] for i in range(G.shape[0])]
        else:
            cover_rows = [np.nonzero(row)[0] for row in G]

        while (max(abs(G.dot(a) - c).flatten()) > accuracy) and (counter < limit):

            for i in range(quantity_c):
                inn_2d = a[cover_rows[i]]
                inn = inn_2d.flatten()
                pos = sum(inn[inn > 0])
                neg = -sum(inn[inn < 0])
                su = sum(inn)

                if (pos + neg) != 0:
                    if pos * neg == 0:
                        r_nom = ((np.sign(su) * c[i]) / (pos + neg))[0]
                        r = r_nom * np.ones([len(inn), 1])
                        a_sign = (np.sign(inn))
                        r = r ** (np.sign(su) * a_sign.T)[np.newaxis].T
                        a[cover_rows[i]] = inn_2d * r
                    else:
                        if pos > 0:
                            r_nom = (c[i] + np.sqrt(c[i] ** 2 + 4. * pos * neg)) / (2. * pos)
                            r = r_nom * np.ones([len(inn), 1])
                            a_sign = (np.sign(inn))
                            r = r ** (a_sign)[np.newaxis].T
                            a[cover_rows[i]] = inn_2d * r
                        else:
                            if c[i] != 0:
                                r = (-neg / c[i]) * np.ones([len(inn), 1])
                                a_sign = (np.sign(inn))
                                r = r ** a_sign[np.newaxis].T
                                a[cover_rows[i]] = inn_2d * r
            counter = counter + 1
        print(counter)
        return a
    except Exception as e:
        logging.error(traceback.format_exc())
        return -1
