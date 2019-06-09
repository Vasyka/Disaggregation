import logging
import traceback
import gurobipy as gr
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix


def kuroda(G, aa, c, mtype, sparsed=False):
    """
    Построение таблицы с помощью метода Kuroda
    Решает систему G*x = c, где G - матрица and c - вектор ограничений.

    Parameters
    ----------
    G : np.ndarray
        матрица коэффициентов для линейных ограничений
    aa: np.ndarray
        векторизованная базовая матрица
    c: np.ndarray
        вектор ограничений
    mtype: int
        вариант метода

    Returns
    -------
    a: np.ndarray
        векторизованная матрица результата
    """
    try:
        model = gr.Model("Kuroda")

        a = aa.astype(float)

        a[aa == 0] = 1e-10

        at = a.flatten()
        x = np.array([model.addVar() for i in at])
        model.update()

        czero = G.dot(a)

        if not sparsed:
            cover_rows = [np.nonzero(row)[0] for row in G]

        if mtype == 1:
            a_2 = a ** 2
            czero_2 = czero ** 2

            if sparsed:
                w = csr_matrix(G, copy=True)
                for i in range(G.shape[0]):
                    start = G.indptr[i]
                    end = G.indptr[i + 1]
                    idx = G.indices[start:end]
                    w.data[start:end] = czero_2[i][0] / a_2[idx, 0]
            else:
                w = np.zeros_like(G)
                for i in range(G.shape[0]):
                    w[i, cover_rows[i]] = czero_2[i][0] / a_2[cover_rows[i], 0]

        elif mtype == 2:
            if sparsed:
                w = G.multiply(c ** 2 / 2)
            else:
                w = G * (c ** 2 / 2)
        elif mtype == 3:
            w = G
        else:
            return -1

        t = gr.QuadExpr()

        mask = (c != 0) & (czero != 0)
        indexes = np.where(mask)[0]

        if sparsed:

            for i in indexes:
                start = G.indptr[i]
                end = G.indptr[i + 1]
                for j in range(start, end):
                    idx = G.indices[j]
                    q = (x[idx] / c[i] - at[idx] / czero[i])
                    t.add(0.5 * q * q * w.data[j])
        else:
            for i in indexes:
                for j in cover_rows[i]:
                    q = (x[j] / c[i] - at[j] / czero[i])
                    t.add(0.5 * q * q * w[i, j])

        model.setObjective(t)

        if sparsed:
            for i in range(G.shape[0]):
                start = G.indptr[i]
                end = G.indptr[i + 1]
                idx = G.indices[start:end]
                coef = G.data[start:end]
                variables = x[idx]
                model.addLConstr(gr.LinExpr(coef, variables), gr.GRB.EQUAL, c[i])
        else:
            for i in range(G.shape[0]):
                variables = x[cover_rows[i]]
                coef = G[i, cover_rows[i]]
                model.addLConstr(gr.LinExpr(coef, variables), gr.GRB.EQUAL, c[i])

        model.setParam('BarConvTol', 1e-8)
        model.setParam('BarQCPConvTol', 1e-6)

        model.setParam('DualReductions', 0)

        model.setParam('OutputFlag', 0)

        model.optimize()

        result = np.array([v.x for v in model.getVars()])
        return result
    except Exception as e:
        logging.error(traceback.format_exc())
        return -1
