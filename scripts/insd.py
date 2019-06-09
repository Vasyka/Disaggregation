import logging
import traceback
import gurobipy as gr
import numpy as np
from scipy.sparse import csr_matrix

def insd(G, aa, c, sparsed=False):
    """
    Построение таблицы с помощью метода INSD(improved normalized square difference).
    Решает систему G*x = c, где G - матрица and c - вектор ограничений.

    Parameters
    ----------
    G : np.ndarray
        матрица коэффициентов для линейных ограничений
    aa: np.ndarray
        векторизованная базовая матрица
    c: np.ndarray
        вектор ограничений

    Returns
    -------
    a: np.ndarray
        векторизованная матрица результата
    """
    try:
        model = gr.Model("INSD")

        a = aa.astype(float)

        a[aa == 0] = 1e-10

        at = a.flatten()

        x = np.array([model.addVar() for i in at])
        model.update()

        t = gr.QuadExpr()
        dif = (at - x) * (at - x) / at
        for i in range(len(at)):
            t.add(dif[i])
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
            cover_rows = [np.nonzero(row)[0] for row in G]

            for i in range(len(G)):
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
