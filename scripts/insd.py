import logging
import traceback
import gurobipy as gr
import numpy as np


def insd(G, aa, c):
    """
    Построение таблицы с помощью метода INSD

    Parameters
    ----------
    G : np.ndarray
        матрица линейных ограничений
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
        for i in range(len(at)):
            t.add((at[i] - x[i]) * (at[i] - x[i]) / at[i])
        model.setObjective(t)

        for i in range(len(G)):
            model.addLConstr(gr.LinExpr(G[i, :], x), gr.GRB.EQUAL, c[i])

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
