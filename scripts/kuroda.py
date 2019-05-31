import logging
import traceback
import gurobipy as gr
import numpy as np


def kuroda(G, aa, c, mtype):
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

        czero = G @ a

        w = np.empty_like(G)

        if mtype == 1:
            for i in range(G.shape[0]):
                for j in range(G.shape[1]):
                    w[i][j] = G[i][j] * (czero[i] ** 2 / a[j] ** 2)
        elif mtype == 2:
            w = G * (c ** 2 / 2)
        elif mtype == 3:
            w = G
        else:
            return -1

        t = gr.QuadExpr()

        for i in range(len(c)):
            for j in range(len(at)):
                t.add(0.5 * (x[j] / c[i] - at[j] / czero[i]) * (x[j] / c[i] - at[j] / czero[i]) * w[i][j])

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
