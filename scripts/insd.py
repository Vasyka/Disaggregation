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
            
        # add variable z to model ( z_ij = x_ij/a_ij)
        z = model.addVars(len(at), lb=0., name='z')
        for i in range(len(z)):
            z[i].start = 1.
        model.update()
        
        # add function for minimization to model
        t = gr.QuadExpr()
        for i in range(len(at)):
            t.add(abs(at[i])*(z[i] - 1.)*(z[i] - 1.))
        model.setObjective(t)
    
        # add constraint G*(z.*a)=c
        if sparsed:
            for i in range(G.shape[0]):
                expr = gr.LinExpr()
                start = G.indptr[i]
                end = G.indptr[i + 1]
                idx = G.indices[start:end]
                coef = G.data[start:end] * at[idx]
                
                for j, k in enumerate(idx):
                    expr.add(z[k],coef[j])
                model.addLConstr(expr, gr.GRB.EQUAL, c[i])
        else:
            cover_rows = [np.nonzero(row)[0] for row in G]

            for i in range(len(G)):
                expr = gr.LinExpr()
                idx = cover_rows[i]
                coef = G[i, idx] *at[idx]
                
                for j, k in enumerate(idx):
                    expr.add(z[k],coef[j])
                model.addLConstr(expr, gr.GRB.EQUAL, c[i])
                
        # Set params
        model.setParam('BarConvTol', 1e-8)
        model.setParam('BarQCPConvTol', 1e-6)

        model.setParam('DualReductions', 0)
        # model.setParam('FeasibilityTol',0.01)                        
        model.setParam('OutputFlag', 0)

        model.optimize()

        result = np.array([v.x for v in model.getVars()])
        return result * at
    except Exception as e:
        logging.error(traceback.format_exc())
        return -1
