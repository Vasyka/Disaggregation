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
        
    
        # add variable z to model ( z_ij = x_ij/a_ij)
        z = model.addVars(len(at), lb=0., name='z')
        for i in range(len(z)):
            z[i].start = 1.
        model.update()
        
        czero = G.dot(a)
        if not sparsed:
            cover_rows = [np.nonzero(row)[0] for row in G]
            
        # Set w
        if mtype == 1:
                a_2 = a ** 2
                czero_2 = czero ** 2

                if sparsed:
                    w = csr_matrix(G, copy=True)
                    # print(w.shape)
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
                w = G.multiply(c**2 / 2) 
                
            else:
                w = G * (c ** 2 / 2)
        elif mtype == 3:
            w = G
        else:
            return -1    
        
        # Get indexes of elements with not null c & c_zero 
        mask = (c != 0) & (czero != 0)
        indexes = np.where(mask)[0]
        
        
        # add function for minimization to model
        t = gr.QuadExpr()
        
        if sparsed:
            for i in indexes:
                start = G.indptr[i]
                end = G.indptr[i + 1]
                idx = G.indices[start:end]
                coef = at[idx]*at[idx]/2*w.data[idx]
                
                for j, k in enumerate(idx):
                    q = z[k]/c[i] - 1. /czero[i]
                    t.add(q*q,coef[j])
        else:
            for i in indexes:
                idx = cover_rows[i]
                coef = at[idx]*at[idx]/2*w[i,idx]
                
                for j, k in enumerate(idx):
                    q = z[k]/c[i] - 1. /czero[i]
                    t.add(q*q,coef[j])
                    
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

        model.setParam('DualReductions', 1)
        #model.setParam('FeasibilityTol',0.01)                        
        model.setParam('OutputFlag', 0)

        model.optimize()

        result = np.array([v.x for v in model.getVars()])
        return result * at
    except Exception as e:
        logging.error(traceback.format_exc())
        return -1

    