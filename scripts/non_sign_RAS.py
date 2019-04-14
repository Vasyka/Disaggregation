import os
import re
import warnings
from operator import add

import numpy as np
import pandas as pd
import timeit

# Отключаем warnings
warnings.simplefilter("ignore")



def nras(a, G, c, accuracy, limit):
    counter = 0
    ret = np.empty([1,3])

    (quantity_c, n) = G.shape

    while ((max(abs(G @ a - c)) > accuracy) and (counter < limit)):

        s = c.size

        (nevyazka, nom_nevyazka) = (max(abs(G @ a - c)), np.argmax(abs(G @ a - c)))

        ret = np.append(ret, [[counter, nevyazka, sum(abs(G @ a - c)) / s]], axis=0)

        for i in range(quantity_c):
            inn = (a.T * G[i, :]).flatten()
            pos = sum(inn[inn > 0])
            neg = -sum(inn[inn < 0])
            su = sum(inn)

            # if np.isnan(a):
            #     out = 9

            if pos + neg == 0 and abs(c[i]) > abs(accuracy):
                del1 = G[i, :]
                del2 = a
                # error('pos+neg=0, c(i)~=0 =>задача не разрешима');

            if (pos + neg) != 0:
                if pos * neg == 0:
                    r_nom = ((np.sign(su) * c[i]) / (pos + neg))[0]
                    r = r_nom * np.ones([n, 1])
                    a_sign = (np.sign(inn))
                    tmp = (np.sign(su) * a_sign.T)[np.newaxis].T
                    r = r ** tmp
                    a = a * r
                else:
                    if pos > 0:
                        r_nom = (c(i) + np.sqrt(c(i) ^ 2 + 4. * pos * neg)) / (2. * pos)
                        r = r_nom * np.ones([n, 1])
                        a_sign = (np.sign(inn))
                        r = r ** (a_sign.T)
                        a = a * r
                    else:
                        if c(i) != 0:
                            r = (-neg / c(i)) * np.ones([n, 1])
                            a_sign = (np.sign(inn))
                            r = r ** (a_sign.T)
                            a = a * r
    ret = np.delete(ret, 0, 0)
    return [a, ret]