import numpy as np

def parse_agg(codes):

    sectr = np.zeros(23)#for rows
    sumsr = np.zeros(73)
    secir = -1
    sumsir = -1

    for i in codes:
        if i[0] != 0 and i[1] != 0 and i[2] == 0 and i[3] == 0:
            secir = secir + 1
        
        if i[0] == 0 and i[1] != 0 and i[2] != 0 and i[3] == 0:
            sumsir = sumsir + 1
        
        if i[0] == 0 and i[1] == 0 and i[2] == 0 and i[3] != 0 and i[4] != 0 and i[5] != '**' and i[5] != '‡':
            sectr[secir]+=1
            sumsr[sumsir]+=1
        
        
        
    sectc = np.zeros(23)#for columns
    sumsc = np.zeros(73)
    secic = -1
    sumsic = -1

    for i in codes:
        if i[0] != 0 and i[1] != 0 and i[2] == 0 and i[3] == 0:
            secic = secic + 1
        
        if i[0] == 0 and i[1] != 0 and i[2] != 0 and i[3] == 0:
            sumsic = sumsic + 1
        
        if i[0] == 0 and i[1] == 0 and i[2] == 0 and i[3] != 0 and i[4] != 0 and i[5] != '††':
            sectc[secic]+=1
            sumsc[sumsic]+=1
        
        
    sumsc = sumsc[sumsc != 0]
    sectc = sectc[sectc != 0]
    sumsc[[51,52]] = sumsc[[52,51]]
    sumsr[[51,52]] = sumsr[[52,51]]


    sectc_fixed = np.zeros(15)
    sectc_fixed[:4] = sectc[:4]
    sectc_fixed[4] = sum(sectc[4:6])#31G
    sectc_fixed[5:9] = sectc[6:10]
    sectc_fixed[9] = sum(sectc[10:12])#FIRE
    sectc_fixed[10]=sum(sectc[12:15])#PROF
    sectc_fixed[11]=sum(sectc[15:17])#6
    sectc_fixed[12]=sum(sectc[17:19])#7
    sectc_fixed[13:]=sectc[19:]


    sectr_fixed = np.zeros(17)
    sectr_fixed[:4] = sectr[:4]
    sectr_fixed[4] = sum(sectr[4:6])#31G
    sectr_fixed[5:9] = sectr[6:10]
    sectr_fixed[9] = sum(sectr[10:12])#FIRE
    sectr_fixed[10]=sum(sectr[12:15])#PROF
    sectr_fixed[11]=sum(sectr[15:17])#6
    sectr_fixed[12]=sum(sectr[17:19])#7
    sectr_fixed[13:]=sectr[19:]


    return sectr_fixed, sectc_fixed, sumsr, sumsc




def make_keym(big, smol, key):
    step = 0
    keym = np.zeros([smol, big])
    for k in range(len(key)):
        for i in range(int(key[k])):
            keym[k][step + i] = 1
        step += int(key[k])
    return keym


def keys_to_g(left, right):

    m = len(left)
    M = len(left[0])
    n = len(right)
    N = len(right[0])

    g = np.zeros([m*n, M*N])

    for i in range(n):
        for j in range(N):
            if right[i][j]:
                g[i*m:(i+1)*m, j*M:(j+1)*M] = left

    return g
