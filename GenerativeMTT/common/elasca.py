import numpy as np

def elasca(imf):
    imf = np.transpose(imf, axes=(1,2,0))
    a, b, c = imf.shape
    Vm = np.max(imf)
    P = imf / Vm
    B = np.sort(P.flatten())
    IX = np.argsort(P.flatten())
    m = B[0]
    G = np.zeros(a * b * c)

    l = 0
    n = -1
    while n < a * b * c:
        n += 1
        if n == a * b * c:
            m = B[n - 1]
            G[l:n] = n / (a * b * c)
        elif B[n] > m:
            m = B[n]
            G[l:n] = (n - 1) / (a * b * c)
            l = n
            n -= 1

    Result = np.zeros((a * b * c, 2))
    Result[:, 0] = IX.T
    Result[:, 1] = G.T
    GIm = Result[Result[:,0].argsort()]
    NIm = GIm[:, 1]
    ImRec = np.reshape(NIm, (a, b, c))

    return np.transpose(ImRec,axes=(2,0,1)).astype(np.float32)
