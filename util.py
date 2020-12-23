import numpy as np
import numpy.linalg as LA
import torch
import random

def weight_analysis(w1, w2):
    rela_transpose = LA.norm(np.transpose(w1) - w2)
    print("transpose distance: ", rela_transpose)
    rela_inverse = LA.norm(LA.pinv(w1) - w2)
    print("inverse distance: ", rela_inverse)


def try_construct():
    nr, nc = 128, 128

    x = np.random.rand(nc, 1)
    A = np.random.rand(nr, nc)

    y = np.matmul(A, x)

    Ai = np.linalg.pinv(A)
    xr = np.matmul(Ai, y)

    print(np.linalg.norm(xr-x))

def set_seed_pytorch(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__=='__main__':
    for _ in range(100):
        try_construct()


