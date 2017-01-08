# -*- coding: utf-8 -*-

import numpy as np
import numba
import time
import sys

def test1(a,b,N):
    y = np.zeros(N*N).reshape([N,N])
    for i in range(N):
        for j in range(N):
            for c in range(N):
                y[i][j] += a[i][c] * b[c][j]
                
    return y

@numba.jit
def test2(a,b,N):
    y = np.zeros(N*N).reshape([N,N])
    for i in range(N):
        for j in range(N):
            for c in range(N):
                y[i][j] += a[i][c] * b[c][j]
    return y

def test3(a,b,N):
    return np.dot(a,b)

if __name__ == "__main__":
    args=sys.argv

    np.random.seed(0)
    N=10
    if len(args) == 2:
        N=(int)(args[1])
    a = np.random.random((N,N))
    b = np.eye(N,N)

    ITER=10
    timeList=[]
    timeList.append(N)
    time1 = time.time()
    for i in range(ITER):
        result=test1(a,b,N) 
    time2 = time.time()
    timeList.append((time2-time1)/ITER)
    print('test1:',(time2-time1)/ITER)

    time1 = time.time()
    for i in range(ITER):
        result=test2(a,b,N)
    time2 = time.time()
    timeList.append((time2-time1)/ITER)
    print('test2:',(time2-time1)/ITER)
    time1 = time.time()
    for i in range(ITER):
        result=test3(a,b,N)
    time2 = time.time()
    timeList.append((time2-time1)/ITER)
    print('test3:',(time2-time1)/ITER)
