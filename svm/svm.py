#!/usr/bin/env python
# -*- coding: utf-8 -*-
############################################################
#
# Title       : CSCI573 Assignment 5
# Author      : David Xu (yx5)
# Description : Python implementation of the SVM training
#               algorithm (Algorithm 21.1). The dual form is
#               solved using stochastic gradient ascent and
#               hinge loss.
#
############################################################

'''
Usage: ./assign5.py -i <inputFile> -k <KernelType>

Available Kernels:
  linear   Linear
  quad     Quadratic (Homogeneous)
'''

import os
import numpy as np
from getopt import getopt, GetoptError
from sys import argv, exit

def read_input_file(inputFile, delimiter=','):
    #Read input file and returns as input data and labels
    if not os.path.exists(inputFile):
        print __doc__
        exit('Error: check input file')

    with open(inputFile) as f:
        flines = np.loadtxt(f, delimiter=delimiter).T
        lx = flines[:-1]
        ly = flines[-1]
    return lx.T, ly

def linear_kernel(x, y):
    #Linear kernel: xTy
    return np.dot(x, y)

def quad_kernel(x, y):
    #Quadratic kernel: ( xTy )**2
    return np.dot(x, y)**2

def svm_dual(lx, ly, kernelType, C=10.0, epsilon=0.0001):
    #Dual SVM algorithm: Stochastic gradient ascent using hinge loss

    #Available kernels
    kernels = {
        'linear' : linear_kernel,
        'quad'   : quad_kernel,
    }

    if kernelType not in kernels:
        print __doc__
        exit('Error: invalid kernel')

    n = np.shape(lx)[0]

    #Map to R^d+1 dimension
    dx1 = np.ones((n, 1))
    lx1 = np.append(lx, dx1, 1)

    #Set step size: eta_k <- 1/K(x_k, x_k)
    etas = np.zeros(n)
    for k in range(n):
        etas[k] = 1. / kernels[kernelType](lx1[k], lx1[k])

    #Stochastic gradient ascent
    alphast = np.zeros(n)
    t = 0
    while 1:
        t += 1
        alphast1 = np.copy(alphast)
        for k in range(n):
            #Update k-th component of alpha
            ayK = 0.0
            for i in range(n):
                ayK += alphast1[i] * ly[i] * kernels[kernelType](lx1[i], lx1[k])
            a_K = alphast1[k] + etas[k] * (1 - ly[k] * ayK)
            #Check bounds
            if a_K < 0.:
                a_K = 0.
            elif a_K > C:
                a_K = C
            alphast1[k] = a_K
        diff = np.linalg.norm(alphast - alphast1)
        alphast = np.copy(alphast1)
        if diff <= epsilon:
            break
        #if not t % 10:
        #    print t, diff

    #Output
    w = np.zeros(np.shape(lx1)[1])
    countsv = 0
    for i, j in enumerate(alphast1):
        if round(j, 12) != 0.: #Non-zero alphas to 12 sigfigs
            countsv += 1
            print '{0}\tx:{1}\ty:{2}\ta:{3}'.format(i+1, lx[i], int(ly[i]), alphast1[i])
        w += np.dot(j * ly[i], lx1[i])
    print '-' * 40
    print 'Iterations: {0}'.format(t)
    print 'Number of Support Vectors: {0}'.format(countsv)
    print 'Weight vector (w): {0}'.format(w[:-1])
    print 'Bias (b): {0}'.format(w[-1])
    return

def main(inputFile, kernelType):
    lx, ly = read_input_file(inputFile) #Read data and labels
    svm_dual(lx, ly, kernelType)
    return

if __name__ == '__main__':
    try:
        opts, args = getopt(argv[1:], 'i:k:')
    except GetoptError as err:
        print __doc__
        exit('Error: {0}'.format(err))
    d = {}
    for o, a in opts:
        d[o] = a
    main(d.get('-i', 'None'), d.get('-k', 'None'))
