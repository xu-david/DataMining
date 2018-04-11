#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

'''
CSCI 57300 Assignment 1
By: David Xu (yx5)

Usage: ./assign1-yx5.py /path/to/input/file
'''

import os
import numpy as np
import numpy.linalg as linalg
from sys import argv
from csv import reader as csvreader

def read_input(inputfile):
    '''
    Read input CSV into NumPy array and transpose. Assumes last column in the
    input data is for classification and is ignored.
    '''
    l = []
    with open(inputfile) as f:
        for line in csvreader(f, delimiter=',', quotechar='"'):
            l.append(list(float(x) for x in line[:-1]))
    return np.array(l).T

def center_matrix(d):
    '''
    Subtracts the variable mean from the input matrix to center
    '''
    mean = d.mean(axis=1)
    return d - mean[:, None]

def covariance_matrix(d):
    '''
    Generates the covariance matrix by cell
    '''
    m = np.shape(d)[0]

    #Populate an empty matrix with size
    c = np.empty([m, m])

    #Calculate the dot product in each cell of the matrix
    for i in range(m):
        for j in range(m):
            if i < j:
                continue
            a = d[i]
            b = d[j]
            covariance = np.dot(a, b)/len(a)
            c[i][j] = covariance
            c[j][i] = covariance
    return c

def power_iter_eig(d, maxiter=1000, thres=0.0001):
    '''
    Power iteration method for dominant eigenvalue with maximum iterations
    'maxiter' and threshold 'thres'.
    '''
    m = len(d)
    #Initial vector x0
    xi = np.random.rand(m)
    xim = np.random.rand(m)

    for i in range(maxiter): #Max iterations
        xi1, xi1m = xi[:], xim
        xi = np.dot(d, xi1)

        #Normalize by x_i, m
        xim = max(abs(x) for x in xi)
        xi = [ xi[x] / xim for x in range(m) ]

        #Check for convergence
        xnorm = linalg.norm(np.array(xi) - np.array(xi1))
        if xnorm < thres:
            break
    else:
        print __doc__
        raise SystemExit('ERROR: Power iteration method failed to converge')

    evalue = xim/xi1m ##Ratio x_i,m/x_i-1,m
    evector = xi/linalg.norm(np.array(xi)) #Norm
    return evalue, evector

def project_evector(d, sorted_eig, ppoints=False, npoints=10):
    '''
    Project samples onto new subspace and find variance
    '''
    w = np.vstack(i[1] for i in sorted_eig).T

    #y = wTx
    #y = w.T.dot(d)
    y = np.dot(w.T, d)
    if ppoints:
        print 'Projection onto first {0} data points:'.format(npoints)
        print y.T[:npoints]
    else:
        ycov = covariance_matrix(y)
        print 'Covariance matrix of new subspace:\n', ycov
    return

def eval_eig(evalues, evectors):
    '''
    Return sorted eigenvalues and eigenvectors as tuples
    '''
    sorted_eig = [(evalues[i], evectors[:, i]) for i in range(len(evalues))]
    sorted_eig = sorted(sorted_eig, reverse=True)
    return sorted_eig

def pca(inputfile, thres_var=0.8):
    '''
    PCA wrapper function
    '''
    d = read_input(inputfile) #Read input
    d = center_matrix(d) #Center data
    c = covariance_matrix(d) #Calculate covariance matrix
    evalues, evectors = linalg.eig(c) #Calculate eigs
    sorted_eig = eval_eig(evalues, evectors) #Sort eigs

    #Evaluate variance
    print 'Evaluating cumulative contribution to overall variance:'
    print 'Threshold: {0}%'.format(round(thres_var*100, 2))
    total_evalues = sum(evalues)
    for i, j in enumerate(sorted_eig, start=1):
        ivar = sum([ sorted_eig[x][0] for x in range(i) ])/total_evalues
        print 'Eigenvector {0}: {1}%'.format(i, round(ivar*100, 2))
        if ivar > thres_var:
            print
            break
    project_evector(d, sorted_eig[:i], ppoints=True)
    return

def eigen_decomp(c, sorted_eig):
    '''
    Eigen decomposition of the covariance matrix CV=V x Lambda
    '''
    evalues = [sorted_eig[i][0] for i in range(len(sorted_eig))]
    evectors = [sorted_eig[i][1] for i in range(len(sorted_eig))]
    V = np.vstack(evectors).T
    Lambda = np.diag(evalues)
    # C = V x Lambda x V^T
    print 'V\n', V
    print 'Lambda\n', Lambda
    print 'V^T\n', V.T

    #Compare covariance with decomposition matrix
    compare_c_vLambdavT = np.allclose(c, np.dot(np.dot(V, Lambda), V.T))
    print 'Comparison with covariance', compare_c_vLambdavT
    return

def main(inputfile):
    #Read the input data into array
    dinput = read_input(inputfile)

    #Center the matrix on variable axis
    dinput = center_matrix(dinput)

    #Part a. Calculate the covariance matrix
    dcov = covariance_matrix(dinput)
    print 'Part A. Calculate the covariance matrix'
    print dcov
    print '-' * 60 + '\n'

    #Part b. Find the dominant eigenvalue and eigenvector from cov matrix
    evalue, evector = power_iter_eig(dcov)
    print 'Part B. Find the dominant eigenvalue and eigenvector'
    print 'Dominant eigenvalue:', evalue
    print 'Dominant eigenvector:', evector
    print '-' * 60 + '\n'

    #Part c. linalg.eig to find the two dominant eigenvectors
    print 'Part C. The two dominant eigenvectors'
    evalues, evectors = linalg.eig(dcov)
    sorted_eig = eval_eig(evalues, evectors)
    project_evector(dinput, sorted_eig[:2])
    print '-' * 60 + '\n'

    #Part d. Decomposition of covariance matrix into eigen-decomposition form
    print 'Part D. Decomposition of covariance matrix'
    eigen_decomp(dcov, sorted_eig)
    print '-' * 60 + '\n'

    #Part e & f. PCA function wrapper
    print 'Parts E and F.'
    pca(inputfile, thres_var=0.9)
    return

if __name__ == '__main__':
    if len(argv) != 2:
        print __doc__
        raise SystemExit('ERROR: Specify path to input file')
    FilePath = argv[1]
    if os.path.exists(FilePath):
        main(FilePath)
    else:
        print __doc__
        raise SystemExit('ERROR: Input file does not exist!')
