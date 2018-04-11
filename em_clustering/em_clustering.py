#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CSCI573000 Assignment 2
By: David Xu
Usage: ./xu-assign2.py <input_file> <k>
"""
import os
import numpy as np
from sys import argv
from math import ceil
from csv import reader as csvreader

def read_input(infile):
    '''
    Read input file into two arrays
    '''
    a, b = [], []
    with open(infile) as f:
        for line in csvreader(f, delimiter=',', quotechar='"'):
            if line:
                a.append([float(x) for x in line[:-1]])
                b.append(line[-1])
    return np.array(a).T, np.array(b)

def slice_matrix(l, n, s):
    '''
    Slice matrix l of size n into chunks of size s
    '''
    for i in range(0, n, s):
        yield l[:, i:i + s]

def multi_norm_dist(x, mu, sigma):
    '''
    Probability density function for multivariate normal random variable
    '''
    dd = np.shape(mu)[0]
    # (2 * pi)^(-k/2) * det(sigma)^(-0.5)
    a = (2*np.pi)**(dd/-2.0) * np.linalg.det(sigma)**-0.5

    # e^(-0.5 * (x-mu) * inv(sigma) * (x-mu)^T)
    xmu = (x - mu)
    b = np.e**( -0.5 * np.dot(np.dot(xmu, np.linalg.inv(sigma)), xmu.T) )
    return a * b

def initialize(d, k):
    '''
    Divide input array into k clusters
    '''
    lmu, lcov, lprob = [], [], []
    n = np.shape(d)[1]
    csize = int(ceil(float(n) / k)) #Number of samples in each cluster

    for i in slice_matrix(d, n, csize):
        mu = i.mean(axis=1) #mean
        cov = np.cov(i, bias=1) #sigma
        prob = 1.0/k #probability
        lmu.append(mu)
        lcov.append(cov)
        lprob.append(prob)
    return lmu, lcov, lprob

def step_expectation(d, l1, k):
    '''
    Calculate posterior probabilities
    '''
    l2 = np.zeros([np.shape(d)[1], k])
    #for i, (mu, sigma, prob) in enumerate(l1):
    for i in range(k):
        mu, sigma, prob = l1[0][i], l1[1][i], l1[2][i]
        for j, x in enumerate(d.T):
            l2[j][i] = multi_norm_dist(x, mu, sigma) * prob
    else:
        for i, j in enumerate(l2):
            a = sum(j)
            l2[i] = j/a #Posterior probability P^t(C_i | x_j)
    return l2.T #w^T_ij

def step_maximization(d, l1, k, l2):
    '''
    Re-estimation of mean, covariance, and probabilities
    '''
    d = d.T
    lmu, lsigma, lprob = [], [], []
    for i, x in enumerate(l2):
        #Re-estimate mean
        mnum = 0.0
        for j in range(len(d)):
            #print i, j, l2[i][j], d[j], l2[i][j] * d[j]
            mnum += l2[i][j] * d[j]
        else:
            mu_i = mnum / sum(l2[i])
            lmu.append(mu_i)

        #Re-estimate diagonal of covariance
        snum = 0.0
        for j in range(len(d)):
            #print i, j, l2[i][j], d[j], lmu[i]
            xn = np.zeros((1, np.shape(d)[1]))
            xn += d[j]
            xmu = xn - lmu[i]
            snum += l2[i][j] * xmu * xmu.T
        else:
            sigma_i = snum / sum(l2[i])
            lsigma.append(sigma_i)

        #Re-estimate priors
        prob_i = sum(l2[i]) / len(l2[i])
        lprob.append(prob_i)
    return lmu, lsigma, lprob

def eval_em(k, lold, lnew, epsilon=0.001):
    '''
    Evaluate EM comparing difference in means with epsilon
    '''
    tot = 0.0
    for i in range(k):
        tot += np.linalg.norm(lnew[i] - lold[i])
    if tot <= epsilon:
        return 1
    else:
        return 0

def cluster_membership(l, k):
    '''
    Find cluster membership for each instance
    '''
    dclusters = dict((x, []) for x in range(k))
    dcr = {}
    for i, j in enumerate(l.T):
        #print i, np.argmax(j), j
        dclusters[np.argmax(j)].append(i)
        dcr[i] = np.argmax(j)
    lsize = []
    print '\nCluster Membership:'
    for k, v in sorted(dclusters.items()):
        print ', '.join(str(x + 1) for x in v)
        lsize.append(len(v))

    print '\nSize: {0}'.format(' '.join(str(x) for x in lsize))
    return dcr #Dictionary of sample: cluster

def eval_purity(k, dy, dc):
    '''
    Purity score of clustering based on class labels
    '''
    dcount = {}
    tot = 0
    for i in range(k):
        dcount[i] = {}
    for i, j in enumerate(dy): #Real label
        b = dc.get(i) #Cluster label
        dcount[b][j] = dcount[b].get(j, 0) + 1
    for i, v in dcount.items():
        #Find tuple with most members
        tot += max(v.items(), key=lambda x: x[1])[1]
    score = tot / float(len(dc))
    return round(score, 3)

def main(infile, k):
    '''
    Main function
    '''
    #Read input file into d (features x samples) and dy (labels)
    d, dy = read_input(infile)

    #Initialize clusters
    l1 = initialize(d, k)
    #Expectation-Maximization iteration
    count = 0
    while 1:
        count += 1
        lold = l1[0]
        l2 = step_expectation(d, l1, k)
        l1 = step_maximization(d, l1, k, l2)
        lnew = l1[0]
        #Break on epsilon threshold
        if eval_em(k, lold, lnew):
            break

    #Outputs
    print 'Mean:'
    for i in range(k):
        print [ round(x, 3) for x in lnew[i] ]

    print '\nCovariance Matrices:'
    for i in range(k):
        print np.around(l1[1][i], decimals=3), '\n'

    print 'Iteration count={0}'.format(count)

    dc = cluster_membership(l2, k)

    print '\nPurity Score: {0}'.format(eval_purity(k, dy, dc))
    return

if __name__ == '__main__':
    if len(argv) != 3:
        print __doc__
        raise SystemExit('ERROR: Check input parameters')
    sn, infile, k = argv
    try:
        k = int(k)
    except ValueError:
        print __doc__
        raise SystemExit('ERROR: Invalid number of clusters!')
    if not os.path.exists(infile):
        print __doc__
        raise SystemExit('ERROR: Input file does not exist!')
    main(infile, k)
