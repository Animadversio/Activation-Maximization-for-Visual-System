# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from numpy.linalg import norm
from numpy.random import randn
from numpy import exp, floor, log, log2, sqrt, zeros, eye, ones, diag
from time import time
import matplotlib.pyplot as plt
# import scipy.integrate as integ
import scipy.stats as stats  # gamma,lognorm,norm
# from numpy.random import choice
# from math import *
import pickle
np.set_printoptions(precision=4, suppress=True)

def frosenbrock(x):
    if len(x) < 2:
        raise('dimension must be greater one')
    # f = sum(x*x)#
    f = 100 * ((x[:-1] ** 2 - x[1:]) ** 2).sum() + ((x[:-1] - 1) ** 2).sum();
    return f


def cmaes(fitnessfunc=frosenbrock, N=10, initX=None, initsgm=0.3, population_size=None,
          maximize=False, stopeval=1e2, stopfitness=None, stopsigma=1e-6, save=False,
          savemrk=''):  # (mu/mu_w, lambda)-CMA-ES
    # User defined input parameters (need to be edited)
    # `fitnessfunc` name of objective/fitness function
    # User defined input parameters (need to be edited)
    # `N` number of objective variables/problem dimension crucial for the problem!
    # objective variables initial point
    # `initsgm` coordinate wise standard deviation (step size)
    # `stopfitness` stop if fitness < stopfitness (minimization)
    #   Use this setting only if the lower bound of the function is known!!! usually we do not know
    # `stopeval` stop after stopeval number of function evaluations
    # `stopsigma` stop if step size is too small

    # --------------------  Initialization --------------------------------
    if initX is None:
        xmean = randn(N, 1)
    else:
        assert len(initX) == N or initX.size == N
        xmean = np.array(initX)
        xmean.shape = (-1, 1)
    sigma = initsgm;
    stopeval = stopeval * N ** 2  # they assume N**2 scaling

    # Strategy parameter setting: Selection
    if population_size == None:
        lambda_ = int(4 + floor(3 * log2(N)));  # population size, offspring number
    else:
        lambda_ = int(population_size)
    # # Note the relation between dimension and population size.
    mu = lambda_ / 2;  # number of parents/points for recombination
    # # Select half the population size as parents
    weights = log(mu + 1 / 2) - (log(np.arange(1, 1 + floor(mu))));  # muXone array for weighted recombination
    mu = int(floor(mu));
    weights = weights / sum(weights);  # normalize recombination weights array
    weights.shape = (-1, 1)  # Reshape the weights mat
    mueff = sum(weights) ** 2 / sum(weights ** 2);  # variance-effectiveness of sum w_i x_i

    # Strategy parameter setting: Adaptation
    cc = (4 + mueff / N) / (N + 4 + 2 * mueff / N)  # time constant for cumulation for C
    cs = (mueff + 2) / (N + mueff + 5)  # t-const for cumulation for sigma control
    c1 = 2 / ((N + 1.3) ** 2 + mueff)  # learning rate for rank-one update of C
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((N + 2) ** 2 + mueff))  # and for rank-mu update
    damps = 1 + 2 * max(0, sqrt((mueff - 1) / (N + 1)) - 1) + cs  # damping for sigma
    # usually close to 1
    # print("cc=%f,cs=%f,c1=%f,cmu=%f,damps=%f"%(cc,cs,c1,cmu,damps))

    # Initialize dynamic (internal) strategy parameters and constants
    pc = zeros((N, 1));
    ps = zeros((N, 1));  # evolution paths for C and sigma
    B = eye(N);  # B defines the coordinate system
    D = ones(N);  # diagonal D defines the scaling
    C = B * diag(D ** 2) * B.T;  # covariance matrix C
    invsqrtC = B * diag(1 / D) * B.T;  # C^-1/2
    D.shape = (-1, 1)
    eigeneval = 0;  # track update of B and D
    chiN = sqrt(N) * (1 - 1 / (4 * N) + 1 / (
                21 * N ** 2));  # expectation of ||N(0,I)|| == norm(randn(N,1)) in 1/N expansion formula

    # -------------------- Generation Loop --------------------------------
    counteval = 0;  # the next 40 lines contain the 20 lines of interesting code  23333
    arx = zeros((N, lambda_))
    arfitness = zeros(lambda_)
    genNo = 0;
    if save is True:
        cacheL = 50
        xmean_cache = zeros((N, cacheL))
        cnt = 0
        iteri = 0
    while counteval < stopeval:
        # Generate and evaluate lambda_ offspring
        for k in range(lambda_):
            arx[:, k:k + 1] = xmean + sigma * B @ (D * randn(N, 1));  # m + sig * Normal(0,C)
            # Clever way to generate multivariate gaussian!!
            # Stretch the guassian hyperspher with D and transform the
            # ellipsoid by B mat linear transform between coordinates
            arfitness[k] = fitnessfunc(arx[:, k]);  # objective function call
            counteval = counteval + 1;

        # Sort by fitness and compute weighted mean into xmean
        if maximize is False:
            arindex = np.argsort(arfitness);  # add - operator it will do maximization.
        else:
            arindex = np.argsort(-arfitness);
        arfitness = arfitness[arindex];  # Ascending order. minimization

        print("Generation %d. sigma: %.2e Fitness scores: Mean:%.2e, Min:%.2e, Max:%.2e" %
              (genNo, sigma, arfitness.mean(), arfitness.min(), arfitness.max()))
        print(arfitness)

        xold = xmean;
        xmean = arx[:, arindex[0:mu]] @ weights;  # recombination, new mean value
        # sliced mat * vect = mean col of feature
        print("Ground Truth Mean L1 Distance per axis:%.2e" % (np.abs(xmean-1).mean()))
        # Cumulation: Update evolution paths
        ps = (1 - cs) * ps + sqrt(cs * (2 - cs) * mueff) * invsqrtC @ (xmean - xold) / sigma;
        hsig = norm(ps) / chiN / sqrt(1 - (1 - cs) ** (2 * counteval / lambda_)) < (1.4 + 2 / (N + 1));
        pc = (1 - cc) * pc + hsig * sqrt(cc * (2 - cc) * mueff) * (xmean - xold) / sigma;

        # Adapt covariance matrix C
        artmp = (1 / sigma) * (arx[:, arindex[0:mu]] - xold);

        C = ((1 - c1 - cmu) * C  # regard old matrix
             + c1 * (pc @ pc.T  # plus rank one update
                     + (1 - hsig) * cc * (2 - cc) * C)  # minor correction if hsig==0
             + cmu * artmp @ diag(weights.flat) @ artmp.T);  # plus rank mu update

        # Adapt step size sigma
        sigma = sigma * exp((cs / damps) * (norm(ps) / chiN - 1));

        # figure(2);hold on
        # scatter(arx(1,:),arx(2,:))
        # scatter(xold(1),xold(2),49,'green','MarkerFaceColor','flat')
        # scatter(xmean(1),xmean(2),64,'red','MarkerFaceColor','flat')
        # plot([xold(1),xmean(1)],[xold(2),xmean(2)],'color','k')
        # theta=0:pi/20:2*pi;
        # Y=sigma*[cos(theta);sin(theta)];
        ##Y2=(D.*Y);
        # Y3=B*(D.*Y);
        # plot(xmean(1)+Y3(1,:),xmean(2)+Y3(2,:))
        # axis('equal')

        # Decomposition of C into B*diag(D.^2)*B' (diagonalization)
        # if counteval - eigeneval > lambda_ / (c1 + cmu) / N / 10:  # to achieve O(N^2)
        #     t1 = time()
        #     eigeneval = counteval;
        #     C = np.triu(C) + np.triu(C, 1).T; # (C + C.T) / 2 #  # enforce symmetry
        #     [D, B] = np.linalg.eig(C);  # eigen decomposition, B==normalized eigenvectors
        #     print("Spectrum Range:%.2f, %.2f" % (D.min(), D.max()))
        #     D = sqrt(D);  # D is a vector of standard deviations now
        #     invsqrtC = B @ diag(1 / D) @ B.T;
        #     D.shape = (-1, 1);
        #     t2 = time()
        #     print("Cov Matrix Eigenvalue Decomposition (linalg) time cost: %.2f s" % (t2 - t1))
        if save is True:
            xmean_cache[:, iteri:iteri + 1] = xmean
            iteri = iteri + 1
            if iteri == cacheL:
                pickle.dump((xmean_cache, C, sigma, pc, ps), open('CMAES_%s_optm_rec%d.p' % (savemrk, cnt), "wb"))
                iteri = 0;
                xmean_cache = zeros((N, cacheL));
                cnt = cnt + 1
                print("Savefile %d, obj func %f, sigma:%.2E, CovMat Spectrum range: (%.2E,%.2E)" % (
                cnt, arfitness[0], sigma, max(D), min(D)))

        genNo = genNo + 1;
        # print(xmean)
        # print(C)
        # print(sigma[0])
        # print(pc)
        # print(ps)
        # Break, if fitness is good enough or condition exceeds 1e14, better termination methods areadvisable
        # Note there is a criterion about condition number.
        if (max(D) > 1e7 * min(D)) or (sigma < stopsigma) or (
                (stopfitness is not None) and (arfitness[0] <= stopfitness)):
            break;
    # Final Save
    if save is True:
        pickle.dump((xmean_cache, C, sigma, pc, ps), open('CMAES_%s_optm_rec%d.p' % (savemrk, cnt), "wb"))
        cnt = cnt + 1
        print("%d savefiles, final iteri %d" % (cnt, iteri))
    # while, end generation loop
    # xmin = arx[:, arindex[0]]; # Return best point of last iteration.
    # Notice that xmean is expected to be even
    # better.
    print("%d generations, sigma:%.2E , CovMat Spectrum range: (%.2E,%.2E)" %
          (counteval // lambda_, sigma, max(D), min(D)) )
    return xmean

np.random.seed(seed=0)
cmaes(N=100, initX=[-1] * 100, initsgm=1, population_size=50)

# pickle.dump( (Vtrace,pfinal,fitinfo) , open( 'Simul_%d_Seqfit_V_0A1_Data.p' % (cnt), "wb" ) )
# (Vtrace_e,pfinal_e,fitinfo_e)=pickle.load(open(RsltDir+'Rec_%d_%d_%d_e_Seqfit_0A1_Data.p' % (freqi,splj,trialk), "rb" ))
