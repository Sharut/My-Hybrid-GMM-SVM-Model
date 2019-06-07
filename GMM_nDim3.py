#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 09:08:48 2019

@author: uiet_mac1
"""

import numpy as np
import random as rd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
 

#import hungarian as hg

def random_parameters(data, K):
    """ K is the number of gaussians"""
    """if dimension is d, then mean is dX1"""
    """ init the means, covariances and mixing coefs"""
    cols = (data.shape)[1]
    #print(len(data))
    mu = np.zeros((K, cols))    #mean of k clusters KXD
    for k in range(K):
        idx = np.floor(rd.random()*len(data))
        for col in range(cols):
            mu[k][col] += (data[int(idx)][col])
 
    sigma = []
    for k in range(K):
        sigma.append(np.cov(data.T))
 
    pi = np.ones(K)*1.0/K
    print(mu) 
    print(sigma)
    return mu, sigma, pi


def e_step(data, K, mu, sigma, pi):
    idvs = (data.shape)[0]
    #cols = (data.shape)[1]
    #print("idvs is " +str(idvs))
    resp = np.zeros((idvs, K))
 
    for i in range(idvs):
        for k in range(K):
            resp[i][k] = pi[k]*gaussian(data[i], mu[k], sigma[k])/likelihood(data[i], K, mu, sigma, pi)
    
    #print("responsibitlies is ")
    #print(resp) 
       
    return resp

def log_likelihood(data, K, mu, sigma, pi):
    """ marginal over X """
    log_likelihood = 0.0
    for n in range (len(data)):
        log_likelihood += np.log(likelihood(data[n], K, mu, sigma, pi))
    return log_likelihood 

 
def likelihood(x, K, mu, sigma, pi):
    rs = 0.0
    for k in range(K):
        rs += pi[k]*gaussian(x, mu[k], sigma[k])
    return rs


def m_step(data, K, resp):
    """ find the parameters that maximize the log-likelihood given the current resp."""
    idvs = (data.shape)[0]
    cols = (data.shape)[1]
    
    mu = np.zeros((K, cols))
    sigma = np.zeros((K, cols, cols))
    pi = np.zeros(K)

    marg_resp = np.zeros(K)
    for k in range(K):
        for i in range(idvs):
            marg_resp[k] += resp[i][k]
            mu[k] += (resp[i][k])*data[i]
        mu[k] /= marg_resp[k]

        for i in range(idvs):
            #x_i = (np.zeros((1,cols))+data[k])
            x_mu = np.zeros((1,cols))+data[i]-mu[k]
            sigma[k] += (resp[i][k]/marg_resp[k])*x_mu*x_mu.T

        pi[k] = marg_resp[k]/idvs        
    
    
    return mu, sigma, pi


def gaussian(x, mu, sigma):
    """ compute the pdf of the multi-var gaussian """
    idvs = len(x)
    norm_factor = (2*np.pi)**idvs

    norm_factor *= np.linalg.det(sigma)
    norm_factor = 1.0/np.sqrt(norm_factor)

    x_mu = np.matrix(x-mu)

    rs = norm_factor*np.exp(-0.5*x_mu*np.linalg.inv(sigma)*x_mu.T)
    return rs

def EM(data, rst, K, threshold):
    converged = False
    mu, sigma, pi = random_parameters(data, K)
    likelihood_list=[]
    current_log_likelihood = log_likelihood(data, K, mu, sigma, pi)
    max_iter = 100
    for it in range(max_iter):
        likelihood_list.append(float(current_log_likelihood[0][0]))
        print(rst, "       |       ", it, "     |     ", current_log_likelihood[0][0])
        #print("Mixing proportion is ", pi )
        resp = e_step(data, K, mu, sigma, pi)
        mu, sigma, pi = m_step(data, K, resp)

        new_log_likelihood = log_likelihood(data, K, mu, sigma, pi)
        if (abs(new_log_likelihood-current_log_likelihood) < threshold):
            converged = True
            break

        current_log_likelihood = new_log_likelihood
    print(converged) 
    plt.plot(likelihood_list)
    plt.ylabel('log likelihood')
    plt.show()
           
    return current_log_likelihood, mu, sigma, pi, resp

#######################################################################
def assign_clusters(K, resp):
    idvs = len(resp)
    clusters = np.zeros(idvs, dtype=int)

    for i in range(idvs):
        #clusters[i][k] = 0
        clss = 0
        for k in range(K):
            if resp[i][k] > resp[i][clss]:
                clss = k
                resp[i][clss]= resp[i][k]
                
        clusters[i] = clss

    return clusters
'''
def compute_statistics(clusters, ref_clusters, K):
    mat = make_ce_matrix(clusters, ref_clusters, K)
    #hung_solver = hg.Hungarian()
    rs = hung_solver.compute(mat, False)

    tmp_clusters = np.array(clusters)
    for old, new in rs:
        clusters[np.where(tmp_clusters == old)] = new
        #print old, new

    #print clusters, ref_clusters
    nbrIts = 0
    for k in range(K):
        ref = np.where(ref_clusters == k)[0]
        clust = np.where(clusters == k)[0]
        nbrIts += len(np.intersect1d(ref, clust))
        print(len(np.intersect1d(ref, clust)))
    return nbrIts

def make_ce_matrix(clusters, ref_clusters, K):    
    mat = np.zeros((K, K), dtype=int)    
    for i in range(K):
        for j in xrange(K):
            ref_i = np.where(ref_clusters == i)[0]
            clust_j = np.where(clusters == j)[0]
            its = np.intersect1d(ref_i, clust_j)
            mat[i,j] = len(ref_i) + len(clust_j) -2*len(its)

    return mat
'''
            
########################################################################
def read_data(file_name):
    """ read the data from filename as numpy array """    
    with open(file_name) as f:
        data =  np.loadtxt(f, delimiter=",", dtype = "float", 
                          skiprows=0, usecols=(0,1,2,3))
        
    with open(file_name) as f:
        ref_classes =  np.loadtxt(f, delimiter=",", dtype = "str", 
                                   skiprows=0, usecols=[4])
        unique_ref_classes = np.unique(ref_classes)
        ref_clusters = np.argmax(ref_classes[np.newaxis,:]==unique_ref_classes[:,np.newaxis],axis=0)
       
            
    return data, ref_clusters


def f(t):
    return t

def plot_ellipse(ax, mu, sigma, color="k"):
    """
    Based on
    http://stackoverflow.com/questions/17952171/not-sure-how-to-fit-data-with-a-gaussian-python.
    """

    # Compute eigenX_embeddedues and associated eigenvectors
    X_embeddeds, vecs = np.linalg.eigh(sigma)

    # Compute "tilt" of ellipse using first eigenvector
    x, y = vecs[:, 0]
    theta = np.degrees(np.arctan2(y, x))

    # EigenX_embeddedues give length of ellipse along each eigenvector
    w, h = 2 * np.sqrt(X_embeddeds)

    ax.tick_params(axis='both', which='major', labelsize=20)
    ellipse = Ellipse(mu, w, h, theta, color=color)  # color="k")
    ellipse.set_clip_box(ax.bbox)
    ellipse.set_alpha(0.2)
    ax.add_artist(ellipse)

def error_ellipse(mu, cov, ax=None, factor=1.0, **kwargs):
    """
    Plot the error ellipse at a point given its covariance matrix.

    """
    # some sane defaults
    facecolor = kwargs.pop('facecolor', 'none')
    edgecolor = kwargs.pop('edgecolor', 'k')

    x, y = mu
    U, S, V = np.linalg.svd(cov)
    theta = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
    ellipsePlot = Ellipse(xy=[x, y],
                          width=2 * np.sqrt(S[0]) * factor,
                          height=2 * np.sqrt(S[1]) * factor,
                          angle=theta,
                          facecolor=facecolor, edgecolor=edgecolor, **kwargs)

    if ax is None:
        ax = plt.gca()
    ax.add_patch(ellipsePlot)

    return ellipsePlot

def _plot_gaussian(mean, covariance, color, zorder=0):
    """Plots the mean and 2-std ellipse of a given Gaussian"""
    
    plt.plot(mean[0], mean[1], color[0] + ".", zorder=zorder)

    if covariance.ndim == 1:
        covariance = np.diag(covariance)

    radius = np.sqrt(5.991)
    eigX_embeddeds, eigvecs = np.linalg.eig(covariance)
    axis = np.sqrt(eigX_embeddeds) * radius
    slope = eigvecs[1][0] / eigvecs[1][1]
    angle = 180.0 * np.arctan(slope) / np.pi

    plt.axes().add_artist(Ellipse(
        mean, 2 * axis[0], 2 * axis[1], angle=angle,
        fill=False, color=color, linewidth=1, zorder=zorder
    ))
    
    plt.show()

    
    
    
    
def _plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
        """
        Plots an `nstd` sigma error ellipse based on the specified covariance
        matrix (`cov`). Additional keyword arguments are passed on to the
        ellipse patch artist.

        Parameters
        ----------
            cov : The 2x2 covariance matrix to base the ellipse on
            pos : The location of the center of the ellipse. Expects a 2-element
                sequence of [x0, y0].
            nstd : The radius of the ellipse in numbers of standard deviations.
                Defaults to 2 standard deviations.
            ax : The axis that the ellipse will be plotted on. Defaults to the
                current axis.
            Additional keyword arguments are pass on to the ellipse patch.

        Returns
        -------
            A matplotlib ellipse artist
        """
        from matplotlib import pyplot as plt
        from matplotlib.patches import Ellipse

        def eigsorted(cov):
            X_embeddeds, vecs = np.linalg.eigh(cov)
            order = X_embeddeds.argsort()[::-1]
            return X_embeddeds[order], vecs[:, order]

        if ax is None:
            ax = plt.gca()

        X_embeddeds, vecs = eigsorted(cov)
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

        # Width and height are "full" widths, not radius
        width, height = 2 * nstd * np.sqrt(X_embeddeds)
        ellip = Ellipse(xy=pos, width=width, height=height, angle=theta,
                        **kwargs)
        ax.add_artist(ellip)
        plt.show()
        return ellip

    
    
def main():
    print("begining...")
    file_name = "iris.data"
    nbr_restarts = 5
    threshold = 0.001
    K = 3
    
    data, ref_clusters = read_data(file_name)
    

    print("#restart | EM iteration | log likelihood")
    print("----------------------------------------")

    max_likelihood_score = float("-inf")
    for rst in range(nbr_restarts):
        log_likelihood, mu, sigma, pi, resp = EM(data, rst, K, threshold)
        if log_likelihood > max_likelihood_score:
            max_likelihood_score = log_likelihood
            max_mu, max_sigma, max_pi, max_resp = mu, sigma, pi, resp
            #print("Iteration is"+ str(rst))
            #print("mixing  is ")
            #print(max_pi)
            #print("mean is ")
            #print(max_mu)
            #print("sigma is ")
            #print(max_sigma)
    
    #print(max_mu, max_sigma, max_pi)
    print("mean matrix is ")
    print(max_mu)
    clusters = assign_clusters(K, max_resp)
    #cost = compute_statistics(clusters, ref_clusters, K)
    print(clusters)
    print(ref_clusters)
    #print(cost*1.0/len(data))
    
    
    from mpl_toolkits.mplot3d import Axes3D
    #with first three variables are on the axis and the fourth being color:
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')
    sp = ax.scatter(data[:,0],data[:,1],data[:,2], s=20, c=data[:,3])
    fig.colorbar(sp)
    plt.show()

    
    
    from sklearn.manifold import TSNE
    data = np.concatenate((data,mu),axis = 0)
    print(data)
    X = np.array(data)
    #means = np.array(mu)
    
    
    '''
    X_embedded = TSNE(n_components=1).fit_transform(X)
    print("!!!!")
    figs = plt.figure(figsize=(15, 12))
    plt.plot(X_embedded,'ro')
    plt.plot( X_embedded[150:153],'g^')   
    t1 = np.linspace(0, 140, 100)
    
    plt.plot(t1,[X_embedded[150]]*100 , 'g^')
    plt.plot(t1,[X_embedded[151]]*100 , 'g^')
    plt.plot(t1,[X_embedded[152]]*100 , 'g^')
    
    
    plt.ylabel('some numbers')
    plt.show()
    '''
             
    
             
    X_embedded = TSNE(n_components=2).fit_transform(X)
    print(X_embedded)
    print("!!!!")
    figs = plt.figure(figsize=(15, 12))
    plt.plot(X_embedded[0:150,0], X_embedded[0:150,1],'ro')
    plt.plot( X_embedded[150:153,0],X_embedded[150:153,1] ,'g^')   
    
    
    plt.ylabel('some numbers')
    A = np.matrix(max_sigma[0])
    N, M = A.shape
    assert N % 2 == 0
    assert M % 2 == 0
    A0 = np.empty((N//2, M//2))
    for i in range(N//2):
        for j in range(M//2):
             A0[i,j] = A[2*i:2*i+2, 2*j:2*j+2].sum()
             
    A = np.matrix(max_sigma[1])
    N, M = A.shape
    assert N % 2 == 0
    assert M % 2 == 0
    A1 = np.empty((N//2, M//2))
    for i in range(N//2):
        for j in range(M//2):
             A1[i,j] = A[2*i:2*i+2, 2*j:2*j+2].sum()
             
             
    A = np.matrix(max_sigma[2])
    N, M = A.shape
    assert N % 2 == 0
    assert M % 2 == 0
    A2 = np.empty((N//2, M//2))
    for i in range(N//2):
        for j in range(M//2):
             A2[i,j] = A[2*i:2*i+2, 2*j:2*j+2].sum()
             
    print(A0)
    print(A1)
    print(A2)
    print(X_embedded[150,:])
    #_plot_cov_ellipse(A0,X_embedded[150,:] )
    mean = X_embedded[150,:]
    covariance = A0
    plt.plot(mean[0], mean[1], 'g' + ".", zorder=0)

    if covariance.ndim == 1:
        covariance = np.diag(covariance)

    radius = np.sqrt(5.991)
    eigX_embeddeds, eigvecs = np.linalg.eig(covariance)
    axis = np.sqrt(eigX_embeddeds) * radius
    slope = eigvecs[1][0] / eigvecs[1][1]
    angle = 180.0 * np.arctan(slope) / np.pi

    plt.axes().add_artist(Ellipse(
        mean, 2 * axis[0], 2 * axis[1], angle=angle,
        fill=False, color='g', linewidth=1, zorder=0
    ))
    
    
    
    mean = X_embedded[151,:]
    covariance = A1
    plt.plot(mean[0], mean[1], 'g' + ".", zorder=0)

    if covariance.ndim == 1:
        covariance = np.diag(covariance)

    radius = np.sqrt(5.991)
    eigX_embeddeds, eigvecs = np.linalg.eig(covariance)
    axis = np.sqrt(eigX_embeddeds) * radius
    slope = eigvecs[1][0] / eigvecs[1][1]
    angle = 180.0 * np.arctan(slope) / np.pi

    plt.axes().add_artist(Ellipse(
        mean, 2 * axis[0], 2 * axis[1], angle=angle,
        fill=False, color='g', linewidth=1, zorder=0
    ))
    
    
    
    
    mean = X_embedded[152,:]
    covariance = A2
    plt.plot(mean[0], mean[1], 'g' + ".", zorder=0)

    if covariance.ndim == 1:
        covariance = np.diag(covariance)

    radius = np.sqrt(5.991)
    eigX_embeddeds, eigvecs = np.linalg.eig(covariance)
    axis = np.sqrt(eigX_embeddeds) * radius
    slope = eigvecs[1][0] / eigvecs[1][1]
    angle = 180.0 * np.arctan(slope) / np.pi

    plt.axes().add_artist(Ellipse(
        mean, 2 * axis[0], 2 * axis[1], angle=angle,
        fill=False, color='g', linewidth=1, zorder=0
    ))
    
    plt.show()    
    #_plot_gaussian(X_embedded[150,:], A0,'r')
    #error_ellipse(X_embedded[150,:], A0)
    #plot_ellipse(plt, X_embedded[150,:], A0 )
    
    
    #np.savetxt("mu.txt",max_mu)

    return max_mu

if __name__ == '__main__':
    main()