import json
import random
import numpy as np


def gmm_clustering(X, K):
    """
    Train GMM with EM for clustering.

    Inputs:
    - X: A list of data points in 2d space, each elements is a list of 2
    - K: A int, the number of total cluster centers

    Returns:
    - mu: A list of all K means in GMM, each elements is a list of 2
    - cov: A list of all K covariance in GMM, each elements is a list of 4
            (note that covariance matrix is symmetric)
    """

    # Initialization:
    pi = []
    mu = []
    cov = []
    for k in range(K):
        pi.append(1.0 / K)
        mu.append(list(np.random.normal(0, 0.5, 2)))
        temp_cov = np.random.normal(0, 0.5, (2, 2))
        temp_cov = np.matmul(temp_cov, np.transpose(temp_cov))
        cov.append(list(temp_cov.reshape(4)))

    ### you need to fill in your solution starting here ###
    X_np=np.array(X)
    mu_np=np.array(mu)
    N=len(X)
    D=len(X[0])
    gamma=np.zeros((N,K))
    # Run 100 iterations of EM updates
    for t in range(100):
        for n in range(N):
            for k in range(K):
                gamma[n,k]=posterier(k,X_np[n],pi, mu_np,cov)
        pi=gamma.sum(0)/gamma.sum()
        mu_np=gamma.T.dot(X)
        for i in range(K):
            mu_np[i,:]=mu_np[i,:]/(gamma.sum(0)[i])
        cov=[]
        for k in range(K):
            cov_k=np.zeros((D,D))
            for n in range(N):
                data=X_np[n].reshape((D,1))
                mu_k=mu_np[k].reshape((D,1))
                cov_k+=gamma[n,k]*(data-mu_k).dot((data-mu_k).T)
            cov_k=cov_k/gamma.sum(0)[k]
            cov.append(list(cov_k.reshape(4)))
    mu=[]
    for k in range(K):
        mu.append(list(mu_np[k,:]))
    return mu, cov

def posterier(k,data,pi,mu,sigma):
    denominator=0
    K=len(mu)
    for i in range(0,K):
        sigma_i=np.array(sigma[i]).reshape((2,2))
        denominator=denominator+pi[i]*mvnpdf(data,mu[i],sigma_i)
    sigma_k=np.array(sigma[k]).reshape((2,2))
    return pi[k]*mvnpdf(data,mu[k],sigma_k)/denominator

def mvnpdf(data,mu,sigma):
    D=len(mu)
    p=np.power(2*np.pi,-D/2)*np.power(np.linalg.det(sigma),-1/2)*np.exp(-0.5*(data-mu).T.dot(np.linalg.inv(sigma)).dot(data-mu))
    return p


def main():
    # load data
    with open('hw4_blob.json', 'r') as f:
        data_blob = json.load(f)

    mu_all = {}
    cov_all = {}

    print('GMM clustering')
    for i in range(5):
        np.random.seed(i)
        mu, cov = gmm_clustering(data_blob, K=3)
        mu_all[i] = mu
        cov_all[i] = cov

        print('\nrun' + str(i) + ':')
        print('mean')
        print(np.array_str(np.array(mu), precision=4))
        print('\ncov')
        print(np.array_str(np.array(cov), precision=4))

    with open('gmm.json', 'w') as f_json:
        json.dump({'mu': mu_all, 'cov': cov_all}, f_json)


if __name__ == "__main__":
    main()