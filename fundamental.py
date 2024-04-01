import numpy
from numpy import *

# Retomando de la tarea anterior y haciendo modificaciones para obtener la fundamental

class RansacModel(object):
    ''' Esta clase se utiliza bajo la propuesta del libro:
        Programming Computer Vision with Python
        y nos apoyamos de: 
        http://www.scipy.org/Cookbook/RANSAC '''

    def __init__(self, debug=False):
        self.debug = debug

    def get_error(self, data, F):

        x1 = data[:int(len(data)/2)].T
        x2 = data[int(len(data)/2):].T

        Fx1 = dot(F, x1)
        Fx2 = dot(F, x2)
        denom = Fx1[0]**2 + Fx1[1]**2 + Fx2[0]**2 + Fx2[1]**2
        error = (diag(dot(x1.T, dot(F, x2))))**2 / denom

        return error

def F_from_ransac(fp, tp, model, maxiter=1000, match_theshold=0.05):
    ''' fp y tp -> son los puntos match
        model -> se obtiene de la clase RansacModel
        maxiter -> es el numero maximo de iteraciones
        match_theshold -> la tolerancia para considerar el consenso (error)
        '''

    fp = hstack((fp, ones((size(fp,0), 1))))
    tp = hstack((tp, ones((size(tp,0), 1))))

    F, ransac_data = ransac(fp, tp, model, 8, maxiter, match_theshold, 20, return_all=True)

    return F, ransac_data['inliers']

def ransac(data_1, data_2, model, n, k, t, d, debug=False, return_all=False):

    ''' Basado en el pseudo codigo de:
            http://en.wikipedia.org/w/index.php?title=RANSAC&oldid=116358182 '''

    iterations = 0
    bestfit = None
    besterr = numpy.inf
    best_inlier_idxs = None

    while iterations < k:
        maybe_idxs, test_idxs = random_partition(n,data_1.shape[0])
        maybeinliers = vstack((data_1[maybe_idxs,:], data_2[maybe_idxs,:]))
        test_points = vstack((data_1[test_idxs], data_2[test_idxs]))

        maybemodel_1 = normalize_fundamental(data_1[maybe_idxs,:], data_2[maybe_idxs,:])
        test_err = model.get_error( test_points, maybemodel_1)

        also_idxs = test_idxs[test_err < t]
        alsoinliers = vstack((data_1[also_idxs,:], data_2[also_idxs,:]))

        if len(alsoinliers) > d:
            betterdata_1 = numpy.concatenate( (maybeinliers[:8], alsoinliers[:int(len(alsoinliers)/2)]) )
            betterdata_2 = numpy.concatenate( (maybeinliers[8:], alsoinliers[int(len(alsoinliers)/2):]) )
            betterdata = vstack((betterdata_1, betterdata_2))

            bettermodel_1 = normalize_fundamental(betterdata_1, betterdata_2)
            better_errs = model.get_error( betterdata, bettermodel_1)
            thiserr = numpy.mean( better_errs )

            if thiserr < besterr:
                bestfit = bettermodel_1
                besterr = thiserr

                best_inlier_idxs = numpy.concatenate( (maybe_idxs, also_idxs) )

        iterations+=1

    if return_all:
        return bestfit, {'inliers':best_inlier_idxs}
    else:
        return bestfit

def random_partition(n, n_data):
    # Seleccion aleatoria de los puntos 
    
    all_idxs = numpy.arange( n_data )
    numpy.random.shuffle(all_idxs)
    idxs1 = all_idxs[:n]
    idxs2 = all_idxs[n:]
    return idxs1, idxs2

def normalize_fundamental(p1, p2): # Pasar puntos 
    # Normalizamos puntos y obtemos F (matriz fundamental)

    p1 = p1.T
    p2 = p2.T
    n = p1.shape[1]

    p1 = p1 / p1[2]
    mean_1 = mean(p1[:2], axis=1)
    S1 = sqrt(2) / std(p1[:2])
    T1 = array([[S1, 0, -S1*mean_1[0]], [0, S1, -S1*mean_1[1]], [0, 0, 1]])
    p1 = dot(T1,p1)
    
    p2 = p2 / p2[2]
    mean_2 = mean(p2[:2],axis=1)
    S2 = sqrt(2) / std(p2[:2])
    T2 = array([[S2, 0, -S2*mean_2[0]], [0, S2, -S2*mean_2[1]], [0, 0, 1]])
    p2 = dot(T2,p2)

    A = zeros((n,9))
    for i in range(n):
        A[i] = [p1[0,i]*p2[0,i], p1[0,i]*p2[1,i], p1[0,i]*p2[2,i],
                p1[1,i]*p2[0,i], p1[1,i]*p2[1,i], p1[1,i]*p2[2,i],
                p1[2,i]*p2[0,i], p1[2,i]*p2[1,i], p1[2,i]*p2[2,i] ]
            
    U,S,V = linalg.svd(A)
    F = V[-1].reshape(3,3)
        
    U,S,V = linalg.svd(F)
    S[2] = 0
    F = dot(U, dot(diag(S), V))

    F = dot(T1.T, dot(F, T2))

    return F/F[2,2]