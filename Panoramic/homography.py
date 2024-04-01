import numpy
from numpy import *

class RansacModel(object):
    ''' Esta clase se utiliza bajo la propuesta del libro:
        Programming Computer Vision with Python
        y nos apoyamos de: 
        http://www.scipy.org/Cookbook/RANSAC '''

    def __init__(self, debug=False):
        self.debug = debug

    def get_error(self, data, H):

        fp = data[:int(len(data)/2)]
        fp = hstack((fp, ones((len(fp), 1))))

        tp = data[int(len(data)/2):]
        tp = hstack((tp, ones((len(tp), 1))))

        fp_transformed = dot(H, fp.T)

        for i in range(3):
            fp_transformed[i] /= fp_transformed[2]

        return sqrt(sum((tp.T-fp_transformed)**2, axis=0))

def H_from_ransac(fp, tp, model, maxiter=1000, match_theshold=10):
    ''' fp y tp -> son los puntos match
        model -> se obtiene de la clase RansacModel
        maxiter -> es el numero maximo de iteraciones
        match_theshold -> la tolerancia para considerar el consenso (error)
        '''

    H, ransac_data = ransac(fp, tp, model, 4, maxiter, match_theshold, 25, return_all=True)

    return H, ransac_data['inliers']

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
        maybemodel_1 = getHomography(data_1[maybe_idxs,:], data_2[maybe_idxs,:])
        test_err = model.get_error( test_points, maybemodel_1)
        also_idxs = test_idxs[test_err < t]
        alsoinliers = vstack((data_1[also_idxs,:], data_2[also_idxs,:]))
        if debug:
            print ('test_err.min()',test_err.min())
            print ('test_err.max()',test_err.max())
            print ('numpy.mean(test_err)',numpy.mean(test_err))
            print ('iteration %d:len(alsoinliers) = %d'%(iterations,len(alsoinliers)))
        if len(alsoinliers) > d:
            betterdata_1 = numpy.concatenate( (maybeinliers[:4], alsoinliers[:int(len(alsoinliers)/2)]) )
            betterdata_2 = numpy.concatenate( (maybeinliers[4:], alsoinliers[int(len(alsoinliers)/2):]) )
            betterdata = vstack((betterdata_1, betterdata_2))
            bettermodel_1 = getHomography(betterdata_1, betterdata_2)
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
    ''' Seleccion aleatoria de los puntos '''
    
    all_idxs = numpy.arange( n_data )
    numpy.random.shuffle(all_idxs)
    idxs1 = all_idxs[:n]
    idxs2 = all_idxs[n:]
    return idxs1, idxs2

def getHomography(p1, p2): # Pasar puntos 
    ''' Calculo de Homografia dados 4 puntos por el metodo DLT '''

    A_up = column_stack((p1,ones(p1.shape[0]),zeros((p1.shape[0],3)),-p1[:,0]*p2[:,0],-p1[:,1]*p2[:,0],-p2[:,0]))
    A_below = column_stack((zeros((p1.shape[0],3)),p1,ones(p1.shape[0]),-p1[:,0]*p2[:,1],-p1[:,1]*p2[:,1],-p2[:,1]))

    A = vstack((A_up,A_below))

    result = linalg.svd(A)[-1][-1]
    result = result/result[-1]
    result = result.reshape((p1.shape[1]+1,-1))

    return result