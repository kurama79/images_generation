'''
    Tarea 5.1 - Computer vision, views of planar scenes
        En esta tarea realizaremos el codigo para generar una nueva imagen (virtual) partiendo de dos imagenes,
        obtendremos su homografia, sera descompuesta y se propondra una rotacion y traslacion diferente.
'''
import cv2
import numpy as np
import calibration
from scipy.stats import special_ortho_group

MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15

# Esta funcion es la que se mostro en clase para obtener la homografia
def registerImages(im1, im2):
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    matches.sort(key=lambda x: x.distance, reverse=False)

    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    # cv2.imshow("Matches", imMatches)

    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))

    return im1Reg, h

# Esta funcion esta basada en el material de la sesion 09...
def descomposition_H(H, K):
    # Obteniendo la homografia euclidiana
    H_new = np.dot(np.dot(np.linalg.inv(K), H), K)

    U, S, V = np.linalg.svd(H_new, full_matrices=True)
    s1 = S[0]/S[1]
    s3 = S[2]/S[1]
    a1 = np.sqrt(1.0 - s3**2)
    b1 = np.sqrt(s1**2 - 1.0)
    m1 = np.array([a1, b1])
    m1 /= np.linalg.norm(m1)
    m2 = np.array([1.0 + s1*s3, a1*b1])
    m2 /= np.linalg.norm(m2)

    V = np.transpose(V)
    v1 = V[:, [0]]
    v3 = V[:, [2]]

    # Posibles vectores normales
    n1 = m1[1]*v1 - m1[0]*v3
    n2 = m1[1]*v1 + m1[0]*v3

    V = np.transpose(V)
    R1 = U.dot(np.array([[m2[0], 0.0, m2[1]], [0, 1.0, 0], [-m2[1], 0, m2[0]]]))
    R1 = R1.dot(V)
    R2 = U.dot(np.array([[m2[0], 0.0, -m2[1]], [0, 1.0, 0], [m2[1], 0, m2[0]]]))
    R2 = R2.dot(V)

    t1 = -R1.dot((m1[1]/s1)*v1 + (m1[0]/s3)*v3)
    t1 /= np.linalg.norm(t1)
    t2 = -R2.dot((m1[1]/s1)*v1 - (m1[0]/s3)*v3)
    t2 /= np.linalg.norm(t2)

    # Criterio para seleccionar la solucion correcta depende de la posicion de las imagenes 
    if H[1,2] < 0: # Si la imagen fue tomada a la derecha de la imagen central
        R = R2
        t = t2
        n = n2

    else: # De la izquierda
        R = R1
        t = t1
        n = n1

    return R, t, n

# Funcion para generar rotacion y traslacion
def ext_parameters(R, t):

    # En esta funcion simplemente inverti la vista pero se puede generar cualquier otro tipo de rotacion 
    # y traslacion, ya sea una intermedia entre las dos vistas, etc...
    
    R_virtual = -R
    t_virtual = -t

    return R_virtual, t_virtual

# Inicio de programa principal..........................................................................
if __name__ == "__main__":

    # Primero obtenemos K de la calibracion de la camara
    K = calibration.get_k_calibration()

    # Obetenmos imagenes
    imCenter = cv2.imread("1.jpeg", cv2.IMREAD_COLOR)
    im1 = cv2.imread("3.jpeg", cv2.IMREAD_COLOR)

    # Obteniendo la homografia
    imReg, H = registerImages(im1, imCenter)

    # Descomponemos H
    R, t, n = descomposition_H(H, K)

    R_virtual, t_virtual = ext_parameters(R, t)

    H_cal = R_virtual - np.dot(t_virtual, np.transpose(n))
    H_new_image = np.dot(np.dot(K, H_cal), np.linalg.inv(K))

    height, width, channels = imCenter.shape
    im1Reg = cv2.warpPerspective(imCenter, H_new_image, (width, height), flags=cv2.INTER_LINEAR)

    cv2.imwrite("virtual_image.jpeg", im1Reg)
    cv2.imshow('Imagen', im1Reg)
    cv2.waitKey(0)