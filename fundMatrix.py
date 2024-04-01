'''
    Tarea 5.2 - Computer vision, two-view geometry
        En esta tarea realizaremos el codigo para obtener la matriz fundamental, realizando nuestra propia funcion.
'''
import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import * 
import fundamental

MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15

# Esta funcion es la que hemos esatdo usando para obtener los puntos de "matcheo"
def registerImages(im1Gray, im2Gray):

    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    matches.sort(key=lambda x: x.distance, reverse=False)

    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # imMatches = cv2.drawMatches(im1Gray, keypoints1, im2Gray, keypoints2, matches, None)
    # cv2.imshow("Matches", imMatches)

    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    F, mask = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC) # OpenCV para comparar

    # Obtencion de F propia para compracion...

    model = fundamental.RansacModel()
    F_mind, ran_data = fundamental.F_from_ransac(points2, points1, model) # Sigue siendo mejor el de OpenCV

    pts1 = points1[mask.ravel()==1]
    pts2 = points2[mask.ravel()==1]

    pts1_mind = points1[ran_data]
    pts2_mind = points2[ran_data]

    return pts1_mind, pts2_mind, F_mind, pts1, pts2, F

def drawlines(im1, im2, lines, pts1, pts2):
    # Tomado del codigo presentado en clase

    r, c = im1.shape
    nimg1 = cv2.cvtColor(im1, cv2.COLOR_GRAY2BGR)
    nimg2 = cv2.cvtColor(im2, cv2.COLOR_GRAY2BGR)

    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        cv2.line(nimg1, (x0, y0), (x1, y1), color, 1)
        cv2.circle(nimg1, tuple(pt1), 5, color, -1)
        cv2.circle(nimg2, tuple(pt2), 5, color, -1)

    return nimg1, nimg2

if __name__ == "__main__":
    # Basandonos el codigo visto en la sesion: 'Computer vision, two-view geometry' 

    img1 = cv2.imread('Par 1-1.jpeg', 0) # Seleccionar imagenes
    img2 = cv2.imread('Par 1-2.jpeg', 0)

    pts1, pts2, F, pts1_cv, pts2_cv, F_cv = registerImages(img1, img2)

    ''' Esta parte tambien corresponde al codigo visto en clase...
            Encontrando epilines correspondientes a la segunda imagen imagen y dibuijando sus lineas en imagen 1
    '''
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    img5, img6 = drawlines(img1, img2, lines1, pts1, pts2)

    lines1_cv = cv2.computeCorrespondEpilines(pts2_cv.reshape(-1, 1, 2), 2, F_cv)
    lines1_cv = lines1_cv.reshape(-1, 3)
    img5_cv, img6_cv = drawlines(img1, img2, lines1_cv, pts1_cv, pts2_cv)
    #   Encontrando epilines correspondientes a la segunda imagen imagen y dibuijando sus lineas en imagen 1
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    img3, img4 = drawlines(img2, img1, lines2, pts2, pts1)

    lines2_cv = cv2.computeCorrespondEpilines(pts1_cv.reshape(-1, 1, 2), 1, F_cv)
    lines2_cv = lines2_cv.reshape(-1, 3)
    img3_cv, img4_cv = drawlines(img2, img1, lines2_cv, pts2_cv, pts1_cv)

    # Mostrando las imagenes con sus respectivas epilineas
    fig1, ax1 = plt.subplots(dpi=216)
    plt.subplot(121), plt.imshow(img5)
    plt.subplot(122), plt.imshow(img3)

    fig2, ax2 = plt.subplots(dpi=216)
    plt.subplot(121), plt.imshow(img5_cv)
    plt.subplot(122), plt.imshow(img3_cv)

    plt.show()