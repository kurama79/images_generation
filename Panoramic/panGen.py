import cv2
import numpy as np
import homography

MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15

def registerImages(im1, im2):
    ''' Esta funcion es la que se mostro en clase '''

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

    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    cv2.imshow("Matches", imMatches)

    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC) # OpenCV para comparar
    model = homography.RansacModel()
    h_mind, ran_data = homography.H_from_ransac(points1, points2, model) # No es tan precisa como la libreria

    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, h_mind, (width, height))

    return im1Reg, h, h_mind

def get_im_tranformed(im1, im2, H, name):
    ''' En esta funcion mostramos las transfromaciones individualmente '''

    corners = np.array([[0,0,im2.shape[1]-1,im2.shape[1]-1], [0,im2.shape[0]-1,0,im2.shape[0]-1]])
    corners = np.append(corners, np.array([1]*4).reshape(1,4), axis = 0)

    trans_corners = np.dot(H, corners)
    trans_corners /= trans_corners[-1,:]
    trans_corners = trans_corners[:2,:]

    coord_max = trans_corners.max(axis = 1)
    coord_min = trans_corners.min(axis = 1)

    im_size = tuple(np.ceil(np.maximum(coord_max, coord_max - coord_min)).astype(int))
    t_x = np.absolute(np.minimum(0,coord_min[0]))
    t_y = np.absolute(np.minimum(0,coord_min[1]))
    T = np.array([[1,0,t_x], [0,1,t_y], [0,0,1]])
    H = np.dot(T, H)

    im2_trans = cv2.warpPerspective(im2, H, im_size)
    im1_trans = cv2.warpPerspective(im1, T, im_size)

    overlap = np.logical_and(im1_trans, im2_trans)
    trans_im = np.where(overlap == True, np.maximum(im1_trans, im2_trans), im1_trans + im2_trans)
    cv2.imwrite('Perspectiva'+name+'.jpg', trans_im)

    return trans_im

def panoramic_generation(imRef, im, H):
    ''' Funcion para generar panorama, acomodando imagen de una en una '''

    warped_img = cv2.warpPerspective(im, H, (imRef.shape[1], imRef.shape[0]), flags=cv2.INTER_LINEAR)
    imRef[warped_img > 0] = warped_img[warped_img > 0]

    return imRef

if __name__ == "__main__":

    # Imagen central
    refFilename = "centro.jpeg"
    imCenter = cv2.imread(refFilename, cv2.IMREAD_COLOR)

    # Obteniendo homografias
    im1 = cv2.imread("1.jpeg", cv2.IMREAD_COLOR)
    imReg1, H_c1_cv, H_c1 = registerImages(im1, imCenter)

    im2 = cv2.imread("2.jpeg", cv2.IMREAD_COLOR)
    imReg2, H_c2_cv, H_c2 = registerImages(im2, imCenter)

    im3 = cv2.imread("6.jpeg", cv2.IMREAD_COLOR)
    imReg3, H_c3_cv, H_c3 = registerImages(im3, imCenter)

    im4 = cv2.imread("5.jpeg", cv2.IMREAD_COLOR)
    imReg4, H_34_cv, H_34 = registerImages(im4, im3)

    # Obtener imagenes transformadas
    im1Reg_cv = get_im_tranformed(imCenter, im1, H_c1_cv, '1 cv')
    im2Reg_cv = get_im_tranformed(imCenter, im2, H_c2_cv, '2 cv')
    im3Reg_cv = get_im_tranformed(imCenter, im3, H_c3_cv, '3 cv')
    im4Reg_cv = get_im_tranformed(im3, im4, H_34_cv, '4 cv')

    im1Reg = get_im_tranformed(imCenter, im1, H_c1, '1')
    im2Reg = get_im_tranformed(imCenter, im2, H_c2, '2')
    im3Reg = get_im_tranformed(imCenter, im3, H_c3, '3')
    im4Reg = get_im_tranformed(im3, im4, H_34, '4')

    # Mostrando transformaciones
    cv2.imshow("Transforamcion 1 (OpenCV)", im1Reg_cv)
    cv2.imshow("Transforamcion 2 (OpenCV)", im2Reg_cv)
    cv2.imshow("Transforamcion 3 (OpenCV)", im3Reg_cv)
    cv2.imshow("Transforamcion 4 (OpenCV)", im4Reg_cv)

    cv2.imshow("Transforamcion 1", im1Reg)
    cv2.imshow("Transforamcion 2", im2Reg)
    cv2.imshow("Transforamcion 3", im3Reg)
    cv2.imshow("Transforamcion 4", im4Reg)

    # Generando imagen negra para transformacion
    imPan = np.zeros(shape=(int(2*imCenter.shape[0]), int(2*imCenter.shape[1]), 3))
    w_offset = int(imPan.shape[0]/2 - imCenter.shape[0]/2)
    h_offset = int(imPan.shape[1]/2 - imCenter.shape[1]/2)
    imPan[w_offset:w_offset+imCenter.shape[0], h_offset:h_offset+imCenter.shape[1]] = imCenter

    imPan_cv = np.zeros(shape=(int(2*imCenter.shape[0]), int(2*imCenter.shape[1]), 3))
    w_offset = int(imPan_cv.shape[0]/2 - imCenter.shape[0]/2)
    h_offset = int(imPan_cv.shape[1]/2 - imCenter.shape[1]/2)
    imPan_cv[w_offset:w_offset+imCenter.shape[0], h_offset:h_offset+imCenter.shape[1]] = imCenter

    H_offset = np.eye(3)
    H_offset[0,2] = h_offset
    H_offset[1,2] = w_offset

    # Trasladando la transformacion
    H_c1_cv = np.dot(H_offset, H_c1_cv)
    H_c2_cv = np.dot(H_offset, H_c2_cv)
    H_c3_cv = np.dot(H_offset, H_c3_cv)
    H_34_cv = np.dot(H_34_cv, H_c3_cv)

    H_c1 = np.dot(H_offset, H_c1)
    H_c2 = np.dot(H_offset, H_c2)
    H_c3 = np.dot(H_offset, H_c3)
    H_34 = np.dot(H_34, H_c3)

    # Generando panoramica
    imPan_cv = panoramic_generation(imPan_cv, im1, H_c1_cv)
    imPan_cv = panoramic_generation(imPan_cv, im2, H_c2_cv)
    imPan_cv = panoramic_generation(imPan_cv, im3, H_c3_cv)
    imPan_cv = panoramic_generation(imPan_cv, im4, H_34_cv)

    imPan = panoramic_generation(imPan, im1, H_c1)
    imPan = panoramic_generation(imPan, im2, H_c2)
    imPan = panoramic_generation(imPan, im3, H_c3)
    imPan = panoramic_generation(imPan, im4, H_34)

    # Mostrando panoramica
    cv2.imshow("Panorama (OpenCV)", imPan_cv)
    cv2.imwrite("Parnorama_cv.jpeg", imPan_cv)

    cv2.imshow("Panorama", imPan)
    cv2.imwrite("Parnorama.jpeg", imPan)

    cv2.waitKey(0)