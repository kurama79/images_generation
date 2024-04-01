import numpy as np
import cv2 as cv
import glob

def get_k_calibration():
	print('[INF] Images to be used for calibration:')
	images = glob.glob('calImages/*.jpeg')
	print(images)

	# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
	objp = np.zeros((6*9,3), np.float32)
	objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

	# Arrays to store object points and image points from all the images.
	objpoints = [] # 3d point in real world space
	imgpoints = [] # 2d points in image plane.
	filenames = []

	# Termination criteria (for spotting the corners in the chessboard )
	criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

	for fname in images:
	    # Read the file
	    img = cv.imread(fname)
	    
	    # Convert it to gray
	    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	    
	    # Find the chess board corners
	    ret, corners = cv.findChessboardCorners(gray, (9,6), None) # Cambiar para los circulos
	    
	    # If found, add object points, image points (after refining them)
	    if ret == True:
	        filenames.append(fname)
	        objpoints.append(objp)
	        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
	        imgpoints.append(corners2)
	        # Draw and display the corners
	        cv.drawChessboardCorners(img, (9,6), corners2, ret)
	        cv.imshow('Image', img)
	        # cv.waitKey(-1)
	    else:
	        print("[ERR] Could not perform the detection")
	        
	cv.destroyAllWindows()

	rms_error, K, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

	return K

if __name__ == '__main__':
	K = get_k_calibration()
	print(K)