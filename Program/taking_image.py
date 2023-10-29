import numpy as np
import cv2

print("[INFO] Starting to take pictures for calibration...")
print("[WARNING] CHECK IF RIGHT IMAGE IS FROM RIGHT CAMERA AND VICE VERSA")
print("[INFO] Press (s) to save image, press (q) to quit")

id_image=0

# termination criteria
criteria =(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Call the two cameras
CamR= cv2.VideoCapture(2)
CamL = cv2.VideoCapture(0)


while True:
    retR, frameR= CamR.read()
    retL, frameL= CamL.read()
    grayR= cv2.cvtColor(frameR,cv2.COLOR_BGR2GRAY)
    grayL= cv2.cvtColor(frameL,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    retR, cornersR = cv2.findChessboardCorners(grayR,(9,6),None)  
    retL, cornersL = cv2.findChessboardCorners(grayL,(9,6),None)  
    cv2.imshow('Image_Right',frameR)
    cv2.imshow('Image_Left',frameL)

    # If found, add object points, image points (after refinding them)
    if (retR == True) & (retL == True):
        corners2R= cv2.cornerSubPix(grayR,cornersR,(11,11),(-1,-1),criteria)  
        corners2L= cv2.cornerSubPix(grayL,cornersL,(11,11),(-1,-1),criteria)
        
        # Draw and display the corners
        cv2.drawChessboardCorners(grayR,(9,6),corners2R,retR)
        cv2.drawChessboardCorners(grayL,(9,6),corners2L,retL)
        cv2.imshow('Chessboard_Right',grayR)
        cv2.imshow('Chessboard_Left',grayL)

        if cv2.waitKey(0) & 0xFF == ord('s'):   # Push "s" to save the images
            print(f"Images {id_image+1} saved for right and left cameras")
            cv2.imwrite(f"images/imageschessboard-R{id_image}.png",frameR) # Save the image in images folder
            cv2.imwrite(f"images/imageschessboard-L{id_image}.png",frameL) # Save the image in images folder
            id_image+=1
        else:
            print('Images not saved')


    if cv2.waitKey(1) & 0xFF == ord('q'):   
        break

# Release the Cameras
CamR.release()
CamL.release()
cv2.destroyAllWindows()    

