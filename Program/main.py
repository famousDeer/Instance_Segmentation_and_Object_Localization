import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import time
from ultralytics import YOLO
from MobileSAM.mobile_sam import sam_model_registry, SamPredictor


# Segmentation config 
model_type = 'vit_t'
checkpoints = 'MobileSAM/weights/mobile_sam.pt'

# Detection config
model_detection = YOLO('weights/best.pt')

# Termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
criteria_stereo= (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points
objp = np.zeros((9*6,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all images
objpoints= []   # 3d points in real world space
imgpointsR= []  # 2d points in image plane from left camera
imgpointsL= []  # 2d points in image plane from right camera

def show_mask(mask, frame, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.7])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    combined_image = cv2.addWeighted(frame, 0.8, mask_image, 0.2, 0)
    
    return combined_image

def calculate_distance(disp, x,y):
    average = 0
    for i in range(-2,3):
        for j in range(-2,3):
            average += disp[y+i, x+j]

    average /= 25
    # distance = -688.89*average**(3) + 1631.2*average**(2) - 1365.9*average + 480.64
    distance = -373.2*average**(3) + 952.48*average**(2) - 972.79*average + 454.29
    distance = np.around(distance*0.01, decimals=2)
    return distance

# Camera calibration
print("[INFO] Starting cameras calibration...")

# Number in range depend on how many image was taken for calibration
for i in range(0,25):
    ChessImaR= cv2.imread(f"images/imageschessboard-R{i}.png")    # Right side
    ChessImaL= cv2.imread(f"images/imageschessboard-L{i}.png")    # Left side
    grayR = cv2.cvtColor(ChessImaR, cv2.COLOR_BGR2GRAY)
    grayL = cv2.cvtColor(ChessImaL, cv2.COLOR_BGR2GRAY)
    retR, cornersR = cv2.findChessboardCorners(grayR,
                                               (9,6),None)  
    retL, cornersL = cv2.findChessboardCorners(grayL,
                                               (9,6),None) 
    if (True == retR) & (True == retL):
        objpoints.append(objp)
        cornersR = cv2.cornerSubPix(grayR,cornersR,(11,11),(-1,-1),criteria)
        cornersL = cv2.cornerSubPix(grayL,cornersL,(11,11),(-1,-1),criteria)
        imgpointsR.append(cornersR)
        imgpointsL.append(cornersL)

# New values after calibration
# Right camera
retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints,
                                                        imgpointsR,
                                                        grayR.shape[::-1],None,None)
hR,wR= grayR.shape[:2]
OmtxR, roiR = cv2.getOptimalNewCameraMatrix(mtxR,distR,
                                                   (wR,hR),1,(wR,hR))
# Left camera
retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints,
                                                        imgpointsL,
                                                        grayL.shape[::-1],None,None)
hL,wL= grayL.shape[:2]
OmtxL, roiL = cv2.getOptimalNewCameraMatrix(mtxL,distL,(wL,hL),1,(wL,hL))

retS, MLS, dLS, MRS, dRS, R, T, E, F = cv2.stereoCalibrate(objpoints,
                                                          imgpointsL,
                                                          imgpointsR,
                                                          OmtxL,
                                                          distL,
                                                          OmtxR,
                                                          distR,
                                                          grayR.shape[::-1],
                                                          criteria = criteria_stereo,
                                                          flags = cv2.CALIB_FIX_INTRINSIC)

# StereoRectify function
rectify_scale= 1 # if 0 image croped, if 1 image not croped
RL, RR, PL, PR, Q, roiL, roiR = cv2.stereoRectify(MLS, dLS, MRS, dRS,
                                                 grayR.shape[::-1], R, T,
                                                 rectify_scale,(0,0)) 
# initUndistortRectifyMap function
Left_Stereo_Map = cv2.initUndistortRectifyMap(MLS, dLS, RL, PL,
                                             grayL.shape[::-1], cv2.CV_16SC2) 
Right_Stereo_Map = cv2.initUndistortRectifyMap(MRS, dRS, RR, PR,
                                              grayR.shape[::-1], cv2.CV_16SC2)

print("[INFO] Cameras calibrations complete.")

# Parameters for stereo vision
window_size = 3
min_disp = 2
num_disp = 130-min_disp
stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
    numDisparities = num_disp,
    blockSize = window_size,
    uniquenessRatio = 5,
    speckleWindowSize = 200,
    speckleRange = 32,
    disp12MaxDiff = 5,
    P1 = 8*3*window_size**2,
    P2 = 32*3*window_size**2,
    mode=2)

# Used for the filtered image
# stereoR=cv2.ximgproc.createRightMatcher(stereo) # Create another stereo for right this time

# Choose device
device = "cuda" if torch.cuda.is_available() else "cpu"

model_detection.to(device)

# Warmup GPU
dummy_input = torch.randn(1,3,640,640)
for _ in range(10):
    _ = model_detection.predict(dummy_input)


# Call both cameras
CamR = cv2.VideoCapture(2)
CamL = cv2.VideoCapture(0)

# Few colors for mask
colors = {"R":[255,200,150,101,32,54,212,4,0], "G":[50,100,200,0,20,34,200,100,101], "B":[0,100,110,90,23,15,150,0,23]}

model_segmentation = sam_model_registry[model_type](checkpoint=checkpoints)
model_segmentation.to(device)
model_segmentation.eval()

frame_count = 0
start_time = time.time()
FPS = 0

while True:
    # Reading images 
    retR, frameR = CamR.read()
    retL, frameL = CamL.read()

    # Rectify the images on rotation and alignement
    Left_good = cv2.remap(frameL,Left_Stereo_Map[0],Left_Stereo_Map[1], interpolation = cv2.INTER_LANCZOS4, borderMode = cv2.BORDER_CONSTANT)
    Right_good = cv2.remap(frameR,Right_Stereo_Map[0],Right_Stereo_Map[1], interpolation = cv2.INTER_LANCZOS4, borderMode = cv2.BORDER_CONSTANT)
    
    # Convert from RGB to gray scale
    grayR = cv2.cvtColor(Right_good, cv2.COLOR_BGR2GRAY)
    grayL = cv2.cvtColor(Left_good, cv2.COLOR_BGR2GRAY)

    # Compute the 2 images for the Depth_image
    disp = stereo.compute(grayL,grayR)   #.astype(np.float32)/ 16
    # dispL= disp
    # dispR= stereoR.compute(grayR,grayL)
    # dispL= np.int16(dispL)
    # dispR= np.int16(dispR)

    disp = ((disp.astype(np.float32)/ 16)-min_disp)/num_disp # Calculation allowing to have 0 for the most distance object able to detect

    # Prediction for Left Camera
    predict_box = model_detection.predict(Left_good)[0]
    if predict_box.boxes.xyxy.nelement() != 0:
        for idx, box in enumerate(predict_box.boxes):
            if box.conf[0] > 0.50:
                for (xA, yA, xB, yB) in box.xyxy.tolist():
                    xA = round(xA)
                    yA = round(yA)
                    xB = round(xB)
                    yB = round(yB)
                    x_centerL = (xB + xA)//2
                    y_centerL = (yB + yA)//2

                    mask_predictor = SamPredictor(model_segmentation)
                    mask_predictor.set_image(Left_good)
                    input_box = np.array([xA,yA,xB,yB])
                    masks, _, _ = mask_predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=input_box[None,:],
                        multimask_output=False
                    )

                    # Display results
                    mask_overlay = np.zeros_like(Left_good)
                    mask_overlay[:, :, 2] = masks[0] * colors["R"][idx]  
                    mask_overlay[:, :, 1] = masks[0] * colors["G"][idx]
                    mask_overlay[:, :, 0] = masks[0] * colors["B"][idx]
                    Left_good = cv2.addWeighted(Left_good, 0.8, mask_overlay, 0.2, 0)

                    cv2.rectangle(Left_good, (xA,yA), (xB,yB), (0, 255, 0), 2)
                    cv2.circle(Left_good, (x_centerL, y_centerL), radius=5, color=(0,0,255), thickness=-1)
                    distance = calculate_distance(disp, x_centerL, y_centerL)
                    cv2.putText(Left_good, f"Distance: {distance} m", (xA, yA), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),2)
            else:
                x_centerL = None

    # Prediction for Right Camera
    predict_box = model_detection.predict(Right_good)[0]
    if predict_box.boxes.xyxy.nelement() != 0:
        for idx, box in enumerate(predict_box.boxes):
            if box.conf[0] > 0.50:
                for (xA, yA, xB, yB) in box.xyxy.tolist():
                    xA = round(xA)
                    yA = round(yA)
                    xB = round(xB)
                    yB = round(yB)
                    x_centerR = xA + (xB - xA)//2
                    y_centerR = yA + (yB - yA)//2

                    mask_predictor = SamPredictor(model_segmentation)
                    mask_predictor.set_image(Right_good)
                    input_box = np.array([xA,yA,xB,yB])
                    masks, _, _ = mask_predictor.predict(
                        point_coords = None,
                        point_labels = None,
                        box = input_box[None,:],
                        multimask_output = False
                    )

                    # Display results
                    mask_overlay = np.zeros_like(Right_good)
                    mask_overlay[:, :, 2] = masks[0] * colors["R"][idx]  
                    mask_overlay[:, :, 1] = masks[0] * colors["G"][idx]
                    mask_overlay[:, :, 0] = masks[0] * colors["B"][idx]
                    Right_good = cv2.addWeighted(Right_good, 0.8, mask_overlay, 0.2, 0)

                    cv2.rectangle(Right_good, (xA,yA), (xB,yB), (0, 255, 0), 2)
                    cv2.circle(Right_good, (x_centerR, y_centerR), radius=5, color=(0,0,255), thickness=-1)
                    distance = calculate_distance(disp, x_centerR, y_centerR)
                    cv2.putText(Right_good, f"Distance: {distance} m", (xA, yA), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),2)
            else:
                x_centerR = None

    # Frame counting
    frame_count += 1
    if frame_count >= 10:
        stop_time = time.time()
        FPS = frame_count/(stop_time - start_time)
        frame_count = 0
        start_time = time.time()


    cv2.putText(Left_good, f"{FPS:.2f} FPS", (0,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
    cv2.putText(Right_good, f"{FPS:.2f} FPS", (0,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
    cv2.putText(Left_good, "q - quit", (550,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
    cv2.putText(Right_good, "q - quit", (550,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
    cv2.imshow("Left", Left_good)
    cv2.imshow("Right", Right_good)

    # Press q to finish program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

CamR.release()
CamL.release()
cv2.destroyAllWindows()
