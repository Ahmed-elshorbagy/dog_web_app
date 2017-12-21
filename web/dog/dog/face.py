import cv2
import numpy as np
import matplotlib as plt
import os
FACE_DETECTOR_PATH = r"{base_path}/haarcascades/haarcascade_frontalface_alt.xml".format(
	base_path=os.path.abspath(os.path.dirname(__file__)))
EYE_DETECTOR_PATH = r"{base_path}/haarcascades/haarcascade_eye.xml".format(
	base_path=os.path.abspath(os.path.dirname(__file__)))
def dog_ear(img,ad="dog ear"):
    if ad=="dog ear":
        imag2 = cv2.imread('ear2.png')
    else:
        imag2 = cv2.imread('crown2.png')
    face_cascade = cv2.CascadeClassifier(FACE_DETECTOR_PATH)
    eye_cascade = cv2.CascadeClassifier(EYE_DETECTOR_PATH)
    
    row,col,chan=img.shape
    blank_image = np.zeros((row,col,3), np.uint8)
    # I want to put logo on top-left corner, So I create a ROI


    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        # cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        

        
        r = w / imag2.shape[1]
        dim = (w, int(imag2.shape[0] * r))
        
        img2 = cv2.resize(imag2, dim, interpolation = cv2.INTER_AREA)
        y1=int(img2.shape[0]*.45)
        y2=img2.shape[0]-y1
        # print(y1)
        # print(y2)
        # print(y)
        # print(y+img2.shape[0])
        # print(y-y1)
        # print(y+y2)
        try:
            blank_image[y-y1:y+y2, x:x+img2.shape[1]]=img2
        except:
            blank_image[0:y+y2, x:x+img2.shape[1]]=img2[-1*(y-y1):,:]
            
    img2=blank_image

        # eyes = eye_cascade.detectMultiScale(roi_gray)
        # for (ex,ey,ew,eh) in eyes:
        #     cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        #     cv2.circle(roi_color,(ex+(ew//2),ey+(eh//2)), eh//4, (0,255,255), 1)


    rows,cols,channels = img2.shape
    roi = img[0:rows, 0:cols ]
    # Now create a mask of logo and create its inverse mask also
    img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)

    # Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(img2,img2,mask = mask)

    # Put logo in ROI and modify the main image
    dst = cv2.add(img1_bg,img2)

    img[0:rows, 0:cols ] = dst
    return img

# cv2.imshow('img',dog_ear(img))
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# detector = cv2.CascadeClassifier(FACE_DETECTOR_PATH)