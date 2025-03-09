import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import time

#reading the image file and also the loaded image is in the BGR form by default. SO, converting it to RGB form.
def ConvertToRGB(images):
    image = cv.cvtColor(images,cv.COLOR_BGR2RGB)
    cv.imshow("myimage",image)
    

def FaceDetection():

    #loading the the pre-trained Haar Cascade classifier for face detection
    haar_cascade_data = cv.CascadeClassifier("data/data/haarcascade_frontalface_alt.xml")
    img = cv.imread("images/image1.jpg")
    if img is None:
            print("Error!! No image generated")
            exit()

    print(img.shape)  #checking the size of an image

    #resizing the image due to oversize
    img=cv.resize(img,(600,570))
    #COnvert the iamge into gray color
    gray_img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

  

     #Showing the loaded image
    cv.imshow("Myimage",gray_img)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return gray_img


if __name__ == "__main__":
    gray_img = FaceDetection()
    ConvertToRGB(gray_img)
