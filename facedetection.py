import cv2 as cv

#function to display the final images
def DisplayImage(images):
    for i, image in enumerate (images):
        cv.imshow(f"image{i+1}",image)
    cv.waitKey(0)
    cv.destroyAllWindows()

# Function to detect faces     
def FaceDetection(photos):

    #loading the the pre-trained Haar Cascade classifier for face detection
    haar_cascade_data1 = cv.CascadeClassifier("data/haarcascade_frontalface_alt.xml")
    haar_cascade_data2 = cv.CascadeClassifier("data/haarcascade_profileface.xml")
    haar_cascade_data3 = cv.CascadeClassifier("data/haarcascade_frontalface_alt2.xml")

    #Check whether the file is loaded sucessfully or not
    if haar_cascade_data1.empty() or haar_cascade_data2.empty() or haar_cascade_data3.empty():
        print(f"Error in loading HaarCascade file. check file paths again")

    images_paths = []

    for path in photos:
        img = cv.imread(path)

        if  img is None:
            print(f"Error!! Couldnot load image {path}")
            continue

        #print(img.shape)  #check the size of an image
        #resize the image due to oversize
        img=cv.resize(img,(600,570))
                           
        gray_img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)   #Convert into gray color
        gray_img = cv.equalizeHist(gray_img)    #improve contrast of gray_image

      
   
        #detect faces in grayscale with different classsifiers
        faces_frontal = haar_cascade_data1.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=3, minSize=(30,30))  
        faces_profile = haar_cascade_data2.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=3, minSize=(30,30))
        faces_alt2 = haar_cascade_data3.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=4, minSize=(30,30) )

        #combine all detected faces
        all_faces=[]
        for i in [faces_frontal, faces_profile, faces_alt2]:
            for j in i:
                all_faces.append(j)


        #check if no faces were detected by verifying if the 'all_faces' array is empty
        if len(all_faces) == 0:
            print(f"no face detected in {path} ")
            continue 
        
         # Merge overlapping detections
        all_faces,_=cv.groupRectangles(all_faces, groupThreshold=1, eps=0.2)
        
        #form rectangle in the detected faces
        for (x,y,w,h) in all_faces:
            cv.rectangle(img,(x,y), (x+w , y+h), (0,255,0),2) 

        images_paths.append(img)

    return images_paths

if __name__ == "__main__":
    photos=["images/image1.jpg","images/image2.JPG","images/image3.JPG","images/image4.JPG","images/image5.JPG"]
    images_paths = FaceDetection(photos)

    if images_paths:
        DisplayImage(images_paths)
    else:
        print("Nothing to display.")
    


