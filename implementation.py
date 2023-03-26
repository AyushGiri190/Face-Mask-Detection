import cv2
import numpy as np
from tensorflow import keras
def return_faces(image_path):
    '''This function return the total faces found in a image using few function of opencv like CascasdeClassifier
    and detectmultiscale. 
    Cascade Classifier are pretrained algorithm for object detection in an image '''

    face_detection_classifier = cv2.CascadeClassifier('./haarcascade_frontalface_alt2.xml') # cascasde classifier to detect faces in a image 
    image_path = str(image_path)  # type casting the image path to reduce the error 
    img = cv2.imread(image_path)  # using cv2 function imread to read a image from its path 
    # detecting faces in a image using detectMultiScale function of the cascasde classifier
    faces = face_detection_classifier.detectMultiScale(img, scaleFactor=1.08, minNeighbors=4)
    return faces


def result_colour(a):
    '''This function gives us the color of the result that we got through our deep learning model 
    0 is for mask and 1 is for No mask  '''

    label = {0: "Mask", 1: "No Mask"}  # a dicitonary to store our binary result values
    label_colour = {0: (0, 255, 0), 1: (255, 0, 0)}  # giving colors to the result 
    return label[a],label_colour[a]   # returning it to the calling function 

def load_model():
    '''In this function we are just loading the saved model that we trained '''
    model = keras.models.load_model('./miniprojectmask.h5') 
    return model


def classification(image_path):
    '''This is the main part of our program where all the faces are predicted to have a mask or not having a mask
    '''
    pad_y=3  # padding for the text
    model = load_model()
    # For detected faces in the image
    faces = return_faces(image_path)  # getting all the faces 
    img = cv2.imread(image_path)  # reading the image
    for i in range(len(faces)):
        (x, y, w, h) = faces[i]  # getting each face
        cropped_face = img[y : y + h, x : x + w]  # cropping the face out of image
        cropped_face = cv2.resize(cropped_face, (128, 128))   #  resizing it to the model fed image size
        cropped_face = np.reshape(cropped_face, [1, 128, 128, 3]) / 255.0  # converting it into float
        mask_result = model.predict(cropped_face)  # make model prediction
        print_label,label_colour= result_colour(mask_result.argmax()) # get mask/no mask based on prediction
         # green for mask, red for no mask

        # Print result
        (t_w, t_h), _ = cv2.getTextSize(print_label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
         # getting the text size
        
        cv2.rectangle(
            img,
            (x, y + pad_y),
            (x + t_w, y - t_h - pad_y - 6),
            label_colour,
            -1,
        )  # draw rectangle
        cv2.putText(
            img,
            print_label,
            (x, y - 6),
            cv2.FONT_HERSHEY_DUPLEX,
            0.4,
            (255, 255, 255), # white
            1,
        )  # print text

        cv2.rectangle(
            img,
            (x, y),
            (x + w, y + h),
            label_colour,
            1,
        )  # draw bounding box on face
    return img


img = './testimage.webp' # Sample input image 
img = classification(img)
while True: 
        key = cv2.waitKey(1) 
        img = cv2.resize(img,(1300,600))
        cv2.imshow('image classifier',img)
        if key == 27 or key == 13:
            break

