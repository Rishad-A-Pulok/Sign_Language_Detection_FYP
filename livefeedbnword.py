import cv2
import tensorflow as tf
import numpy as np
import keyboard
from time import sleep
from skimage import transform

CATEGORIES = ['Color', 'Friend', 'Myself', 'Promise', 'Request', 'Salam', 'Surprise', 'They', 'Think', 'You']
def convert(np_image):
   np_image = np.array(np_image).astype('float32')
   np_image = transform.resize(np_image, (64, 64, 3))
   np_image = np.expand_dims(np_image, axis=0)
   return np_image
model = tf.keras.models.load_model("bdsl_word_eng_resnet.h5")

cap = cv2.VideoCapture(0)#use 0 if using inbuilt webcam
font = cv2.FONT_HERSHEY_SIMPLEX 
org = (70, 40) 
fontScale = 1.0
color = (255, 255, 0) 
thickness = 3
# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    frame1 = cv2.resize(frame, (200, 200))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    prediction = model.predict([convert(gray)])
    final = (CATEGORIES[int(np.argmax(prediction[0]))])
    sleep(0.1)
    try:
        cv2.putText(frame, "App prediction is :-", (50,30), font,  
                            0.5, (255,0,0), 1, cv2.LINE_AA)
        cv2.putText(frame, final, (85,60), font,  
                               fontScale, (0,0,255), thickness, cv2.LINE_AA) 
        cv2.imshow("frame",frame)
    except Exception as e:
        cv2.imshow("frame",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
  

cap.release()
cv2.destroyAllWindows()