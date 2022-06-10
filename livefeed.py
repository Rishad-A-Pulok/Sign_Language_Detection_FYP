import cv2
import tensorflow as tf
import numpy as np
import keyboard
from time import sleep
# from spellchecker import SpellChecker
from skimage import transform

CATEGORIES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "del", "nothing", "space"]
word=""
# def prepare(img):
#     IMG_SIZE = 64
#     # img = cv2.imread(filepath)
#     image = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation = cv2.INTER_AREA)
#     image = image.reshape(-1,64,64,3)
#     return image
def convert(np_image):
   np_image = np.array(np_image).astype('float32')
   np_image = transform.resize(np_image, (64, 64, 3))
   np_image = np.expand_dims(np_image, axis=0)
   return np_image
model = tf.keras.models.load_model("full_model_asl_CNN.h5")

cap = cv2.VideoCapture(0)#use 0 if using inbuilt webcam
# spell = SpellChecker()
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
    # print(prediction)
    final = (CATEGORIES[int(np.argmax(prediction[0]))])
    # print(final)
    if keyboard.is_pressed('f'):
        word=word+final
        print("Test", word)
        sleep(0.1)
    if keyboard.is_pressed("s"):
        lst=word.split(' ')
#        correct = spell.correction(lst.pop())
#        lst.append(correct)
        word=" ".join(lst)
#         word=TextBlob(word)
#         word=word.correct()
        # print(word)
        word=word+" "
        # print("Final_Word",word)
        sleep(0.1)
    try:
        cv2.putText(frame, "App prediction is :-", (50,30), font,  
                           0.5, (255,0,0), 1, cv2.LINE_AA)
        cv2.putText(frame, final, (85,60), font,  
                               fontScale, (0,0,255), thickness, cv2.LINE_AA)
        cv2.imshow("frame",frame)
    except Exception as e:
#         print(e)
        cv2.imshow("frame",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
  

cap.release()
cv2.destroyAllWindows()