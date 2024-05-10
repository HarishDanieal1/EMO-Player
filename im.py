import time
import cv2
from pygame import mixer
import pygame
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import random
mixer.init()

emotion =  ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
now = time.time()
model = keras.models.load_model("model_35_91_61.h5")
font = cv2.FONT_HERSHEY_SIMPLEX
cam = cv2.VideoCapture(0)
face_cas = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')

def __draw_label(img, text, pos, bg_color):
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.0
    color = (0, 0, 0)
    thickness = cv2.FILLED
    margin = 2

    txt_size = cv2.getTextSize(text, font_face, scale, thickness)

    end_x = pos[0] + txt_size[0][0] + margin
    end_y = pos[1] - txt_size[0][1] - margin

    cv2.rectangle(img, pos, (end_x, end_y), bg_color, thickness)
    cv2.putText(img, text, pos, font_face, scale, color, 1, cv2.LINE_AA)


while True:
    ret, frame = cam.read()
    x=""
    if ret==True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #gray = cv2.flip(gray,1)
        faces = face_cas.detectMultiScale(gray, 1.3,5)
        
        for (x, y, w, h) in faces:
            face_component = gray[y:y+h, x:x+w]
            fc = cv2.resize(face_component, (48, 48))
            inp = np.reshape(fc,(1,48,48,1)).astype(np.float32)
            inp = inp/255.
            prediction = model.predict_proba(inp)
            em = emotion[np.argmax(prediction)]
            score = np.max(prediction)
            cv2.putText(frame, em+"  "+str(score*100)+'%', (x, y), font, 1, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            
            if(em=='Anger'):
                if(em=='Anger'):
                    path1 ="static/Songs/Angry Song"
                    __draw_label(frame, 'you are Angry camdown', (20,20), (255,0,0))
                    all_mp1 = [os.path.join(path1, f) for f in os.listdir(path1) if f.endswith('.mp3')]
                    randomfile1 = random.choice(all_mp1)
                    if(pygame.mixer.music.get_busy()==False):
                        cam.release()
                        cv2.destroyAllWindows()
                        print('you are '+em+'cam down')
                        pygame.mixer.music.load(randomfile1)
                        pygame.mixer.music.play()
                        break

            if(em=='Disgust'):
                if(em=='Disgust'):
                    path2 ="static/Songs/Disgust song"
                    __draw_label(frame, 'you are Disgust Dont show that face', (20,20), (255,0,0))
                    all_mp2 = [os.path.join(path2, f) for f in os.listdir(path2) if f.endswith('.mp3')]
                    randomfile2 = random.choice(all_mp2)
                    if(pygame.mixer.music.get_busy()==False):
                        cam.release()
                        print('you are '+em)
                        cv2.destroyAllWindows()
                        pygame.mixer.music.load(randomfile2)
                        pygame.mixer.music.play()
                        break

            if(em=='Fear'):
                if(em=='Fear'):
                    path3 ="static/Songs/fear song"
                    __draw_label(frame, 'you are having fear nothing to vary', (20,20), (255,0,0))
                    all_mp3 = [os.path.join(path3, f) for f in os.listdir(path3) if f.endswith('.mp3')]
                    randomfile3 = random.choice(all_mp3)
                    if(pygame.mixer.music.get_busy()==False):
                        cam.release()
                        print('your are' +em+ ' have some fun')
                        cv2.destroyAllWindows()
                        pygame.mixer.music.load(randomfile3)
                        pygame.mixer.music.play()
                        break

            if(em=='Happy'):
                if(em=='Happy'):
                    path4 ="static/Songs/Happy song"
                    __draw_label(frame, 'I will Make you More Happy', (20,20), (255,0,0))
                    all_mp4 = [os.path.join(path4, f) for f in os.listdir(path4) if f.endswith('.mp3')]
                    randomfile4 = random.choice(all_mp4)
                    if(pygame.mixer.music.get_busy()==False):
                        cam.release()
                        cv2.destroyAllWindows()
                        print('you are '+em+ ' have some more')
                        pygame.mixer.music.load(randomfile4)
                        pygame.mixer.music.play()
                        break

            if(em=='Neutral'):
                if(em=='Neutral'):
                    path5 ="static/Songs/Neutral song"
                    __draw_label(frame, 'you are being Netural', (20,20), (255,0,0))
                    all_mp5 = [os.path.join(path5, f) for f in os.listdir(path5) if f.endswith('.mp3')]
                    randomfile5 = random.choice(all_mp5)
                    if(pygame.mixer.music.get_busy()==False):
                        cam.release()
                        cv2.destroyAllWindows()
                        print('you are being '+em)
                        pygame.mixer.music.load(randomfile5)
                        pygame.mixer.music.play()
                        break

            if(em=='Sad'):
                if(em=='Sad'):
                    path6 ="static/Songs/Sad Song"
                    __draw_label(frame, 'you are Sad Enjoy with the song', (20,20), (255,0,0))
                    all_mp6 = [os.path.join(path6, f) for f in os.listdir(path6) if f.endswith('.mp3')]
                    randomfile6 = random.choice(all_mp6)
                    if(pygame.mixer.music.get_busy()==False):
                        cam.release()
                        cv2.destroyAllWindows()
                        print('you are '+em+ ' have some fun')
                        pygame.mixer.music.load(randomfile6)
                        pygame.mixer.music.play()
                        break
            if(em=='Surprise'):
                if(em=='Surprise'):
                    path7 ="static/Songs/Suprise song"
                    __draw_label(frame, 'you are Surprised ahahah', (20,20), (255,0,0))
                    all_mp7 = [os.path.join(path7, f) for f in os.listdir(path7) if f.endswith('.mp3')]
                    randomfile7 = random.choice(all_mp7)
                    if(pygame.mixer.music.get_busy()==False):
                        cam.release()
                        cv2.destroyAllWindows()
                        print('you are '+em+'ed haahaa')
                        pygame.mixer.music.load(randomfile7)
                        pygame.mixer.music.play()
                        break  
        cv2.imshow("image", frame)   
    if cv2.waitKey(1) == 27:
        break
    else:
        future = now + 60
        if time.time() > future:
            break

cam.release()
cv2.destroyAllWindows()