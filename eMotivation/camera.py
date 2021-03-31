# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 16:58:12 2021

@author: adria
"""


import cv2
from psychopy.core import MonotonicClock

import numpy as np
import dlib
import os

# calcola il punto medio tra due punti, viene chiamata sotto
def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)



vid_cod = cv2.VideoWriter_fourcc(*'MP4V') # definisco formato video




class Camera:
    
    def __init__(self):
        
        self.Capture = cv2.VideoCapture(0)
        self.recording = False
        
        self.vid_cod = cv2.VideoWriter_fourcc(*'MP4V')
        self.Capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
        self.Capture.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
        
        
        save_path = os.path.join(os.getcwd(), "prova_video.mp4")
        self.output = cv2.VideoWriter(
            save_path, 
            vid_cod, 
            10, 
            (640,480)
            )

        
        
        self.ff_detector = dlib.get_frontal_face_detector()# works only with gray images
        self.kp_detector = dlib.shape_predictor(
                "shape_predictor_68_face_landmarks.dat"
                )# prende regioni dove prevede che ci sia qualcosa
        
        
        
    def get_camera(self, max_time = 10.0):
        
        # fps: frame per second
        # res: [height,width]
        self.timestamps = []
        self.recording = True
        self.clock = MonotonicClock()
        while self.recording:
            
            _, self.frame = self.Capture.read()# accumula frame, salva il video(non è statico)
            
            self.output.write(self.frame)# writes the specified image to video file
            
            self.timestamps.append(self.clock.getTime())# a cosa corrispondono i numeri?
            
            self.get_ff(self.frame)
            
            if self.clock.getTime() > max_time:
                self.recording = False
            
            cv2.imshow("camera frame", self.frame)
            key = cv2.waitKey(1)
            if key == 27:
                break              
            
        self.Capture.release() #chiude il capture
        self.output.release()
        cv2.destroyAllWindows()
        print('freq : {}'.format(len(self.timestamps)/max_time))
        
        
    def get_ff(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("gray frame", gray)
        
        faces = self.ff_detector(gray)
        for face in faces:
           
            x, y = face.left(), face.top()
            x1, y1 = face.right(), face.bottom()
            cv2.rectangle(frame, (x, y), (x1, y1), (255, 0, 0), 2)
            
            
            self.landmarks = self.kp_detector(gray, face)
            
            self.get_face_feature(face_feature = 'right eye')
            self.get_face_feature(face_feature = 'left eye')
            self.get_face_feature(face_feature = 'mouth')
                          
    
    def get_face_feature(self,
                      face_feature = 'part'):
        
        if face_feature == 'right eye':
            left_and_right_point  = [36,36,39,39]
            center_top_and_bottom = [37,38,41,40]
        elif face_feature == 'left eye':
            left_and_right_point  = [42,42,45,45]
            center_top_and_bottom = [43,44,47,46]
        elif face_feature == 'mouth':
            left_and_right_point  = [48,48,54,54]
            center_top_and_bottom = [51,51,57,57]
            
        
        left_point    = (self.landmarks.part(left_and_right_point[0]).x,
                         self.landmarks.part(left_and_right_point[1]).y)
                      
        right_point   = (self.landmarks.part(left_and_right_point[2]).x,
                         self.landmarks.part(left_and_right_point[3]).y)
        
        center_top    = midpoint(self.landmarks.part(center_top_and_bottom[0]),
                                 self.landmarks.part(center_top_and_bottom[1]))
                              
        center_bottom = midpoint(self.landmarks.part(center_top_and_bottom[2]),
                                 self.landmarks.part(center_top_and_bottom[3]))
    
        hor_line = cv2.line(self.frame,
                            left_point, 
                            right_point,
                            (0, 255, 0),
                            2)
        
        ver_line = cv2.line(self.frame,
                            center_top, 
                            center_bottom,
                            (0, 255, 0),
                            2)
    
    
if __name__ == '__main__':
    
    camera = Camera()
    camera.get_camera()
    
    
            
            
            
            
            
            
            
            
            
            
            
            
            
            