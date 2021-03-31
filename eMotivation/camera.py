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
        self.gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("gray frame", self.gray)
        
        faces = self.ff_detector(self.gray)
        for face in faces:
           
            x, y = face.left(), face.top()
            x1, y1 = face.right(), face.bottom()
            cv2.rectangle(frame, (x, y), (x1, y1), (255, 0, 0), 2)
            
            
            self.landmarks = self.kp_detector(self.gray, face)
            
            self.get_face_feature(face_feature = 'right eye')
            self.get_face_feature(face_feature = 'left eye')
            self.get_face_feature(face_feature = 'mouth')
            
            self.extract_eyes()
                          
    
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
        
    def extract_eyes(self):
        
        #extract right eye region
        self.right_eye_region = np.array([(self.landmarks.part(36).x,
                                      self.landmarks.part(36).y),
                                     (self.landmarks.part(37).x,
                                      self.landmarks.part(37).y),
                                     (self.landmarks.part(38).x, 
                                      self.landmarks.part(38).y),
                                     (self.landmarks.part(39).x,
                                      self.landmarks.part(39).y),
                                     (self.landmarks.part(40).x,
                                      self.landmarks.part(40).y),
                                     (self.landmarks.part(41).x,
                                      self.landmarks.part(41).y)],
                                      np.int32)
                                     
        
        frame_height, frame_width, _ = self.frame.shape                             
        self.right_mask = np.zeros((frame_height, frame_width), np.uint8)

        cv2.polylines(self.right_mask, [self.right_eye_region], True, 255, 2)
        cv2.fillPoly(self.right_mask, [self.right_eye_region], 255)
        
        self.bitw_right_eye = cv2.bitwise_and(self.gray,
                                         self.gray, 
                                         mask=self.right_mask
                                         )
        cv2.imshow("right eye", self.right_mask)
        cv2.imshow("btws right eye", self.bitw_right_eye)
        
        
        # extract region
        right_min_x = np.min(self.right_eye_region[:, 0])
        right_max_x = np.max(self.right_eye_region[:, 0])
        right_min_y = np.min(self.right_eye_region[:, 1])
        right_max_y = np.max(self.right_eye_region[:, 1])

        self.gray_right_eye = self.bitw_right_eye[
            right_min_y: right_max_y, 
            right_min_x: right_max_x
            ]


        _, self.threshold_right_eye = cv2.threshold(
                self.gray_right_eye, 
                70, 
                255,
                cv2.THRESH_BINARY
                )

        self.threshold_right_eye = cv2.resize(self.threshold_right_eye, None, fx=5, fy=5)
        cv2.imshow("thrs right eye", self.threshold_right_eye)
        
        self.right_sclera_ratio = self.get_sclera_ratio(self.threshold_right_eye)
        self.get_gaze(self.right_sclera_ratio)
        
        
    def get_sclera_ratio(self, threshold_eye):
        height, width = threshold_eye.shape    
        left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
        left_side_white = cv2.countNonZero(left_side_threshold)
        right_side_threshold = threshold_eye[0: height, int(width / 2): width]
        right_side_white = cv2.countNonZero(right_side_threshold)
        try:
            sclera_ratio = left_side_white / right_side_white
        except:
            sclera_ratio = 1.
        return sclera_ratio 
    
    
    def get_gaze(self, sclera_ratio):
        # Gaze detection

        if sclera_ratio <= 1:
            cv2.putText(self.frame, 
                        "Right", (50, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX,
                        7, (255, 255, 255),10)

    
        elif 1 < sclera_ratio < 2.5:
            cv2.putText(self.frame, 
                        "Center", (50, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX,
                        7, (255, 255, 255),10)

        else:
            cv2.putText(self.frame, 
                        "Left", (50, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX,
                        7, (255, 255, 255), 10)
    
        cv2.imshow('image', self.frame)

        
                
    
    
if __name__ == '__main__':
    
    camera = Camera()
    camera.get_camera(max_time=50)
    
    
            
            
            
            
            
            
            
            
            
            
            
            
            
            