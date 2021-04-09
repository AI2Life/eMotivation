import cv2
from psychopy.core import MonotonicClock

import numpy as np
import dlib
import os
import math
import time
import threading

# calcola il punto medio tra due punti, viene chiamata sotto
def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)



vid_cod = cv2.VideoWriter_fourcc(*'mp4v') # definisco formato video



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
        
        self.left_eye_points = np.arange(36,42)
        self.right_eye_points = np.arange(36,48)
        self.mouth_points = np.arange(48,68)
        self.nose_points = np.arange(27,36)
        
        self.lips_dist = 1
        self.lips_dist_new = 1
        self.possible_talk_counter = 0
        self.talk_counter = 0
        
        
        
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
            
            cv2.imshow("color camera frame", self.frame)
            key = cv2.waitKey(1)
            if key == 27:
                break              
            
        self.Capture.release() #chiude il capture
        self.output.release()
        cv2.destroyAllWindows()
        print('talking detected : {}'.format(self.talk_counter))
        print('freq : {}'.format(len(self.timestamps)/max_time))
        
        
    def get_ff(self, frame):
        self.gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("gray frame", self.gray)
        
        self.thread(5.0, self.game_over())
        
        
        self.faces = self.ff_detector(self.gray)
        for face in self.faces:
           
            x, y = face.left(), face.top()
            x1, y1 = face.right(), face.bottom()
            cv2.rectangle(frame, (x, y), (x1, y1), (255, 0, 0), 2)
            
            
            self.landmarks = self.kp_detector(self.gray, face)
            
            self.get_face_feature(face_feature = 'right eye')
            self.get_face_feature(face_feature = 'left eye')
            self.get_face_feature(face_feature = 'mouth')
            
            
            self.get_nose_distance()
            self.get_lips()
            
            if round(time.time(),1) % 2 == 0:
                self.get_lips_dist()
            else:
                self.get_lips_dist_new()
                
            self.get_change(self.lips_dist_new, self.lips_dist)
            self.detect_speaking()
            self.extract_eyes()
            self.detect_blink()
            #print(self.lips_dist, self.mouth_dist)
            #print(self.left_eye_dist, self.right_eye_dist)
            
            
            
    def calculate_distance(self, x1, x2, y1, y2):
        dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        return dist    
        
           
                          
    
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
        #elif face_feature == 'lips':
            
            
        
        left_point    = (self.landmarks.part(left_and_right_point[0]).x,
                         self.landmarks.part(left_and_right_point[1]).y)
                      
        right_point   = (self.landmarks.part(left_and_right_point[2]).x,
                         self.landmarks.part(left_and_right_point[3]).y)
        
        center_top    = midpoint(self.landmarks.part(center_top_and_bottom[0]),
                                 self.landmarks.part(center_top_and_bottom[1]))
                              
        center_bottom = midpoint(self.landmarks.part(center_top_and_bottom[2]),
                                 self.landmarks.part(center_top_and_bottom[3]))
    
        self.hor_line = cv2.line(self.frame,
                            left_point, 
                            right_point,
                            (0, 255, 0),
                            2)
        
        self.ver_line = cv2.line(self.frame,
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
        
        # self.left_eye_region = np.array([(self.landmarks.part(42).x,
        #                                   self.landmarks.part(42).y)],
        #                                   np.int32)
        # for i in self.left_eye_points:
        #     np.append(self.left_eye_region,
        #               [[self.landmarks.part(i+1).x, self.landmarks.part(i+1).y]], axis=0)
            
        self.left_eye_region= np.array([(self.landmarks.part(42).x,
                                      self.landmarks.part(42).y),
                                      (self.landmarks.part(43).x,
                                      self.landmarks.part(43).y),
                                      (self.landmarks.part(44).x, 
                                      self.landmarks.part(44).y),
                                      (self.landmarks.part(45).x,
                                      self.landmarks.part(45).y),
                                      (self.landmarks.part(46).x,
                                      self.landmarks.part(46).y),
                                      (self.landmarks.part(47).x,
                                      self.landmarks.part(47).y)],
                                      np.int32)
        
        
        frame_height, frame_width, _ = self.frame.shape           
        #costruzione della maschera per ogni occhio                  
        self.right_mask = np.zeros((frame_height, frame_width), np.uint8)
        
        self.left_mask = np.zeros((frame_height, frame_width), np.uint8)
        

        cv2.polylines(self.right_mask, [self.right_eye_region], True, 255, 2)
        cv2.fillPoly(self.right_mask, [self.right_eye_region], 255)
        
        cv2.polylines(self.left_mask, [self.left_eye_region], True, 255, 2)
        cv2.fillPoly(self.left_mask, [self.left_eye_region], 255)
        
        self.bitw_right_eye = cv2.bitwise_and(self.gray,
                                         self.gray, 
                                         mask=self.right_mask
                                         )
        self.bitw_left_eye = cv2.bitwise_and(self.gray,
                                         self.gray, 
                                         mask=self.left_mask
                                         )
        
        cv2.imshow("right eye", self.right_mask)
        cv2.imshow("btws right eye", self.bitw_right_eye)
        
        cv2.imshow("left eye", self.left_mask)
        cv2.imshow("btws left eye", self.bitw_left_eye)
        # extract region
        right_min_x = np.min(self.right_eye_region[:, 0])
        right_max_x = np.max(self.right_eye_region[:, 0])
        right_min_y = np.min(self.right_eye_region[:, 1])
        right_max_y = np.max(self.right_eye_region[:, 1])
        
        left_min_x = np.min(self.left_eye_region[:, 0])
        left_max_x = np.max(self.left_eye_region[:, 0])
        left_min_y = np.min(self.left_eye_region[:, 1])
        left_max_y = np.max(self.left_eye_region[:, 1])
        
        
        self.bitw_right_eye = self.bitw_right_eye[
            right_min_y: right_max_y, 
            right_min_x: right_max_x
            ]

        # todo : calibrare i threshold


        self.threshold_right_eye = cv2.adaptiveThreshold(
                self.bitw_right_eye,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 4)
        
        self.bitw_left_eye = self.bitw_left_eye[
            left_min_y: left_max_y, 
            left_min_x: left_max_x
            ]
        self.threshold_left_eye = cv2.adaptiveThreshold(
                self.bitw_left_eye,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 4)
        

        #self.threshold_right_eye = cv2.resize(self.threshold_right_eye, None, fx=5, fy=5)
        #self.threshold_left_eye = cv2.resize(self.threshold_left_eye, None, fx=5, fy=5)

        self.threshold_right_eye = cv2.resize(self.threshold_right_eye, (600,300), fx=5, fy=5)
        self.threshold_left_eye = cv2.resize(self.threshold_left_eye, (600, 300), fx=5, fy=5)
        
        cv2.imshow("thrs right eye", self.threshold_right_eye)
        cv2.imshow("thrs left eye", self.threshold_left_eye)
        
        self.right_sclera_ratio = self.get_sclera_ratio(self.threshold_right_eye)
        self.left_sclera_ratio = self.get_sclera_ratio(self.threshold_left_eye)
        self.get_gaze(self.right_sclera_ratio, self.left_sclera_ratio)
        
        
        
        
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
    
    
    
    def get_gaze(self, right_sclera_ratio, left_sclera_ratio):
        # Gaze detection
        if (right_sclera_ratio + left_sclera_ratio)/2 >= 1.50:
            cv2.putText(self.frame, 
                        "Left", (50, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX,
                        5, (255, 255, 255),10)
        
        elif 0.50 < (right_sclera_ratio + left_sclera_ratio)/2< 1.50:
            cv2.putText(self.frame, 
                        "Center", (50, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX,
                        5, (255, 255, 255),10)
        
        else:
            cv2.putText(self.frame, 
                        "Right", (50, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX,
                        5, (255, 255, 255), 10)
        #cv2.imshow('image', self.frame) 
        
        
        
    def get_nose_distance(self):
        
        self.up_nose = (self.landmarks.part(self.nose_points[3]).x,
                            self.landmarks.part(self.nose_points[3]).y)
        
        self.bottom_nose = (self.landmarks.part(self.nose_points[2]).x,
                           self.landmarks.part(self.nose_points[2]).y)
        
        up_lip_line = cv2.line(self.frame,
                               self.up_nose, 
                               self.bottom_nose,
                               (0, 0, 255),
                               2)
        
        self.list_up_nose = list(self.up_nose)
        self.list_bottom_nose = list(self.bottom_nose)
        
        self.nose_dist = self.calculate_distance(self.list_up_nose[0], 
                                                      self.list_bottom_nose[0],
                                                      self.list_up_nose[1], 
                                                      self.list_bottom_nose[1])
        
        
        
    def get_lips(self):
        
        self.left_up_lip = (self.landmarks.part(self.mouth_points[13]).x,
                                self.landmarks.part(self.mouth_points[13]).y)
        
        self.mid_up_lip = (self.landmarks.part(self.mouth_points[14]).x,
                           self.landmarks.part(self.mouth_points[14]).y)
                      
        self.right_up_lip = (self.landmarks.part(self.mouth_points[15]).x,
                             self.landmarks.part(self.mouth_points[15]).y)
        
        
        self.left_bottom_lip = (self.landmarks.part(self.mouth_points[19]).x,
                                self.landmarks.part(self.mouth_points[19]).y)
        
        self.mid_bottom_lip = (self.landmarks.part(self.mouth_points[18]).x,
                               self.landmarks.part(self.mouth_points[18]).y)
                      
        self.right_bottom_lip = (self.landmarks.part(self.mouth_points[17]).x,
                                 self.landmarks.part(self.mouth_points[17]).y)
        
        up_lip_line = cv2.line(self.frame,
                            self.left_up_lip, 
                            self.right_up_lip,
                            (0, 0, 255),
                            2)
        
        bottom_lip_line = cv2.line(self.frame,
                            self.left_bottom_lip, 
                            self.right_bottom_lip,
                            (0, 0, 255),
                            2)
        
        self.list_left_up_lip = list(self.left_up_lip)
        self.list_mid_up_lip = list(self.mid_up_lip)
        self.list_right_up_lip = list(self.right_up_lip)
        
        self.list_left_bottom_lip = list(self.left_bottom_lip)
        self.list_mid_bottom_lip = list(self.mid_bottom_lip)
        self.list_right_bottom_lip = list(self.right_bottom_lip)
        
        

        self.left_dist_lips = self.calculate_distance(self.list_left_up_lip[0], 
                                                      self.list_left_bottom_lip[0],
                                                      self.list_left_up_lip[1], 
                                                      self.list_left_bottom_lip[1])
        
        self.mid_dist_lips = self.calculate_distance(self.list_mid_up_lip[0], 
                                                     self.list_mid_bottom_lip[0],
                                                     self.list_mid_up_lip[1], 
                                                     self.list_mid_bottom_lip[1])
        
        self.right_dist_lips = self.calculate_distance(self.list_right_up_lip[0],
                                                       self.list_right_bottom_lip[0],
                                                       self.list_right_up_lip[1],
                                                       self.list_right_bottom_lip[1])            
        
    def  get_lips_dist(self):
        self.lips_dist = (self.left_dist_lips + 
                     self.mid_dist_lips + 
                     self.right_dist_lips) / 3
        
        return self.lips_dist
    
    def  get_lips_dist_new(self):
        self.lips_dist_new = (self.left_dist_lips + 
                     self.mid_dist_lips + 
                     self.right_dist_lips) / 3
        
        return self.lips_dist_new
    

        
    def get_change(self, current, previous):

        self.perc = (abs((current+1) - (previous+1)) / (previous+1)) * 100.0
        
        return self.perc

        
    
    def detect_speaking(self):
        
        perc_threshold = 78.0
        if self.perc > perc_threshold: 
                cv2.putText(self.frame, 
                            "Possible Talking detected!", (100, 350), 
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255),3)
                self.possible_talk_counter += 1
                print('Possible Talking detected!', self.perc)
        else:
            print(self.perc)
        #     cv2.putText(self.frame, 
        #                 "Silence", (100, 450), 
        #                 cv2.FONT_HERSHEY_SIMPLEX,
        #                 1, (0, 255, 0),2)
        # cv2.imshow('image', self.frame) 
        
    def thread(self, countdown, function):
        
        threading.Timer(countdown, function).start
        #time.sleep(1)
    
    def game_over(self):
         
         talk_threshold = 20
         if self.possible_talk_counter > talk_threshold:
             cv2.putText(self.frame, 
                            "Talking!", (150, 450), 
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255),3)
             
             self.talk_counter += 1
             print()
             print('Talking detected!')
             print()
             self.possible_talk_counter = 0
        
        
    def get_eyes(self):
        
        self.left_up_left_eye = (self.landmarks.part(self.left_eye_points[1]).x,
                                  self.landmarks.part(self.left_eye_points[1]).y)
                      
        self.right_up_left_eye = (self.landmarks.part(self.left_eye_points[2]).x,
                                  self.landmarks.part(self.left_eye_points[2]).y)
        
        self.left_bottom_left_eye = midpoint(self.landmarks.part(self.left_eye_points[5]),
                                      self.landmarks.part(self.left_eye_points[5]))
                              
        self.right_bottom_left_eye = midpoint(self.landmarks.part(self.left_eye_points[4]),
                                      self.landmarks.part(self.left_eye_points[4]))
        
        
        self.left_up_right_eye = (self.landmarks.part(self.right_eye_points[1]).x,
                                  self.landmarks.part(self.right_eye_points[1]).y)
                      
        self.right_up_right_eye = (self.landmarks.part(self.right_eye_points[2]).x,
                                  self.landmarks.part(self.right_eye_points[2]).y)
        
        self.left_bottom_right_eye = midpoint(self.landmarks.part(self.right_eye_points[5]),
                                              self.landmarks.part(self.right_eye_points[5]))
                              
        self.right_bottom_right_eye = midpoint(self.landmarks.part(self.right_eye_points[4]),
                                              self.landmarks.part(self.right_eye_points[4]))
        
        
    def detect_blink(self):
        
        self.get_eyes()
        
        self.list_left_up_left_eye = list(self.left_up_left_eye)
        self.list_right_up_left_eye = list(self.right_up_left_eye)
        self.list_left_bottom_left_eye = list(self.left_bottom_left_eye)
        self.list_right_bottom_left_eye = list(self.right_bottom_left_eye)
        
        self.list_left_up_right_eye = list(self.left_up_right_eye)
        self.list_right_up_right_eye = list(self.right_up_right_eye)
        self.list_left_bottom_right_eye = list(self.left_bottom_right_eye)
        self.list_right_bottom_right_eye = list(self.right_bottom_right_eye)
        
        self.left_dist_left_eye = self.calculate_distance(self.list_left_up_left_eye[0],
                                                      self.list_left_bottom_left_eye[0],
                                                      self.list_left_up_left_eye[1], 
                                                      self.list_left_bottom_left_eye[1])
        
        self.right_dist_left_eye = self.calculate_distance(self.list_right_up_left_eye[0],
                                                            self.list_right_bottom_left_eye[0],
                                                            self.list_right_up_left_eye[1], 
                                                            self.list_right_bottom_left_eye[1])
        
        self.left_dist_right_eye = self.calculate_distance(self.list_left_up_right_eye[0],
                                                      self.list_left_bottom_right_eye[0],
                                                      self.list_left_up_right_eye[1], 
                                                      self.list_left_bottom_right_eye[1])
        
        self.right_dist_right_eye = self.calculate_distance(self.list_right_up_right_eye[0],
                                                            self.list_right_bottom_right_eye[0],
                                                            self.list_right_up_right_eye[1], 
                                                            self.list_right_bottom_right_eye[1])
        
        self.left_eye_dist = (self.left_dist_left_eye + 
                      self.right_dist_left_eye) / 2
        
        self.right_eye_dist = (self.left_dist_right_eye + 
                      self.right_dist_right_eye) / 2
        
        if self.left_eye_dist*2.25 < self.nose_dist: #or self.right_eye_dist*1.3 < self.nose_dist:
            cv2.putText(self.frame, 
                        "Blink ;)", (350, 250), 
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (200, 255, 0),3)
        
        # if self.right_eye_dist*1.43 < self.nose_dist:
        #     cv2.putText(self.frame, 
        #                 "Left Blink ;)", (400, 150), 
        #                 cv2.FONT_HERSHEY_SIMPLEX,
        #                 1, (0, 255, 200),3)


        
    
    
if __name__ == '__main__':
    camera = Camera()
    camera.get_camera(max_time=60)
    
    
            
            
            
            
            
            
            
            
            
            
            
            
            
            