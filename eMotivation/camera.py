import cv2
from psychopy.core import MonotonicClock

import numpy as np
import dlib
import os
import math


# takes 2 points, returns the middle point between them
def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

# takes 2 points, returns the distance between them
def euclidean_distance(point1 , point2):
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
BLINK_RATIO_THRESHOLD = 5

vid_cod = cv2.VideoWriter_fourcc(*'mp4v') 


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

        self.ff_detector = dlib.get_frontal_face_detector() # works only with gray images
        self.kp_detector = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") 
        
        # key points
        self.right_eye_pts  = [36,37,38,39,40,41]
        self.left_eye_pts   = [42,43,44,45,46,47]
        
          
    def get_camera(self, max_time = 10.0):
        
        # fps: frame per second
        # res: [height,width]
        self.timestamps = []
        self.recording = True
        self.clock = MonotonicClock()
        while self.recording:
            
            _, self.frame = self.Capture.read()
            
            self.output.write(self.frame)
            self.timestamps.append(self.clock.getTime())
            self.get_ff(self.frame)
            
            if self.clock.getTime() > max_time:
                self.recording = False
            
            cv2.imshow("camera frame", self.frame)
            key = cv2.waitKey(1)
            if key == 27:
                break              
            
        self.Capture.release() 
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
            
            #detect blinks
            self.landmarks = self.kp_detector(self.gray, face)
            
            self.right_eye_ratio = self.get_blink_ratio(self.right_eye_pts, self.landmarks)
            self.left_eye_ratio  = self.get_blink_ratio(self.left_eye_pts, self.landmarks)
            self.blink_ratio     = (self.left_eye_ratio + self.right_eye_ratio) / 2
        
            if self.blink_ratio > BLINK_RATIO_THRESHOLD:
                cv2.putText(self.frame,
                            "BLINKING",
                            (10,50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            2,(255,255,255),2,
                            cv2.LINE_AA)
        
            # detect eyes and mouth
            self.get_face_feature(face_feature = 'right eye')
            self.get_face_feature(face_feature = 'left eye')
            self.get_face_feature(face_feature = 'mouth')
            
            self.extract_feature(feature="right eye")
            self.extract_feature(feature="left eye")
            
            self.get_gaze(self.right_sclera_ratio, self.left_sclera_ratio)
                          
    
    def get_face_feature(self,
                         face_feature = 'part'):
        
        if face_feature == 'right eye':
            self.eye_points = self.right_eye_pts 
            
        elif face_feature == 'left eye':
            self.eye_points = self.left_eye_pts 
            
        # elif face_feature == 'mouth':
        #     self.mouth_points = self.all_points_mouth
        
            
        self.left_point    = (self.landmarks.part(self.eye_points[0]).x,
                              self.landmarks.part(self.eye_points[0]).y)
        self.right_point   = (self.landmarks.part(self.eye_points[3]).x,
                              self.landmarks.part(self.eye_points[3]).y)
        
        self.center_top    = midpoint(self.landmarks.part(self.eye_points[1]),
                                      self.landmarks.part(self.eye_points[2]))
        self.center_bottom = midpoint(self.landmarks.part(self.eye_points[5]),
                                      self.landmarks.part(self.eye_points[4]))
    
        self.hor_line = cv2.line(self.frame,
                            self.left_point, 
                            self.right_point,
                            (0, 255, 0),
                            2)

        self.ver_line = cv2.line(self.frame,
                            self.center_top, 
                            self.center_bottom,
                            (0, 255, 0),
                            2)
        
    def extract_feature(self, feature):
        
        if feature == "right eye":
            self.points = self.right_eye_pts
        elif feature == "left eye":
            self.points = self.left_eye_pts
        
        self.region = np.array([(self.landmarks.part(self.points[0]).x,
                                          self.landmarks.part(self.points[0]).y),
                                         (self.landmarks.part(self.points[1]).x,
                                          self.landmarks.part(self.points[1]).y),
                                         (self.landmarks.part(self.points[2]).x, 
                                          self.landmarks.part(self.points[2]).y),
                                         (self.landmarks.part(self.points[3]).x,
                                          self.landmarks.part(self.points[3]).y),
                                         (self.landmarks.part(self.points[4]).x,
                                          self.landmarks.part(self.points[4]).y),
                                         (self.landmarks.part(self.points[5]).x,
                                          self.landmarks.part(self.points[5]).y)],
                                          np.int32)
        
        frame_height, frame_width, _ = self.frame.shape     
        
        self.mask = np.zeros((frame_height, frame_width), np.uint8)
        
        cv2.polylines(self.mask, [self.region], True, 255, 2)
        cv2.fillPoly(self.mask, [self.region], 255)
        
        self.bitw_eye = cv2.bitwise_and(self.gray,
                                         self.gray, 
                                         mask=self.mask
                                         )
                                  
        cv2.imshow(feature, self.mask)
        cv2.imshow("btws" + feature, self.bitw_eye)
    
        min_x = np.min(self.region[:, 0])
        max_x = np.max(self.region[:, 0])
        min_y = np.min(self.region[:, 1])
        max_y = np.max(self.region[:, 1])
        
        self.bitw_eye  =   self.bitw_eye [
            min_y: max_y, 
            min_x: max_x
            ]

        #ADAPTIVE
        self.threshold_eye = cv2.adaptiveThreshold(
                self.bitw_eye,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 147, -50)
    
        self.threshold_eye = cv2.resize(self.threshold_eye, (600,300), fx=5, fy=5)

        cv2.imshow("thrs" + feature, self.threshold_eye)
        
        self.sclera_ratio = self.get_sclera_ratio(self.threshold_eye)
        
        if feature == "right eye":
            self.right_sclera_ratio = self.sclera_ratio
            return  self.right_sclera_ratio
        elif feature == "left eye":
            self.left_sclera_ratio  = self.sclera_ratio
            return  self.left_sclera_ratio
        
        self.get_gaze(self.right_sclera_ratio, self.left_sclera_ratio)
        
      
    def get_sclera_ratio(self, threshold_eye):
        
        height, width        = threshold_eye.shape    
        left_side_threshold  = threshold_eye[0: height, 0: int(width / 2)]
        left_side_white      = cv2.countNonZero(left_side_threshold)
        right_side_threshold = threshold_eye[0: height, int(width / 2): width]
        right_side_white     = cv2.countNonZero(right_side_threshold) 
        
        try:
            sclera_ratio = left_side_white / right_side_white
        except:
            sclera_ratio = 1.
        return sclera_ratio 
    
    
    def get_gaze(self, right_sclera_ratio, left_sclera_ratio ):
        
        # Gaze detection
        if (right_sclera_ratio + left_sclera_ratio)/2 >= 1.50:
            cv2.putText(self.frame, 
                        "Right", (50, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX,
                        7, (255, 255, 255),10)
        
        elif 0.50 < (right_sclera_ratio + left_sclera_ratio)/2< 1.50:
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
        
        
    def get_blink_ratio(self, eye_points, facial_landmarks):
        
        #loading all the required points
        self.corner_left_e  = (facial_landmarks.part(eye_points[0]).x, 
                                facial_landmarks.part(eye_points[0]).y)
        self.corner_right_e = (facial_landmarks.part(eye_points[3]).x, 
                                facial_landmarks.part(eye_points[3]).y)
        
        self.center_top_e    = midpoint(facial_landmarks.part(eye_points[1]), 
                                        facial_landmarks.part(eye_points[2]))
        self.center_bottom_e = midpoint(facial_landmarks.part(eye_points[5]), 
                                        facial_landmarks.part(eye_points[4]))
    
        self.horizontal_length = euclidean_distance(self.corner_left_e,self.corner_right_e)
        self.vertical_length   =   euclidean_distance(self.center_top_e,self.center_bottom_e)
    
        self.blink_ratio = self.horizontal_length / self.vertical_length
    
        return self.blink_ratio
    
        
if __name__ == '__main__':
    camera = Camera()
    camera.get_camera(max_time=30)
    
    
            
            
            
            
            
            
            
            
            
            
            
            
            
            