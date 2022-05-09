import cv2
import mediapipe as mp
from drawing_utils import plot_landmarks
import time
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
import numpy as np
# from drawing_utils import plot_landmarks
import os
import cv2
import math
import vg
# class classifyPose(landmark)

class BlazePose_inf():
  def __init__(self):

    self.mp_drawing = mp.solutions.drawing_utils
    self.mp_drawing_styles = mp.solutions.drawing_styles
    self.mp_pose = mp.solutions.pose
    self.pose = mp_pose.Pose(
      min_detection_confidence=0.1,
      min_tracking_confidence=0.5,
      model_complexity=1
      ) 
  

  def detect_img(self,img):
    landmarks = []
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, _ = image.shape
    results = self.pose.process(image)
    # print('result',results)
    self.mp_drawing.draw_landmarks(
          image,
          results.pose_landmarks,
          mp_pose.POSE_CONNECTIONS,
          landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    if results.pose_landmarks is not None:
      for landmark in results.pose_landmarks.landmark:
              
              # Append the landmark into the list.
              landmarks.append((int(landmark.x * width), int(landmark.y * height),
                                    (landmark.z * width)))
      
    # print(landmarks)
    # label = self.classifyPose(landmarks)
    # t1 = time.time()
    # fig,ax = plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
    # t2=time.time()
    # print('3dtime spend:',t2-t1)
    # fig ,ax= plot_landmarksplot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
    # return image,label,landmarks,fig,ax
    return image,landmarks,results.pose_world_landmarks,mp_pose.POSE_CONNECTIONS
    # return image,label,landmarks



  def compare_reverse(self,landmarks1,landmarks2):
    acc  = 0
    # print(type_pose)
    # print(landmarks2)
    if len(landmarks1)>0 and len(landmarks2)>0:

      right_elbow_angle1 = self.calculateAngle(landmarks1[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                          landmarks1[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                          landmarks1[mp_pose.PoseLandmark.LEFT_WRIST.value])
          
      # Get the angle between the right shoulder, elbow and wrist points. 
      left_elbow_angle1 = self.calculateAngle(landmarks1[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                        landmarks1[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                        landmarks1[mp_pose.PoseLandmark.RIGHT_WRIST.value])   
      
      # Get the angle between the left elbow, shoulder and hip points. 
      right_shoulder_angle1 = self.calculateAngle(landmarks1[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                          landmarks1[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                          landmarks1[mp_pose.PoseLandmark.LEFT_HIP.value])

      # Get the angle between the right hip, shoulder and elbow points. 
      left_shoulder_angle1 = self.calculateAngle(landmarks1[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                            landmarks1[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                            landmarks1[mp_pose.PoseLandmark.RIGHT_ELBOW.value])

      
      # Get the angle between the left elbow, shoulder and hip points. 
      right_waist_angle1 = self.calculateAngle(landmarks1[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                          landmarks1[mp_pose.PoseLandmark.LEFT_HIP.value],
                                          landmarks1[mp_pose.PoseLandmark.LEFT_KNEE.value])

      # Get the angle between the right hip, shoulder and elbow points. 
      left_waist_angle1 = self.calculateAngle(landmarks1[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                            landmarks1[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                            landmarks1[mp_pose.PoseLandmark.RIGHT_KNEE.value])

                                      

      # Get the angle between the left hip, knee and ankle points. 
      right_knee_angle1= self.calculateAngle(landmarks1[mp_pose.PoseLandmark.LEFT_HIP.value],
                                      landmarks1[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                      landmarks1[mp_pose.PoseLandmark.LEFT_ANKLE.value])

      # Get the angle between the right hip, knee and ankle points 
      left_knee_angle1 = self.calculateAngle(landmarks1[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                        landmarks1[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                        landmarks1[mp_pose.PoseLandmark.RIGHT_ANKLE.value])

      # Get the angle between the left hip, knee and ankle points. 
      right_foot_angle1= self.calculateAngle(landmarks1[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                      landmarks1[mp_pose.PoseLandmark.LEFT_ANKLE.value],
                                      landmarks1[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value])

      # Get the angle between the right hip, knee and ankle points 
      left_foot_angle1 = self.calculateAngle(landmarks1[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                        landmarks1[mp_pose.PoseLandmark.RIGHT_ANKLE.value],
                                        landmarks1[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value])

        


      
      left_elbow_angle2 = self.calculateAngle(landmarks2[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                          landmarks2[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                          landmarks2[mp_pose.PoseLandmark.LEFT_WRIST.value])
          
      # Get the angle between the right shoulder, elbow and wrist points. 
      right_elbow_angle2 = self.calculateAngle(landmarks2[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                        landmarks2[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                        landmarks2[mp_pose.PoseLandmark.RIGHT_WRIST.value])   
      
      # Get the angle between the left elbow, shoulder and hip points. 
      left_shoulder_angle2 = self.calculateAngle(landmarks2[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                          landmarks2[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                          landmarks2[mp_pose.PoseLandmark.LEFT_HIP.value])

      # Get the angle between the right hip, shoulder and elbow points. 
      right_shoulder_angle2 = self.calculateAngle(landmarks2[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                            landmarks2[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                            landmarks2[mp_pose.PoseLandmark.RIGHT_ELBOW.value])


      # Get the angle between the left elbow, shoulder and hip points. 
      left_waist_angle2 = self.calculateAngle(landmarks2[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                          landmarks2[mp_pose.PoseLandmark.LEFT_HIP.value],
                                          landmarks2[mp_pose.PoseLandmark.LEFT_KNEE.value])

      # Get the angle between the right hip, shoulder and elbow points. 
      right_waist_angle2 = self.calculateAngle(landmarks2[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                            landmarks2[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                            landmarks2[mp_pose.PoseLandmark.RIGHT_KNEE.value])

      # Get the angle between the left hip, knee and ankle points. 
      left_knee_angle2= self.calculateAngle(landmarks2[mp_pose.PoseLandmark.LEFT_HIP.value],
                                      landmarks2[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                      landmarks2[mp_pose.PoseLandmark.LEFT_ANKLE.value])

      # Get the angle between the right hip, knee and ankle points 
      right_knee_angle2 = self.calculateAngle(landmarks2[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                        landmarks2[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                        landmarks2[mp_pose.PoseLandmark.RIGHT_ANKLE.value])


      # Get the angle between the left hip, knee and ankle points. 
      left_foot_angle2= self.calculateAngle(landmarks2[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                      landmarks2[mp_pose.PoseLandmark.LEFT_ANKLE.value],
                                      landmarks2[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value])

      # Get the angle between the right hip, knee and ankle points 
      right_foot_angle2 = self.calculateAngle(landmarks2[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                        landmarks2[mp_pose.PoseLandmark.RIGHT_ANKLE.value],
                                        landmarks2[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value])
      
    # if type_pose == 'tree':
      # print('left_knee_angle1',left_knee_angle1,left_knee_angle2)
      # print('right_knee_angle1',right_knee_angle1,right_knee_angle2)
      # lt_knee = abs(left_knee_angle1-left_knee_angle2)
      # rt_knee = abs(right_knee_angle1-right_knee_angle2)
      # acc = 1-((lt_knee+rt_knee)/360/2)
      left_elbow_diff = abs(left_elbow_angle2-left_elbow_angle1)
      right_elbow_diff = abs(right_elbow_angle1-right_elbow_angle2)
      left_shoulder_diff = abs(left_shoulder_angle1-left_shoulder_angle2)
      right_shoulder_diff = abs(right_shoulder_angle1-right_shoulder_angle2)
      left_waist_diff = abs(left_waist_angle1-left_waist_angle2)
      right_waist_diff = abs(right_waist_angle1-right_waist_angle2)
      left_knee_diff = abs(left_knee_angle1-left_knee_angle2)
      right_knee_diff =abs(right_knee_angle1-right_knee_angle2)
      left_foot_diff = abs(left_foot_angle1-left_foot_angle2)
      right_foot_diff = abs(right_foot_angle1-right_foot_angle2)


      # print('left_elbow_diff',left_elbow_angle2)
      # acc = 1-((left_elbow_diff+right_elbow_diff+left_shoulder_diff+right_shoulder_diff+left_waist_diff+right_waist_diff+left_knee_diff+right_knee_diff+left_foot_diff+right_foot_diff)/180/10)
      acc = 1-((left_elbow_diff+right_elbow_diff+left_shoulder_diff+right_shoulder_diff+left_waist_diff+right_waist_diff+left_knee_diff+right_knee_diff)/180/8)
      
      
      # if acc<0:
        # acc=0
    #   elif type_pose =='war_2':
    #     knee1 = abs(max(left_knee_angle1,right_knee_angle1)-max(left_knee_angle2,right_knee_angle2))

    #     knee2 = abs(min(left_knee_angle1,right_knee_angle1)-min(left_knee_angle2,right_knee_angle2))
    #     acc = 1-((knee1+knee2)/360/2)
    # print(acc)
    return round(acc,2)

  def compare(self,landmarks1,landmarks2):
    acc  = 0
    # print(type_pose)
    # print(landmarks2)
    if len(landmarks1)>0 and len(landmarks2)>0:

      left_elbow_angle1 = self.calculateAngle(landmarks1[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                          landmarks1[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                          landmarks1[mp_pose.PoseLandmark.LEFT_WRIST.value])
          
      # Get the angle between the right shoulder, elbow and wrist points. 
      right_elbow_angle1 = self.calculateAngle(landmarks1[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                        landmarks1[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                        landmarks1[mp_pose.PoseLandmark.RIGHT_WRIST.value])   
      
      # Get the angle between the left elbow, shoulder and hip points. 
      left_shoulder_angle1 = self.calculateAngle(landmarks1[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                          landmarks1[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                          landmarks1[mp_pose.PoseLandmark.LEFT_HIP.value])

      # Get the angle between the right hip, shoulder and elbow points. 
      right_shoulder_angle1 = self.calculateAngle(landmarks1[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                            landmarks1[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                            landmarks1[mp_pose.PoseLandmark.RIGHT_ELBOW.value])

      
      # Get the angle between the left elbow, shoulder and hip points. 
      left_waist_angle1 = self.calculateAngle(landmarks1[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                          landmarks1[mp_pose.PoseLandmark.LEFT_HIP.value],
                                          landmarks1[mp_pose.PoseLandmark.LEFT_KNEE.value])

      # Get the angle between the right hip, shoulder and elbow points. 
      right_waist_angle1 = self.calculateAngle(landmarks1[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                            landmarks1[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                            landmarks1[mp_pose.PoseLandmark.RIGHT_KNEE.value])

                                      

      # Get the angle between the left hip, knee and ankle points. 
      left_knee_angle1= self.calculateAngle(landmarks1[mp_pose.PoseLandmark.LEFT_HIP.value],
                                      landmarks1[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                      landmarks1[mp_pose.PoseLandmark.LEFT_ANKLE.value])

      # Get the angle between the right hip, knee and ankle points 
      right_knee_angle1 = self.calculateAngle(landmarks1[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                        landmarks1[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                        landmarks1[mp_pose.PoseLandmark.RIGHT_ANKLE.value])

      # Get the angle between the left hip, knee and ankle points. 
      left_foot_angle1= self.calculateAngle(landmarks1[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                      landmarks1[mp_pose.PoseLandmark.LEFT_ANKLE.value],
                                      landmarks1[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value])

      # Get the angle between the right hip, knee and ankle points 
      right_foot_angle1 = self.calculateAngle(landmarks1[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                        landmarks1[mp_pose.PoseLandmark.RIGHT_ANKLE.value],
                                        landmarks1[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value])

        


      
      left_elbow_angle2 = self.calculateAngle(landmarks2[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                          landmarks2[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                          landmarks2[mp_pose.PoseLandmark.LEFT_WRIST.value])
          
      # Get the angle between the right shoulder, elbow and wrist points. 
      right_elbow_angle2 = self.calculateAngle(landmarks2[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                        landmarks2[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                        landmarks2[mp_pose.PoseLandmark.RIGHT_WRIST.value])   
      
      # Get the angle between the left elbow, shoulder and hip points. 
      left_shoulder_angle2 = self.calculateAngle(landmarks2[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                          landmarks2[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                          landmarks2[mp_pose.PoseLandmark.LEFT_HIP.value])

      # Get the angle between the right hip, shoulder and elbow points. 
      right_shoulder_angle2 = self.calculateAngle(landmarks2[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                            landmarks2[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                            landmarks2[mp_pose.PoseLandmark.RIGHT_ELBOW.value])


      # Get the angle between the left elbow, shoulder and hip points. 
      left_waist_angle2 = self.calculateAngle(landmarks2[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                          landmarks2[mp_pose.PoseLandmark.LEFT_HIP.value],
                                          landmarks2[mp_pose.PoseLandmark.LEFT_KNEE.value])

      # Get the angle between the right hip, shoulder and elbow points. 
      right_waist_angle2 = self.calculateAngle(landmarks2[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                            landmarks2[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                            landmarks2[mp_pose.PoseLandmark.RIGHT_KNEE.value])

      # Get the angle between the left hip, knee and ankle points. 
      left_knee_angle2= self.calculateAngle(landmarks2[mp_pose.PoseLandmark.LEFT_HIP.value],
                                      landmarks2[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                      landmarks2[mp_pose.PoseLandmark.LEFT_ANKLE.value])

      # Get the angle between the right hip, knee and ankle points 
      right_knee_angle2 = self.calculateAngle(landmarks2[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                        landmarks2[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                        landmarks2[mp_pose.PoseLandmark.RIGHT_ANKLE.value])


      # Get the angle between the left hip, knee and ankle points. 
      left_foot_angle2= self.calculateAngle(landmarks2[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                      landmarks2[mp_pose.PoseLandmark.LEFT_ANKLE.value],
                                      landmarks2[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value])

      # Get the angle between the right hip, knee and ankle points 
      right_foot_angle2 = self.calculateAngle(landmarks2[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                        landmarks2[mp_pose.PoseLandmark.RIGHT_ANKLE.value],
                                        landmarks2[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value])
      
    # if type_pose == 'tree':
      # print('left_knee_angle1',left_knee_angle1,left_knee_angle2)
      # print('right_knee_angle1',right_knee_angle1,right_knee_angle2)
      # lt_knee = abs(left_knee_angle1-left_knee_angle2)
      # rt_knee = abs(right_knee_angle1-right_knee_angle2)
      # acc = 1-((lt_knee+rt_knee)/360/2)
      left_elbow_diff = abs(left_elbow_angle2-left_elbow_angle1)
      right_elbow_diff = abs(right_elbow_angle1-right_elbow_angle2)
      left_shoulder_diff = abs(left_shoulder_angle1-left_shoulder_angle2)
      right_shoulder_diff = abs(right_shoulder_angle1-right_shoulder_angle2)
      left_waist_diff = abs(left_waist_angle1-left_waist_angle2)
      right_waist_diff = abs(right_waist_angle1-right_waist_angle2)
      left_knee_diff = abs(left_knee_angle1-left_knee_angle2)
      right_knee_diff =abs(right_knee_angle1-right_knee_angle2)
      left_foot_diff = abs(left_foot_angle1-left_foot_angle2)
      right_foot_diff = abs(right_foot_angle1-right_foot_angle2)


      # print('left_elbow_diff',left_elbow_angle2)
      acc = 1-((left_elbow_diff+right_elbow_diff+left_shoulder_diff+right_shoulder_diff+left_waist_diff+right_waist_diff+left_knee_diff+right_knee_diff+left_foot_diff+right_foot_diff)/180/10)
      # if acc<0:
        # acc=0
    #   elif type_pose =='war_2':
    #     knee1 = abs(max(left_knee_angle1,right_knee_angle1)-max(left_knee_angle2,right_knee_angle2))

    #     knee2 = abs(min(left_knee_angle1,right_knee_angle1)-min(left_knee_angle2,right_knee_angle2))
    #     acc = 1-((knee1+knee2)/360/2)
    # print(acc)
    return acc

  
  def classifyPose(self,landmarks, display=False):
    
    
    # Initialize the label of the pose. It is not known at this stage.
    label = 'Unknown Pose'
    # print(mp_pose.PoseLandmark.LEFT_ELBOW)

    # Specify the color (Red) with which the label will be written on the image.
    color = (0, 0, 255)
    
    # Calculate the required angles.
    #----------------------------------------------------------------------------------------------------------------
    
    if len(landmarks)!=0:
      # Get the angle between the left shoulder, elbow and wrist points. 
      left_elbow_angle = self.calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
      
      # Get the angle between the right shoulder, elbow and wrist points. 
      right_elbow_angle = self.calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                        landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])   
      
      # Get the angle between the left elbow, shoulder and hip points. 
      left_shoulder_angle = self.calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                          landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                          landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])

      # Get the angle between the right hip, shoulder and elbow points. 
      right_shoulder_angle = self.calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])

      # Get the angle between the left hip, knee and ankle points. 
      left_knee_angle = self.calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])

      # Get the angle between the right hip, knee and ankle points 
      right_knee_angle = self.calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
      

      # print('left_knee_angle',left_knee_angle)
      # print('right_knee_angle',right_knee_angle)
      # # print()
      # if left_elbow_angle>60 and left_elbow_angle<120:
      #   label = 'raise left hand'
      # elif right_elbow_angle>60 and right_elbow_angle<120:
      #   label = 'raise right hand'
      
      
      # elif left_knee_angle > 160 and left_knee_angle < 195 and right_knee_angle > 160 and right_knee_angle < 195:
  
      #             # Specify the label of the pose that is tree pose.
      #             label = 'T Pose'

      if left_knee_angle > 165 and left_knee_angle < 195 or right_knee_angle > 165 and right_knee_angle < 195:
  
          # Check if the other leg is bended at the required angle.
          if left_knee_angle > 300 and left_knee_angle < 360 or right_knee_angle > 15 and right_knee_angle < 60:
  
              # Specify the label of the pose that is tree pose.
              label = 'Tree Pose'
      # Check if it is the warrior II pose.
      #----------------------------------------------------------------------------------------------------------------

      # Check if one leg is straight.
      if left_knee_angle > 150 and left_knee_angle < 215 or right_knee_angle > 150 and right_knee_angle < 215:
          
          # Check if the other leg is bended at the required angle.
          if left_knee_angle > 80 and left_knee_angle < 130 or right_knee_angle > 80 and right_knee_angle < 130:

              # Specify the label of the pose that is Warrior II pose.
              label = 'Warrior II Pose' 


    return label
  def rcalculateAngle(sepf,point_1, point_2, point_3):
    """
    根据三点坐标计算夹角
    :param point_1: 点1坐标
    :param point_2: 点2坐标
    :param point_3: 点3坐标
    :return: 返回任意角的夹角值，这里只是返回点2的夹角
    """
    a=math.sqrt((point_2[0]-point_3[0])*(point_2[0]-point_3[0])+(point_2[1]-point_3[1])*(point_2[1] - point_3[1]))
    b=math.sqrt((point_1[0]-point_3[0])*(point_1[0]-point_3[0])+(point_1[1]-point_3[1])*(point_1[1] - point_3[1]))
    c=math.sqrt((point_1[0]-point_2[0])*(point_1[0]-point_2[0])+(point_1[1]-point_2[1])*(point_1[1]-point_2[1]))
    A=math.degrees(math.acos((a*a-b*b-c*c)/(-2*b*c)))
    B=math.degrees(math.acos((b*b-a*a-c*c)/(-2*a*c)))
    C=math.degrees(math.acos((c*c-a*a-b*b)/(-2*a*b)))
    return B

  def calculateAngle(self,landmark1, landmark2, landmark3):
 
    
 
    # Get the required landmarks coordinates.
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3
 
    # Calculate the angle between the three points
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    
    # Check if the angle is less than zero.
    if angle <0:
 
        # Add 360 to the found angle.
        # angle += 360
        angle = 360+angle
    
    # Return the calculated angle.
    return angle

  # def get_keypoint(self)
  def rcalculateAngle(self,landmark1, landmark2, landmark3):
 
    
 
    # Get the required landmarks coordinates.
    x1, y1, z1 = landmark1
    x2, y2, z2 = landmark2
    x3, y3, z3 = landmark3
    v1 = np.array([x2-x1,y2-y1,z2-z1])

    v2 = np.array([x2-x3,y2-y3,z2-z3])
    angle = vg.angle(v1,v2)
    # Calculate the angle between the three points
    # angle = math.degrees(math.atan2(y/s3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    
    # Check if the angle is less than zero.
    if angle <0:
 
        # Add 360 to the found angle.
        # angle += 360
        angle = 360+angle
    
    # Return the calculated angle.
    return angle

  def check_angle(self,angle):
    if angle>=180:
      return 360-angle
    else:
      return angle
  def get_each_keypoint_distance(self,landmarks_video,landmarks_live,action_index):
        # video_body_length = landmarks_video[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        if len(landmarks_live)>0 and len(landmarks_video)>0:
          # print(landmarks_video[mp_pose.PoseLandmark.LEFT_HIP.value])
          # print(landmarks_video[mp_pose.PoseLandmark.RIGHT_HIP.value])
          # crotch_point =( (np.array(landmarks_video[mp_pose.PoseLandmark.LEFT_HIP.value]) + np.array(landmarks_video[mp_pose.PoseLandmark.RIGHT_HIP.value]))/2).astype(int)
          # # print(crotch_point)
          # crotch_angle = self.calculateAngle(landmarks_video[mp_pose.PoseLandmark.LEFT_KNEE.value],
          #                               crotch_point,
          #                               landmarks_video[mp_pose.PoseLandmark.RIGHT_KNEE.value])
          video_left_knee_angle= self.calculateAngle(landmarks_video[mp_pose.PoseLandmark.LEFT_HIP.value],
                                        landmarks_video[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                        landmarks_video[mp_pose.PoseLandmark.LEFT_ANKLE.value])
          video_right_knee_angle = self.calculateAngle(landmarks_video[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                          landmarks_video[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                          landmarks_video[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
          video_crotch_point =( (np.array(landmarks_video[mp_pose.PoseLandmark.LEFT_HIP.value]) + np.array(landmarks_video[mp_pose.PoseLandmark.RIGHT_HIP.value]))/2).astype(int)
          

          video_crotch_angle = self.calculateAngle(landmarks_video[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                        video_crotch_point,
                                        landmarks_video[mp_pose.PoseLandmark.RIGHT_KNEE.value])
          video_body_point = ( (np.array(landmarks_video[mp_pose.PoseLandmark.LEFT_SHOULDER.value]) + np.array(landmarks_video[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]))/2).astype(int)
          video_right_hip_angle = self.calculateAngle(video_body_point,
                                        video_crotch_point,
                                        landmarks_video[mp_pose.PoseLandmark.RIGHT_KNEE.value])
          video_left_hip_angle = self.calculateAngle(video_body_point,
                                        video_crotch_point,
                                        landmarks_video[mp_pose.PoseLandmark.LEFT_KNEE.value])

          
          body_point = ( (np.array(landmarks_live[mp_pose.PoseLandmark.LEFT_SHOULDER.value]) + np.array(landmarks_live[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]))/2).astype(int)
          crotch_point =( (np.array(landmarks_live[mp_pose.PoseLandmark.LEFT_HIP.value]) + np.array(landmarks_live[mp_pose.PoseLandmark.RIGHT_HIP.value]))/2).astype(int)
          # print(crotch_point)
          crotch_angle = self.calculateAngle(landmarks_live[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                        crotch_point,
                                        landmarks_live[mp_pose.PoseLandmark.RIGHT_KNEE.value])
          left_knee_angle= self.calculateAngle(landmarks_live[mp_pose.PoseLandmark.LEFT_HIP.value],
                                        landmarks_live[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                        landmarks_live[mp_pose.PoseLandmark.LEFT_ANKLE.value])
          right_knee_angle = self.calculateAngle(landmarks_live[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                          landmarks_live[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                          landmarks_live[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
          
          right_hip_angle = self.calculateAngle(body_point,
                                        video_crotch_point,
                                        landmarks_live[mp_pose.PoseLandmark.RIGHT_KNEE.value])
          left_hip_angle = self.calculateAngle(body_point,
                                        crotch_point,
                                        landmarks_live[mp_pose.PoseLandmark.LEFT_KNEE.value])


          crotch_angle = self.check_angle(crotch_angle)
          video_crotch_angle = self.check_angle(video_crotch_angle)
          right_knee_angle = self.check_angle(right_knee_angle)
          left_knee_angle = self.check_angle(left_knee_angle)
          right_knee_angle = self.check_angle(right_knee_angle)


          # print('crotch_angle',video_crotch_angle)
          if action_index ==1:
            # print('crotch_angle',video_crotch_angle)
            # print('right_knee_angle',right_knee_angle)
            # print('left_knee_angle',left_knee_angle)
            if crotch_angle>120 and right_knee_angle<150 and left_knee_angle<150:
              # print('ok')
              return 'ok'
            else:
              # print('need down')
              return 'sit'
          elif action_index==2:
            # print('right knee angle',right_knee_angle)
            # print('left kenn angle',video_left_knee_angle)
            # print('left knee point',landmarks_video[mp_pose.PoseLandmark.LEFT_KNEE.value])
            # print('right knee point',landmarks_video[mp_pose.PoseLandmark.RIGHT_KNEE.value])
            if landmarks_live[mp_pose.PoseLandmark.LEFT_KNEE.value][1]<landmarks_live[mp_pose.PoseLandmark.RIGHT_KNEE.value][1] and  left_knee_angle<60 : 
              # print('ok')
            # if landmarks_live[mp_pose.PoseLandmark.LEFT_KNEE.value][1]<landmarks_live[mp_pose.PoseLandmark.RIGHT_KNEE.value][1]:
              return 'ok'
            else:
              # print('raise_leg_left')
              return 'raise_leg_left'
          elif action_index==3:
            # print('right knee angle',video_right_knee_angle)
            # print('left kenn angle',video_left_knee_angle)
            # print('left knee point',landmarks_video[mp_pose.PoseLandmark.LEFT_KNEE.value])
            # print('right knee point',landmarks_video[mp_pose.PoseLandmark.RIGHT_KNEE.value])
            if landmarks_live[mp_pose.PoseLandmark.RIGHT_KNEE.value][1]<landmarks_live[mp_pose.PoseLandmark.LEFT_KNEE.value][1] and  right_knee_angle<60 : 
              # print('ok')
              return 'ok'
            else:
              # print('raise_leg_right')
              return 'raise_leg_right'
          elif action_index==4:
            print('crotch_angle',crotch_angle)
            print('left_knee_angle',left_knee_angle)
            # print('right knee angle',vidseo_right_knee_angle)
            # print('left kenn angle',video_left_knee_angle)
            # print('left knee point',landmarks_video[mp_pose.PoseLandmark.LEFT_KNEE.value])
            # print('right knee point',landmarks_video[mp_pose.PoseLandmark.RIGHT_KNEE.value])
            # print('video_crotch_angle',video_crotch_angle)
            if 70<left_knee_angle< 135 and crotch_angle >95: 
              # print('ok')
              return 'ok'
            # elif right_hip_angle<200:
              # return 'straight_body'
            else:
              # print('curve_left')
              return 'curve_left'
          elif action_index==5:
            print('video_right_hip_angle',video_right_hip_angle)
            print('video_left_hip_angle',video_left_hip_angle)
            # print('right knee angle',video_right_knee_angle)
            # print('left kenn angle',video_left_knee_angle)
            # print('left knee point',landmarks_video[mp_pose.PoseLandmark.LEFT_KNEE.value])
            # print('right knee point',landmarks_video[mp_pose.PoseLandmark.RIGHT_KNEE.value])
            # print('video_crotch_angle',video_crotch_angle)
            if  70<right_knee_angle< 135 and crotch_angle >95: 
              # print('ok')
              return 'ok'
            # elif right_hip_angle>165:
              # return 'straight_body'
            else:
              # print('curve_right')
              return 'curve_right'

          else:
            return None
          



# if __name__ == '__main__':
#   a=BlazePose_inf()
#   img = cv2.imread('war_2.jpg')
#   # a.detect_img(img
#   landmarks = []
#   image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#   height, width, _ = image.shape
#   mp_drawing = mp.solutions.drawing_utils
#   mp_drawing_styles = mp.solutions.drawing_styles
#   mp_pose = mp.solutions.pose
#   pose=mp_pose.Pose(
#       min_detection_confidence=0.5,
#       min_tracking_confidence=0.5,
#       model_complexity=1)
  

#   results = pose.process(image)
#   mp_drawing.draw_landmarks(
#         image,
#         results.pose_landmarks,
#         mp_pose.POSE_CONNECTIONS,
#         landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
#   if results.pose_landmarks is not None:
#     for landmark in results.pose_landmarks.landmark:
            
#             # Append the landmark into the list.
#             landmarks.append((int(landmark.x * width), int(landmark.y * height),
#                                   (landmark.z * width)))
#   landmarksnp = np.array(landmarks)
#   np.save('war_2.npy',landmarksnp)
#   # print(landmarks)
  
