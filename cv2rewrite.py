import cv2

import os
# cap = cv2.VideoCapture(0)
# for i in range(5,13):
# cap = cv2.VideoCapture('/home/n200/yoga_video/cut_video/yoga_30m_'+str(3)+'.mp4')
cap = cv2.VideoCapture('/home/n200/drc/ceo_demo_blazepose/video/video4.mp4')

# cap = cv2.VideoCapture(video_path)
width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
output_path = '/home/n200/drc/ceo_demo_blazepose/video/video4_re.mp4'
# print(output_path)
writer = cv2.VideoWriter(output_path, fourcc, 30, (int(width), int(height)))
# count_frame=0

while cap.isOpened():
  
  success, image = cap.read()
  if not success:
    print("Ignoring empty camera frame.")
    # If loading a video, use 'break' instead of 'continue'.
    break

#   # To improve performance, optionally mark the image as not writeable to
#   # pass by reference.
#   image.flags.writeable = False
#   image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#   results = pose.process(image)

#   #print landmarks
#   # if results.pose_landmarks:
  
#   #   # Iterate two times as we only want to display first two landmarks.
#   #   for i in range(2):
        
#   #       # Display the found normalized landmarks.
#   #       print(f'{mp_pose.PoseLandmark(i).name}:\n{results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value]}') 


#   image_height, image_width, _ = image.shape

#   # # Check if any landmarks are found.
#   # if results.pose_landmarks:
  
#   # # Iterate two times as we only want to display first two landmark.
#   #   for i in range(2):
        
#   #       # Display the found landmarks after converting them into their original scale.
#   #       print(f'{mp_pose.PoseLandmark(i).name}:') 
#   #       print(f'x: {results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].x * image_width}')
#   #       print(f'y: {results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].y * image_height}')
#   #       print(f'z: {results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].z * image_width}')
#   #       print(f'visibility: {results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].visibility}\n')

#   # # Draw the pose annotation on the image.
#   image.flags.writeable = True
#   image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#   # mp_drawing.plot_landmarks(imageresults.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
#   f = plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS,count_frame=count_frame )
  
  
  
#   count_frame+=1
#   # print('shape image3d:',image_3d.shape)

#   mp_drawing.draw_landmarks(
#       image,
#       results.pose_landmarks,
#       mp_pose.POSE_CONNECTIONS,
#       landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
#   # Flip the image horizontally for a selfie-view display.
#   t2=time.time()
#   spend_time = t2-t1

#   cv2.putText(image,str(round(1/spend_time,2)),(200,100),cv2.FONT_HERSHEY_SIMPLEX,2, (255,0,255))
  cv2.imshow('MediaPipe Pose', image)
  writer.write(image)
  if cv2.waitKey(1) & 0xFF == 27:
    break
cap.release()