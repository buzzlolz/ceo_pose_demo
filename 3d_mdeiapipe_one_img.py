import cv2
import mediapipe as mp
import time
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
from drawing_utils import ori_plot_landmarks

from drawing_utils import plot_landmarks

# cap = cv2.VideoCapture(0)
# for i in range(5,13):
# cap = cv2.VideoCapture('/home/n200/yoga_video/cut_video/yoga_30m_'+str(3)+'.mp4')
# cap = cv2.VideoCapture('/home/n200/drc/pitch_action_detection_module/yoga_video/body.mp4')


with mp_pose.Pose(
    min_detection_confidence=0.1,
    min_tracking_confidence=0.5,
    model_complexity=1
    ) as pose:
    image = cv2.imread('yg1.jpg')
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    #print landmarks
    if results.pose_landmarks:
    
      # Iterate two times as we only want to display first two landmarks.
      for i in range(33):
          
          # Display the found normalized landmarks.
          print(f'{mp_pose.PoseLandmark(i).name}:\n{results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value]}') 
  

    image_height, image_width, _ = image.shape
 
    # # Check if any landmarks are found.
    # if results.pose_landmarks:
    
    # # Iterate two times as we only want to display first two landmark.
    #   for i in range(2):
          
    #       # Display the found landmarks after converting them into their original scale.
    #       print(f'{mp_pose.PoseLandmark(i).name}:') 
    #       print(f'x: {results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].x * image_width}')
    #       print(f'y: {results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].y * image_height}')
    #       print(f'z: {results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].z * image_width}')
    #       print(f'visibility: {results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].visibility}\n')

    # # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
    ori_plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
    # plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS,count_frame=count_frame )
    # count_frame+=1
    # print('shape image3d:',image_3d.shape)
    annotated_image = image.copy()
    mp_drawing.draw_landmarks(
        annotated_image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    # Flip the image horizontally for a selfie-view display.
    # t2=time.time()
    # spend_time = t2-t1

    # cv2.putText(image,str(round(1/spend_time,2)),(200,100),cv2.FONT_HERSHEY_SIMPLEX,2, (255,0,255))
    # image = cv2.cvtColor(image ,COLOR_RGB2BGR)
    cv2.imshow('MediaPipe Pose', annotated_image)
    cv2.imwrite('test.jpg',annotated_image)
    cv2.waitKey(0)
cv2.destroyAllWindows()
    # writer.write(image)
    # if cv2.waitKey(1) & 0xFF == 27:
    #   break
  # cap.release()
  # writer.release()
