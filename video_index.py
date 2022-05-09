import cv2
import  os
# 
# video_folder  ='./video'
# 
# video_name_list= os.listdir(video_folder)


# for video_name in video_name_list:
# video_path = os.path.join(video_folder,video_name)
video_path = './video4.mp4'
cap = cv2.VideoCapture(video_path)
width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
print(video_path)
output_path = './video_index/video4_out.mp4'
print(output_path)
index = 0
writer = cv2.VideoWriter(output_path, fourcc, 30, (int(width), int(height)))
while True:
    
        ret,frame = cap.read()
        if not ret:
            break

        cv2.putText(frame,str(index),(200,100),cv2.FONT_HERSHEY_SIMPLEX,2, (255,0,255))
        cv2.imshow('MediaPipe Pose', frame)
        writer.write(frame)
        index+=1
        if cv2.waitKey(1) == ord('q'):
            break
cap.release()
writer.release()