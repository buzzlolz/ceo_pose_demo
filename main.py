from PyQt5 import QtWidgets,uic,QtGui,QtCore
from PyQt5.QtCore import *

import sys
from PyQt5.QtGui import *
import cv2
from PyQt5.QtWidgets import *
import blazepose_inf
from PyQt5.QtWidgets import QFileDialog
import numpy as np
import os
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg  as FigureCanvas
import matplotlib.pyplot as plt
# from matplotlib.figure import Figure 
from matplotlib.animation import FuncAnimation
import  time
import random


from PyQt5.QtMultimedia import *
from PyQt5 import QtMultimedia
from PyQt5.QtMultimediaWidgets import *

import os
import pandas as pd
import dataclasses
from typing import List, Mapping, Optional, Tuple, Union
from mediapipe.framework.formats import detection_pb2
from mediapipe.framework.formats import location_data_pb2
from mediapipe.framework.formats import landmark_pb2

from scipy.spatial.transform import Rotation as R

WHITE_COLOR = (224, 224, 224)
BLACK_COLOR = (0, 0, 0)
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 128, 0)
BLUE_COLOR = (255, 0, 0)

_PRESENCE_THRESHOLD = 0.5
_VISIBILITY_THRESHOLD = 0.0
_RGB_CHANNELS = 3

from pygame import mixer
import tempfile
import pygame

VideoCapture_Id = 0


@dataclasses.dataclass
class DrawingSpec:
  # Color for drawing the annotation. Default to the white color.
  color: Tuple[int, int, int] = WHITE_COLOR
  # Thickness for drawing the annotation. Default to 2 pixels.
  thickness: int = 2
  # Circle radius. Default to 2 pixels.
  circle_radius: int = 2


# class Read_Video(QtCore.QThread):
#     changePixmap = pyqtSignal(object)
#     fps =''

#     def run(self):
#         cap = cv2.VideoCapture('./video/4min_video.mp4')
#         fps = cap.get(cv2.CAP_PROP_FPS) 
#         print(fps)

#         while True:
#             ret, frame = cap.read()
#             if ret:
                
#                 # rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 # h, w, ch = rgbImage.shape
#                 # bytesPerLine = ch * w
#                 # convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
#                 # p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
#                 # print(int(500/fps))
                
                
#                 self.changePixmap.emit(frame)
                
class BlazePose_Thread(QtCore.QThread):
    keypoint_frame_sig = pyqtSignal(object)
    live_landmarks_sig = pyqtSignal(object)
    pose_world_landmarks_sig = pyqtSignal(object)
    pose_connection_sig = pyqtSignal(object)
    def __init__(self,frame):
        super(BlazePose_Thread,self).__init__()
        self.blazepose = blazepose_inf.BlazePose_inf()
        self.video_status=False
        self.frame = frame
        
    def run(self):
        while True :
            if self.video_status:
                if self.frame !='':
                    # print(self.frame)
                    # rtn_result = []
                    rtn_img ,live_landmarks,pose_world_landmarks,pose_connection=self.blazepose.detect_img(self.frame)
                    
                    self.keypoint_frame_sig.emit(rtn_img)
                    self.live_landmarks_sig.emit(live_landmarks)
                    self.pose_world_landmarks_sig.emit(pose_world_landmarks)
                    self.pose_connection_sig.emit(pose_connection)

            
    def video_start(self):
        
        self.video_status = True
        print('blazepoe change stus',self.video_status)
    def change_frame(self,img):
        self.frame = img


class Cam(QtCore.QThread):
    frame_sig = pyqtSignal(object)
    
    def __init__(self):
        super(Cam,self).__init__()
        self.cap  = cv2.VideoCapture(VideoCapture_Id)
        self.video_status=False
    def run(self):
        while self.cap.isOpened() :
            try:
                ret,frame = self.cap.read()
                if ret and self.video_status:
                    self.frame_sig.emit(frame)

            except Exception as e:
                print(str(e))
    def video_start(self):
        self.video_status = True


class Ui(QtWidgets.QMainWindow):
    video_open_sig = pyqtSignal(bool)
    def __init__(self):
        super(Ui,self).__init__()
        # self.sub_window = SubWindow()
        self.window = uic.loadUi('compare_pose4.ui',self)
        self.window.pushButton_load_video.clicked.connect(self.openFile)
        self.window.pushButton_start_video.clicked.connect(self.play)
        self.window.pushButton_update.clicked.connect(self.click_update)
        self.select_video_path = ''
        self.sample_video_action_count = 0
        self.video_frame_index = 0
        self.pose_similarity_score = 0
        self.grade = 'None'
        self.pose_action_score_list = []
        self.action_index = 1
        self.action_time_list=[]
        self.video_keypoint_npy_array=[]
        self.video_landmarks=[]
        self.live_landmarks=[]
        self.df_csv=[]
        self.live_pose_world_landmarks= []
        self.video_pose_world_landmarks= None
        self.video_world_npy_array = []
        self.pose_connection = ''
        self.response=''
        self.pose3d_direction = ''
        self.pose3d_waist_angle = ''
        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        
        pygame.mixer.init()

        
        self.window.horizontalSlider.setRange(0, 0)
        self.window.horizontalSlider.sliderMoved.connect(self.setPosition)

        # self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile('/home/n200/drc/ceo_demo_blazepose/video/4min_video.mp4')))

        # Set widget
        self.videoWidget = QVideoWidget(self.window.centralwidget)
        self.videoWidget.setGeometry(750, 30, 640, 480)
        # self.setCentralWidget(self.videoWidget)
        self.mediaPlayer.setVideoOutput(self.videoWidget)


        # self.mediaPlayer.setVideoOutput(videoWidget)
        # self.mediaPlayer.stateChanged.connect(self.mediaStateChanged)
        # self.mediaPlayer.positionChanged.connect(self.positionChanged)
        self.mediaPlayer.durationChanged.connect(self.durationChanged)
        self.mediaPlayer.error.connect(self.handleError)

        # self.window.pushButton_load_video.clicked.connect(self.load_video)
        self.blazepose = blazepose_inf.BlazePose_inf()

        self.CamThread = Cam()
        self.CamThread.frame_sig.connect(self.get_cv_frame)
        self.CamThread.start()
        self.actual_frame = ''
        self.BlazePose_Thread = BlazePose_Thread(self.actual_frame)
        self.BlazePose_Thread.keypoint_frame_sig.connect(self.refreshWindow)
        self.BlazePose_Thread.live_landmarks_sig.connect(self.refreshScore)
        self.BlazePose_Thread.pose_world_landmarks_sig.connect(self.refresh_pose_world_landmarks)
        self.BlazePose_Thread.pose_connection_sig.connect(self.refresh_pose_connection)
        self.BlazePose_Thread.start()
        # self.Read_Video = Read_Video()
        # self.Read_Video.changePixmap.connect(self.refreshSampleVideo)
        # self.Read_Video.start()
        self.mediaPlayer.setNotifyInterval(1)
        self.mediaPlayer.positionChanged.connect(self.update_video_frame)

       

       


        self.pushButton_camstart.clicked.connect(self.video_on)
        # self.sample_npy=[]
        # self.live_landmarks=[]

        
        # self.window.label_score.setFont(QFont('Times', 20))
        # self.window.label_grade.setFont(QFont('Times', 20))
        # self.window.label_score2.setFont(QFont('Times', 20))
        # self.window.label_grade2.setFont(QFont('Times', 20))
        # self.fig = Figure(figsize=(10, 10), dpi=100)
        # self.ax = self.fig.add_subplot(111)
  
        # self.count = 0
        # self.CamThread.frame_sig.connect()
        # layout = QtWidgets.QVBoxLayout(self.window)
        self.live_figure = plt.figure("live_figure") 
        self.live_ax = plt.axes(projection='3d')  # 三维坐标轴

        self.live_canvas = FigureCanvas(self.live_figure)         #增加画布

        # self.video_figure = plt.figure("video_figure") 
        # self.video_ax = plt.axes(projection='3d')  # 三维坐标轴

        # self.video_canvas = FigureCanvas(self.video_figure)         #增加画布
        
        # # self.testwidget = QVideoWidget(self.window.centralwidget)
        # # self.testwidget.setGeometry(10, 10, 700, 700)
        # self.window.Vertalayout_live.addWidget(self.live_canvas)  
        self.window.Vertalayout_video.addWidget(self.live_canvas) 
        self.mytimer = QTimer(self) 
        self.mytimer.timeout.connect(self.onTimer)
        self.mytimer.start(500)  
        self.speak_mode = ''
        self.window.horizontalSlider.sliderMoved.connect (self.setPosition)


        
        # self.showbox()   
    # def showbox(self): 
    #         self.ax.clear() 
    #         self.ax.bar3d(0,0,0,0,12,30, color="green",zsort='average',edgecolor='white',linewidth=0.5,alpha=1) 
    #         # self.ax.bar3d(0,0,0,self.spinBox.value(),self.spinBox_2.value(),self.spinBox_3.value(), color="green",zsort='average',edgecolor='red',linewidth=0.5,alpha=0) 

    #         self.canvas.draw()

    # def update_video_frame(self, position):

    #     # if len(self.action_time_list)<=0:
    #     #     self.action_index+=1
    #     #     df_index = 'a'+str(self.action_index)+'_start'
    #     #     self.action_time_list = self.df_csv['a1_start'].values

        
    #     # print('flash')
        
    #     self.video_frame_index  = round(self.mediaPlayer.position()/33.3)
    #     # print(self.action_time_list)
    #     if len(self.video_keypoint_npy_array)!=0:
    #         if self.video_frame_index<len(self.video_keypoint_npy_array):
    #             self.video_keypoint_frame = self.video_keypoint_npy_array[self.video_frame_index]
    

        
    #     if self.video_frame_index  in self.action_time_list:

    #         #count avg grade in one action
    #         if len(self.pose_action_score_list)>0:
    #             avg_score = sum(self.pose_action_score_list)/len(self.pose_action_score_list)
    #             if avg_score>0.80:
    #                 self.grade ='A'
    #             elif avg_score>=0.70:
    #                 self.grade = 'B'
    #             else:
    #                 self.grade = 'C'
    #             self.window.label_grade2.setText(self.grade)
                
            

    #         self.sample_video_action_count+=1
    #         self.window.label_frame_count.setFont(QFont('Times', 20))
    #         # self.window.label_frame_count.setText(str(self.sample_video_action_count))
    #         # print('in self.action_time_list',frame_in)
    #         self.action_time_list  = np.delete(self.action_time_list,0)
    #         if len(self.action_time_list) == 0:
    #             self.sample_video_action_count= 0
    #             self.action_index += 1
    #             self.action_time_list = self.df_csv['a'+str(self.action_index)+'_start'].values
    #             self.action_time_list=self.action_time_list[~np.isnan(self.action_time_list)]
        
    #     self.pose_action_score_list.clear()

    #     #get video world landmarks
    #     if len(self.video_world_npy_array)!=0:
    #         self.video_pose_world_landmarks =self.video_world_npy_array[self.video_frame_index]

    def setPosition(self,position):
        self.mediaPlayer.setPosition(position)

    def speak(self,mode):
        mp3_path = './audio/'+mode+'.mp3'
        t1 = time.time()
        
        with tempfile.NamedTemporaryFile(delete=True) as fp:
            
            mixer.music.load(mp3_path.format(fp.name))
            mixer.music.play()
            # print('speak')
       
    def onTimer(self):
        if self.speak_mode=='sit' :
            self.speak('sit')
        elif self.speak_mode=='raise_leg_right':
            self.speak('raise_leg_right')
        elif self.speak_mode=='raise_leg_left':
            self.speak('raise_leg_left')
        elif self.speak_mode=='curve_left':
            self.speak('curve_left')
        elif self.speak_mode=='curve_right':
            self.speak('curve_right')
        elif self.speak_mode == 'straight_body':
            self.speak('straight_body')
        else:
            pass
    def update_video_frame(self, position):

        # if len(self.action_time_list)<=0:
        #     self.action_index+=1
        #     df_index = 'a'+str(self.action_index)+'_start'
        #     self.action_time_list = self.df_csv['a1_start'].values

        self.window.horizontalSlider.setValue(position)
        
        # self.speak_mode=self.blazepose.get_each_keypoint_distance(self.video_landmarks,self.live_landmarks,self.action_index)
        self.speak_mode=self.blazepose.get_each_keypoint_distance_3d(self.video_pose_world_landmarks,self.live_pose_world_landmarks,self.action_index)
        
        # print('flash')
        # print('self.action index',self.action_index)
        self.video_frame_index  = round(self.mediaPlayer.position()/33.3)
        # print(self.action_time_list)
        if len(self.video_keypoint_npy_array)!=0:
            if self.video_frame_index<len(self.video_keypoint_npy_array):
                self.video_landmarks = self.video_keypoint_npy_array[self.video_frame_index]
    

        #get action id'
        if len(self.df_csv)!=0:
            # print(self.df_csv)
            # print(self.df_csv['a1_start'].values)
            if self.df_csv['a1_start'].values[0]<self.video_frame_index<=self.df_csv['a1_start'].values[1]:
                self.action_index=1
            elif self.df_csv['a2_start'].values[0]<self.video_frame_index<=self.df_csv['a2_start'].values[1]:
                self.action_index=2
            elif self.df_csv['a3_start'].values[0]<self.video_frame_index<=self.df_csv['a3_start'].values[1]:
                self.action_index=3
            elif self.df_csv['a4_start'].values[0]<self.video_frame_index<=self.df_csv['a4_start'].values[1]:
                self.action_index=4
            elif self.df_csv['a5_start'].values[0]<self.video_frame_index<=self.df_csv['a5_start'].values[1]:
                self.action_index=5
            else:
                self.action_index=-1
        # self.window.label_response_show.setText(self.response)
        if self.speak_mode=='sit':
            
            self.window.label_response_show.setText("蹲低一點")
        elif self.speak_mode=='raise_leg_right':
            self.window.label_response_show.setText("右腳抬高一點")
        elif self.speak_mode=='raise_leg_left':
            self.window.label_response_show.setText("左腳抬高一點")
        elif self.speak_mode=='curve_left':
            self.window.label_response_show.setText("左腳再彎一點")
        elif self.speak_mode=='curve_right':
            self.window.label_response_show.setText("右腳再彎一點")
        elif self.speak_mode=='straight_body':
            self.window.label_response_show.setText("身體挺直一點")
        elif self.speak_mode=='ok':
            self.window.label_response_show.setText('Good')
        
        else:
            self.speak_mode = ''
            self.window.label_response_show.setText("")
        
        # self.speak('sit')
        
            
        # if self.video_frame_index  in self.action_time_list:

        #     #count avg grade in one action
        #     if len(self.pose_action_score_list)>0:
        #         avg_score = sum(self.pose_action_score_list)/len(self.pose_action_score_list)
        #         if avg_score>0.80:
        #             self.grade ='A'
        #         elif avg_score>=0.70:
        #             self.grade = 'B'
        #         else:
        #             self.grade = 'C'
        #         self.window.label_grade2.setText(self.grade)
                
            

        #     self.sample_video_action_count+=1
        #     self.window.label_frame_count.setFont(QFont('Times', 20))
        #     # self.window.label_frame_count.setText(str(self.sample_video_action_count))
        #     # print('in self.action_time_list',frame_in)
        #     self.action_time_list  = np.delete(self.action_time_list,0)
        #     if len(self.action_time_list) == 0:
        #         self.sample_video_action_count= 0
        #         self.action_index += 1
        #         self.action_time_list = self.df_csv['a'+str(self.action_index)+'_start'].values
        #         self.action_time_list=self.action_time_list[~np.isnan(self.action_time_list)]
        
        # self.pose_action_score_list.clear()

        #get video world landmarks
        if len(self.video_world_npy_array)!=0:
            if self.video_frame_index<len(self.video_world_npy_array):
                self.video_pose_world_landmarks =self.video_world_npy_array[self.video_frame_index]
        

    def openFile(self):
        fileName, _ = QFileDialog.getOpenFileName(self,"select files","./video", "All Files (*);;Excel Files (*.xls)")


        if fileName:
            self.mediaPlayer.setMedia(
                QtMultimedia.QMediaContent(QtCore.QUrl.fromLocalFile(fileName)))
        video_name = os.path.basename(fileName).split('.')[0]

        
        self.load_csv(video_name)
        self.load_npy(video_name)
            # self.playButton.setEnabled(True)


    def load_csv(self,video_name):
        csv_path = os.path.join('csv',video_name+'.csv')
        self.df_csv = pd.read_csv(csv_path)
        
        self.action_time_list = self.df_csv['a1_start'].values
        self.action_time_list = self.action_time_list[~np.isnan(self.action_time_list)]
        
        
        

    def load_npy(self,video_name):
        npy_path = os.path.join('npy',video_name+'.npy')
        self.video_keypoint_npy_array = np.load(npy_path,allow_pickle=True)
        world_npy = os.path.join('npy',video_name+'_world.npy')
        self.video_world_npy_array = np.load(world_npy,allow_pickle=True)
        
    def load_npy_3d(self,video_name):
        npy_path = os.path.join('npy',video_name+'3d_.npy')
        self.video_keypoint_npy_array = np.load(npy_path,allow_pickle=True)
        world_npy = os.path.join('npy',video_name+'_world.npy')
        self.video_world_npy_array = np.load(world_npy,allow_pickle=True)
        

    def play(self):
        if self.mediaPlayer.state() == QtMultimedia.QMediaPlayer.PlayingState:
            self.mediaPlayer.pause()
            self.window.pushButton_start_video.setText('Play')
            # print(self.mediaPlayer.position())
           

        else:
            
            self.mediaPlayer.play()
            self.window.pushButton_start_video.setText('Pause')
          
            # print(self.mediaPlayer.position())

    def mediaStateChanged(self, state):
        if self.mediaPlayer.state() == QtMultimedia.QMediaPlayer.PlayingState:
            print('pause')
        else:
            # self.playButton.setIcon(
            #     self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))
            print('play')
    # def positionChanged(self, position):
    #     self.positionSlider.setValue(position)

    def durationChanged(self, duration):
        self.window.horizontalSlider.setRange(0, duration)

    def setPosition(self, position):
        self.mediaPlayer.setPosition(position)

    def handleError(self):
        self.playButton.setEnabled(False)
        self.errorLabel.setText("Error: " + self.mediaPlayer.errorString())

    
    # def refreshSampleVideo(self,sample_img):

    #     bytesPerLine = 3 *self.window.QLabel_sample_img.width() 
    #     img = cv2.resize(sample_img,(self.window.QLabel_sample_img.width(),self.window.QLabel_sample_img.height()))
    #     image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     qimg = QImage(image.data, self.window.QLabel_sample_img.width(), self.window.QLabel_sample_img.height(),bytesPerLine,QImage.Format_RGB888)
    #     qimg=QPixmap.fromImage(qimg)
        
    #     self.window.QLabel_sample_img.setPixmap(qimg)

    def get_cv_frame(self,img):
        self.actual_frame = img
        self.BlazePose_Thread.change_frame(img)
    
    def refresh_pose_world_landmarks(self,pose_world_landmarks):
        self.live_pose_world_landmarks= pose_world_landmarks
    def refresh_pose_connection(self,pose_connection):
        self.pose_connection= pose_connection
        # print(ax)
        
    def refreshScore(self,live_landmarks):
        # self.canvas.draw_idle()
        self.live_landmarks = live_landmarks
        self.pose_similarity_score = self.blazepose.compare_reverse(self.video_landmarks,live_landmarks)
        self.pose_action_score_list.append(self.pose_similarity_score)
       

        # self.window.label_score2.setText(str(self.pose_similarity_score))
        # self.window.label_score2.setFont(QFont('Times', 20))
        # self.window.label_score2.setFont(QFont('Times', 20))
    
    def refreshWindow(self, rtn_img):
        # print(img.shape)
        # w=int(img.shape[1])
        # h=int(img.shape[0])

        
        # rtn_img ,pose_label,self.live_landmarks,self.figure,self.ax= self.blazepose.detect_img(img)
        # rtn_img ,pose_label,self.live_landmarks= self.blazepose.detect_img(img)
        # self.canvas = FigureCanvas(self.figure)         #增加画布
        # self.canvas.draw()
        # # self.testwidget = QVideoWidget(self.window.centralwidget)
        # # self.testwidget.setGeometry(10, 10, 700, 700)
        # self.window.testvertalayout.addWidget(self.canvas)
        # self.window.label_poselabel.setText(pose_label)
        # self.window.label_poselabel.setFont(QFont('Times', 20))
        # print(rtn_img)
        self.bytesPerLine = 3 *self.window.QLabel_live_img.width()        
        img = cv2.resize(rtn_img,(self.window.QLabel_live_img.width(),self.window.QLabel_live_img.height()))
        # image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qimg = QImage(img.data, self.window.QLabel_live_img.width(), self.window.QLabel_live_img.height(),self.bytesPerLine,QImage.Format_RGB888)
        qimg = QPixmap.fromImage(qimg)
        self.window.QLabel_live_img.setPixmap(qimg)
       
        # self.plot_landmarks(self.pose_world_landmarks,self.pose_connection)
        
        # t = np.arange(0.0, 5.0, 0.01)
        # s = np.cos(2 * np.pi * t)
        # self.sub_window.set_mat_func(t,s)
        # self.sub_window.plot_tick()
       
        # self.count+=1

        # print(score)

        # self.canvas.draw()

        
    # def load_video(self):
    #     fileName1, filetype = QFileDialog.getOpenFileName(self,"选取文件","./video", "All Files (*);;Excel Files (*.xls)")  #设置文件扩展名过滤,注意用双分号间隔
    #     print('filename:',fileName1)
    #     self.select_video_path = fileName1
    #     self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(self.select_video_path)))


    #      # QMediaPlayer
       
    #     # Play
    #     # self.mediaPlayer.play()





    # def load_img(self,img):
    #     fileName1, filetype = QFileDialog.getOpenFileName(self,"选取文件","./", "All Files (*);;Excel Files (*.xls)")  #设置文件扩展名过滤,注意用双分号间隔
        
    #     # image = cv2.imread(fileName1)
    #     # rtn_img_img ,landmark= self.blazepose.detect_img(image)
    #     # bytesPerLine = 3 *self.window.QLabel_sample_img.width() 
    #     # w=int(rtn_img_img.shape[1])
    #     # h=int(rtn_img_img.shape[0])
    #     # rtn_img_img = cv2.resize(rtn_img_img,(self.window.QLabel_sample_img.width(),self.window.QLabel_sample_img.height()))
    #     # # image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     # rtn_img_img = QImage(rtn_img_img.data, self.window.QLabel_sample_img.width(), self.window.QLabel_sample_img.width(),bytesPerLine,QImage.Format_RGB888)
    #     # rtn_img_img=QPixmap.fromImage(rtn_img_img)
    #     # # self.window.QLabel_live_img.setPixmap(qimg)


    #     jpg = QtGui.QPixmap(fileName1).scaled(self.window.QLabel_sample_img.width(),self.window.QLabel_sample_img.height())
    #     self.window.QLabel_sample_img.setPixmap(jpg)
    #     # print(fileName1.split('.')[0])
    #     filename = os.path.basename(fileName1)
    #     self.window.label_samplelabel.setText(str(filename.split('.')[0]))
    #     self.window.label_samplelabel.setFont(QFont('Times', 20))
    #     self.sample_npy  = np.load(filename.split('.')[0]+'.npy')
    #     self.type = filename.split('.')[0]
        
    #     # print(self.sample_npy)
    


    def click_update(self):
        # self.canvas.draw_idle()
        self.live_anim = FuncAnimation(self.live_figure, self.live_plot_landmarks, interval=400)
        self.live_anim._start()
        # self.video_anim  = FuncAnimation(self.video_figure, self.live_plot_landmarks, interval=400)
        # self.video_anim._start()
    def _normalize_color(self,color):
        return tuple(v / 255. for v in color)



    def rotate_3d(self,vec,angle):
        # print('vec1',vec)
        # print('angle r',angle)
        if angle<0:
            angle = 360+ angle

        n_array =  [vec.x,vec.y,vec.z]

        rotation_degrees = angle
        rotation_radians = np.radians(rotation_degrees)
        rotation_axis = np.array([0, 1, 0])

        rotation_vector = rotation_radians * rotation_axis
        rotation = R.from_rotvec(rotation_vector)
        rotated_vec = rotation.apply(n_array)
        # print('angle real',angle)
        # print(rotated_vec)
        # vec.x = n_array[0]
        # vec.y = n_array[1]
        # vec.z = n_array[2]
        # print('vec',vec.x)
        # print('r',)
        return rotated_vec[0],rotated_vec[1],rotated_vec[2]
        

    def get_3d_angle(self,a1,a2):
        a1= np.array(a1)
        a2=np.array(a2)
        l_a1 = np.sqrt(a1.dot(a1))
        l_a2 = np.sqrt(a2.dot(a2))
        # print('la1 la2',l_a1,l_a2)
        dia = a1.dot(a2)
        # print('dia',dia)
        cos = dia/(l_a1*l_a2)
        # print('cos',cos)
        angle = np.arccos(cos)
        # print('angle',angle)
        real_angle = angle*180/np.pi
        # print('angle2',angle2)
        return  real_angle
        
    def live_plot_landmarks(self,landmark_list: landmark_pb2.NormalizedLandmarkList,
                    connections: Optional[List[Tuple[int, int]]] = None,
                    landmark_drawing_spec: DrawingSpec = DrawingSpec(
                        color=RED_COLOR, thickness=2),
                    connection_drawing_spec: DrawingSpec = DrawingSpec(
                        color=BLACK_COLOR, thickness=5),
                    elevation: int = 10,
                    azimuth: int = 10,
                    count_frame=0):
        """Plot the landmarks and the connections in matplotlib 3d.

        Args:
            landmark_list: A normalized landmark list proto message to be plotted.
            connections: A list of landmark index tuples that specifies how landmarks to
            be connected.
            landmark_drawing_spec: A DrawingSpec object that specifies the landmarks'
            drawing settings such as color and line thickness.
            connection_drawing_spec: A DrawingSpec object that specifies the
            connections' drawing settings such as color and line thickness.
            elevation: The elevation from which to view the plot.
            azimuth: the azimuth angle to rotate the plot.
        Raises:
            ValueError: If any connetions contain invalid landmark index.
        """
        # print('pose world landmarks',self.live_pose_world_landmarks)
        self.live_ax.clear()
        if not self.live_pose_world_landmarks:
            return
        if self.live_pose_world_landmarks.landmark[23].y-self.live_pose_world_landmarks.landmark[24].y>0:
            self.pose3d_direction =True
        vec_waist  = [self.live_pose_world_landmarks.landmark[23].x-self.live_pose_world_landmarks.landmark[24].x,self.live_pose_world_landmarks.landmark[23].y-self.live_pose_world_landmarks.landmark[24].y,self.live_pose_world_landmarks.landmark[23].z-self.live_pose_world_landmarks.landmark[24].z]
        # print(new_a1)
        self.pose3d_waist_angle = self.get_3d_angle(vec_waist,[1,0,0])
        # print('self.pose3d_waist_angle',self.pose3d_waist_angle)
        if  not self.pose3d_direction and self.pose3d_direction!='':
            pass
       
        else:
            if self.pose3d_waist_angle!='':
                self.pose3d_waist_angle = -self.pose3d_waist_angle 
        # self.fig = plt.figure()
        # self.ax = plt.axes(projection='3d')
        self.live_ax.view_init(elev=elevation, azim=azimuth)
        self.live_ax.set_zlim(-0.5,0.5)
        self.live_ax.set_ylim(-1,1)
        self.live_ax.set_xlim(-1,1)
        plotted_landmarks = {}
        for idx, landmark in enumerate(self.live_pose_world_landmarks.landmark):
            if ((landmark.HasField('visibility') and
                landmark.visibility < _VISIBILITY_THRESHOLD) or
                (landmark.HasField('presence') and
                landmark.presence < _PRESENCE_THRESHOLD)):
                continue
            # print('landmark',landmark)
            x_p,y_p,z_p = self.rotate_3d(landmark,self.pose3d_waist_angle)
            # print('change',x_p,y_p,z_p)
            # x_p,y_p,z_p = landmark.x,landmark.y,landmark.z
            # print('change',x_p,y_p,z_p)
            self.live_ax.scatter3D(
                xs=[-z_p],
                ys=[x_p],
                zs=[-y_p],
                color=self._normalize_color(landmark_drawing_spec.color[::-1]),
                linewidth=landmark_drawing_spec.thickness)
            plotted_landmarks[idx] = (-z_p, x_p, -y_p)
        if self.pose_connection:
            num_landmarks = len(self.live_pose_world_landmarks.landmark)
            # Draws the connections if the start and end landmarks are both visible.
            for connection in self.pose_connection:
                start_idx = connection[0]
                end_idx = connection[1]
                if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
                    raise ValueError(f'Landmark index is out of range. Invalid connection '
                                    f'from landmark #{start_idx} to landmark #{end_idx}.')
                if start_idx in plotted_landmarks and end_idx in plotted_landmarks:
                    landmark_pair = [
                        plotted_landmarks[start_idx], plotted_landmarks[end_idx]
                    ]
                    self.live_ax.plot3D(
                        xs=[landmark_pair[0][0], landmark_pair[1][0]],
                        ys=[landmark_pair[0][1], landmark_pair[1][1]],
                        zs=[landmark_pair[0][2], landmark_pair[1][2]],
                        color=self._normalize_color(connection_drawing_spec.color[::-1]),
                        linewidth=connection_drawing_spec.thickness)
        # self.canvas.draw()
        
        # self.canvas.flush_events() 


        
    def video_plot_landmarks(self,landmark_list: landmark_pb2.NormalizedLandmarkList,
                    connections: Optional[List[Tuple[int, int]]] = None,
                    landmark_drawing_spec: DrawingSpec = DrawingSpec(
                        color=RED_COLOR, thickness=2),
                    connection_drawing_spec: DrawingSpec = DrawingSpec(
                        color=BLACK_COLOR, thickness=5),
                    elevation: int = 10,
                    azimuth: int = 10,
                    count_frame=0):
        """Plot the landmarks and the connections in matplotlib 3d.

        Args:
            landmark_list: A normalized landmark list proto message to be plotted.
            connections: A list of landmark index tuples that specifies how landmarks to
            be connected.
            landmark_drawing_spec: A DrawingSpec object that specifies the landmarks'
            drawing settings such as color and line thickness.
            connection_drawing_spec: A DrawingSpec object that specifies the
            connections' drawing settings such as color and line thickness.
            elevation: The elevation from which to view the plot.
            azimuth: the azimuth angle to rotate the plot.
        Raises:
            ValueError: If any connetions contain invalid landmark index.
        """
        # print('pose world landmarks',self.video_pose_world_landmarks)
        self.video_ax.clear()
        if not self.video_pose_world_landmarks:
            return
        # self.fig = plt.figure()
        # self.ax = plt.axes(projection='3d')
        self.video_ax.view_init(elev=elevation, azim=azimuth)
        self.video_ax.set_zlim(-0.5,0.5)
        self.video_ax.set_ylim(-1,1)
        self.video_ax.set_xlim(-1,1)
        plotted_landmarks = {}
        for idx, landmark in enumerate(self.video_pose_world_landmarks.landmark):
            if ((landmark.HasField('visibility') and
                landmark.visibility < _VISIBILITY_THRESHOLD) or
                (landmark.HasField('presence') and
                landmark.presence < _PRESENCE_THRESHOLD)):
                continue
            # print('landmark.z',landmark.z)
            self.video_ax.scatter3D(
                xs=[-landmark.z],
                ys=[landmark.x],
                zs=[-landmark.y],
                color=self._normalize_color(landmark_drawing_spec.color[::-1]),
                linewidth=landmark_drawing_spec.thickness)
            plotted_landmarks[idx] = (-landmark.z, landmark.x, -landmark.y)
        if self.pose_connection:
            num_landmarks = len(self.video_pose_world_landmarks.landmark)
            # Draws the connections if the start and end landmarks are both visible.
            for connection in self.pose_connection:
                start_idx = connection[0]
                end_idx = connection[1]
                if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
                    raise ValueError(f'Landmark index is out of range. Invalid connection '
                                    f'from landmark #{start_idx} to landmark #{end_idx}.')
                if start_idx in plotted_landmarks and end_idx in plotted_landmarks:
                    landmark_pair = [
                        plotted_landmarks[start_idx], plotted_landmarks[end_idx]
                    ]
                    self.video_ax.plot3D(
                        xs=[landmark_pair[0][0], landmark_pair[1][0]],
                        ys=[landmark_pair[0][1], landmark_pair[1][1]],
                        zs=[landmark_pair[0][2], landmark_pair[1][2]],
                        color=self._normalize_color(connection_drawing_spec.color[::-1]),
                        linewidth=connection_drawing_spec.thickness)
        # self.canvas.draw()
        
        # self.canvas.flush_events() 
# 

    def video_on(self):
        self.CamThread.video_start()
        self.BlazePose_Thread.video_start()
        # self.status = False
        self.mytimer.start(5000) 



if __name__ == '__main__':
    app=QtWidgets.QApplication(sys.argv)
    window = Ui()
    window.show()
    sys.exit(app.exec_())