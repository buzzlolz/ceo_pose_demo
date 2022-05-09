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
                    rtn_img ,pose_label,live_landmarks=self.blazepose.detect_img(self.frame)
                    
                    self.keypoint_frame_sig.emit(rtn_img)
                    self.live_landmarks_sig.emit(live_landmarks)
            
    def video_start(self):
        
        self.video_status = True
        print('blazepoe change stus',self.video_status)
    def change_frame(self,img):
        self.frame = img


class Cam(QtCore.QThread):
    frame_sig = pyqtSignal(object)
    
    def __init__(self):
        super(Cam,self).__init__()
        self.cap  = cv2.VideoCapture(0)
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
        self.window = uic.loadUi('compare_pose2.ui',self)
        self.window.pushButton_load_video.clicked.connect(self.openFile)
        self.window.pushButton_start_video.clicked.connect(self.play)
        self.select_video_path = ''
        self.sample_video_action_count = 0
        self.frame_index = 0
        self.pose_similarity_score = 0
        self.grade = 'None'
        self.pose_action_score_list = []
        self.action_index = 1
        self.action_time_list=[]
        self.keypoint_npy_array=[]
        self.keypoint_sample_frame=[]
        self.df_csv=[]
        
        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        

        # self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile('/home/n200/drc/ceo_demo_blazepose/video/4min_video.mp4')))

        # Set widget
        self.videoWidget = QVideoWidget(self.window.centralwidget)
        self.videoWidget.setGeometry(10, 10, 300, 300)
        # self.setCentralWidget(self.videoWidget)
        self.mediaPlayer.setVideoOutput(self.videoWidget)


        # self.mediaPlayer.setVideoOutput(videoWidget)
        self.mediaPlayer.stateChanged.connect(self.mediaStateChanged)
        # self.mediaPlayer.positionChanged.connect(self.positionChanged)
        # self.mediaPlayer.durationChanged.connect(self.durationChanged)
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
        self.BlazePose_Thread.start()
        # self.Read_Video = Read_Video()
        # self.Read_Video.changePixmap.connect(self.refreshSampleVideo)
        # self.Read_Video.start()
        self.mediaPlayer.setNotifyInterval(1)
        self.mediaPlayer.positionChanged.connect(self.update_position)

       

       


        self.pushButton_camstart.clicked.connect(self.video_on)
        self.sample_npy=[]
        self.live_landmarks=[]

        
        self.window.label_score.setFont(QFont('Times', 20))
        self.window.label_grade.setFont(QFont('Times', 20))
        self.window.label_score2.setFont(QFont('Times', 20))
        self.window.label_grade2.setFont(QFont('Times', 20))
        # self.fig = Figure(figsize=(10, 10), dpi=100)
        # self.ax = self.fig.add_subplot(111)
  
        # self.count = 0
        # self.CamThread.frame_sig.connect()
        # layout = QtWidgets.QVBoxLayout(self.window)
        # self.figure = plt.figure("test") 
        # self.ax = plt.axes(projection='3d')  # 三维坐标轴

        # self.canvas = FigureCanvas(self.figure)         #增加画布
        # # self.testwidget = QVideoWidget(self.window.centralwidget)
        # # self.testwidget.setGeometry(10, 10, 700, 700)
        # self.window.testvertalayout.addWidget(self.canvas)      
    def showbox(self): 
            self.ax.clear() 
            self.ax.bar3d(0,0,0,self.spinBox_4.value(),self.spinBox_5.value(),self.spinBox_6.value(), color="green",zsort='average',edgecolor='white',linewidth=0.5,alpha=1) 
            self.ax.bar3d(0,0,0,self.spinBox.value(),self.spinBox_2.value(),self.spinBox_3.value(), color="green",zsort='average',edgecolor='red',linewidth=0.5,alpha=0) 

            self.canvas.draw()

    def update_position(self, position):

        # if len(self.action_time_list)<=0:
        #     self.action_index+=1
        #     df_index = 'a'+str(self.action_index)+'_start'
        #     self.action_time_list = self.df_csv['a1_start'].values

        

        self.frame_index  = round(self.mediaPlayer.position()/33.3)
        # print(self.action_time_list)
        if len(self.keypoint_npy_array)!=0:
            if self.frame_index<len(self.keypoint_npy_array):
                self.keypoint_sample_frame = self.keypoint_npy_array[self.frame_index]
            # print( self.keypoint_sample_frame)

        
        if self.frame_index  in self.action_time_list:

            #count avg grade in one action
            if len(self.pose_action_score_list)>0:
                avg_score = sum(self.pose_action_score_list)/len(self.pose_action_score_list)
                if avg_score>0.80:
                    self.grade ='A'
                elif avg_score>=0.70:
                    self.grade = 'B'
                else:
                    self.grade = 'C'
                self.window.label_grade2.setText(self.grade)
                
            

            self.sample_video_action_count+=1
            self.window.label_frame_count.setFont(QFont('Times', 20))
            # self.window.label_frame_count.setText(str(self.sample_video_action_count))
            # print('in self.action_time_list',frame_in)
            self.action_time_list  = np.delete(self.action_time_list,0)
            if len(self.action_time_list) == 0:
                self.sample_video_action_count= 0
                self.action_index += 1
                self.action_time_list = self.df_csv['a'+str(self.action_index)+'_start'].values
                self.action_time_list=self.action_time_list[~np.isnan(self.action_time_list)]
        
        self.pose_action_score_list.clear()
        # self.action_time_list

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
        self.keypoint_npy_array = np.load(npy_path,allow_pickle=True)
        

    def play(self):
        if self.mediaPlayer.state() == QtMultimedia.QMediaPlayer.PlayingState:
            self.mediaPlayer.pause()
            # print(self.mediaPlayer.position())
           

        else:
            
            self.mediaPlayer.play()
          
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

    # def durationChanged(self, duration):
    #     self.positionSlider.setRange(0, duration)

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
    def refreshScore(self,live_landmarks):
        self.pose_similarity_score = self.blazepose.compare_reverse(self.keypoint_sample_frame,live_landmarks)
        self.pose_action_score_list.append(self.pose_similarity_score)
       

        self.window.label_score2.setText(str(self.pose_similarity_score))
        self.window.label_score2.setFont(QFont('Times', 20))
        self.window.label_score2.setFont(QFont('Times', 20))
    
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
        # print( self.keypoint_sample_frame)
        
        
        # t = np.arange(0.0, 5.0, 0.01)
        # s = np.cos(2 * np.pi * t)
        # self.sub_window.set_mat_func(t,s)
        # self.sub_window.plot_tick()
       
        # self.count+=1

        # print(score)

        
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

    

    def video_on(self):
        self.CamThread.video_start()
        self.BlazePose_Thread.video_start()



if __name__ == '__main__':
    app=QtWidgets.QApplication(sys.argv)
    window = Ui()
    window.show()
    sys.exit(app.exec_())