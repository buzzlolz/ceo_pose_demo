from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *
import sys
from ufrmVideo import Ui_MainWindow

#--- 時間換算 
def hhmmss(ms):
    # s = 1000
    # m = 60000
    # h = 360000
    s = round(ms / 1000)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return ("%02d:%02d:%02d" % (h, m, s)) if h else ("%02d:%02d" % (m, s))
    
#--- 定義 播放窗 
class ViewerWindow(QMainWindow):
    # state for view  
    state = pyqtSignal(bool)
    # sign for pause 
    sign = pyqtSignal(bool)
    
    def closeEvent(self, e):
        # emit to update the viewer button 
        self.state.emit()
    
    def mousePressEvent(self, e):
        if e.button() == Qt.RightButton:
            print('Rt click pause')
            self.sign.emit(True)
                        
    def keyPressEvent(self,e ):
        if e.key() == Qt.Key_Escape:
            print('Esc pressed')
            self.state.emit()
            
#--- playlist
class PlaylistModel(QAbstractListModel):
    def __init__(self, playlist, *args, **kwargs):
        super(PlaylistModel, self).__init__(*args, **kwargs)
        self.playlist = playlist

    def data(self, index, role):
        if role == Qt.DisplayRole:
            media = self.playlist.media(index.row())
            return media.canonicalUrl().fileName()

    def rowCount(self, index):
        return self.playlist.mediaCount()

#--- 定義 控制窗  
class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)
        # 設定global ui 就是畫面 ufrmVideo  
        global ui
        ui = Ui_MainWindow()
        ui.setupUi(self)
    
        self.player = QMediaPlayer()
        self.player.error.connect(self.erroralert)
        self.player.play()
                 
        # Setup the playlist
        self.playlist = QMediaPlaylist()
        self.player.setPlaylist(self.playlist)
  
        # Add viewer for video playback, separate floating window.
        self.viewer = ViewerWindow(self)
        self.viewer.setWindowFlags(self.viewer.windowFlags() | Qt.WindowStaysOnTopHint)
        self.viewer.setMinimumSize(QSize(480, 360))
        videoWidget = QVideoWidget()
        self.viewer.setCentralWidget(videoWidget)
        self.player.setVideoOutput(videoWidget)

        # menubar 
        ui.actionOpen_Video_File.triggered.connect(self.open_file)
        ui.actionExit.triggered.connect(self.close_event)
       
        #--- 定義 button onClick 啟動 xx 事件----
        ui.playButton.pressed.connect(self.player.play)
        ui.pauseButton.pressed.connect(self.player.pause)
        ui.stopButton.pressed.connect(self.player.stop)
        ui.volumeSlider.valueChanged.connect(self.player.setVolume)
        
        # viewButton 
        ui.viewButton.toggled.connect(self.toggle_viewer)
        self.viewer.state.connect(ui.viewButton.setChecked)
        # under constrution ....
        ui.pauseButton.toggled.connect(self.toggle_pause)    
        self.viewer.sign.connect(ui.pauseButton.setChecked)
             
        ui.prevButton.pressed.connect(self.playlist.previous)
        ui.nextButton.pressed.connect(self.playlist.next)
        
        # playList
        self.model = PlaylistModel(self.playlist)
        ui.playlistView.setModel(self.model)
        self.playlist.currentIndexChanged.connect(self.playlist_position_changed)
        selection_model = ui.playlistView.selectionModel()
        selection_model.selectionChanged.connect(self.playlist_selection_changed)
        
        # 進度改變
        self.player.durationChanged.connect(self.update_duration)
        self.player.positionChanged.connect(self.update_position)
        ui.timeSlider.valueChanged.connect(self.player.setPosition)
        
        self.setAcceptDrops(True)
        self.show()
    
    #--- 事件執行部
    def close_event(self):
        self.close()
   
    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,'Open file','',
            'mp4(*.mp4);;Movie(*.mov);;mp3 (*.mp3);;All files(*.*)',)
        if path:
            self.playlist.addMedia(QMediaContent(QUrl.fromLocalFile(path)))
        self.model.layoutChanged.emit()

    def update_duration(self, duration):
        ui.timeSlider.setMaximum(duration)

        if duration >= 0:
            ui.totalTime.setText(hhmmss(duration))

    def update_position(self, position):
        if position >= 0:
            #print('position: ',position)
            ui.currentTime.setText(hhmmss(position))
            
        # Disable the events to prevent updating 
        # triggering a setPosition event (can cause stuttering).
        ui.timeSlider.blockSignals(True)
        ui.timeSlider.setValue(position)
        ui.timeSlider.blockSignals(False)

    def playlist_selection_changed(self, ix):
        # We receive a QItemSelection from selectionChanged.
        i = ix.indexes()[0].row()
        self.playlist.setCurrentIndex(i)

    def playlist_position_changed(self, i):
        if i > -1:
            ix = self.model.index(i)
            ui.playlistView.setCurrentIndex(ix)

    def toggle_viewer(self, state):
        if state:
            self.viewer.show()
        else:
            self.viewer.hide()
            
    def toggle_pause(self, sign):
        print(self.player.state())
        if sign:
            self.player.pause()
        else:
            self.player.play()
   
    def erroralert(self, *args):
        print(args)
    

if __name__ == '__main__':
     app = QApplication([])
     app.setApplicationName('VideoPlayer')
     window = MainWindow()
     window.setWindowTitle('Video Player')
     window.show()
     sys.exit(app.exec_())