


import sys
sys.path.append('../../../..')
from libgex.libgex.libgx11 import Hand

import cv2
import numpy as np
import pyrealsense2 as rs
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QHBoxLayout
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer


import time
import os



class RealSenseApp(QWidget):
    def __init__(self):
        super().__init__()


        self.joint_sign = np.array([1, 1, 1,
                      1, -1, -1, -1,
                      -1, -1, 1, 1])

        self.init_hand()

        
        self.initUI()
        self.init_realsense()
        self.init_timer()

        self.is_recording = False
        self.rgb_video_writer = None
        self.depth_video_writer = None
        self.hand_data = []
        self.hand_timer = QTimer(self)
        self.hand_timer.timeout.connect(self.sample_hand_position)
        self.frame_count = 0
    
        self.set_root_dir()
    
    def set_root_dir(self):
        self.root_dir = '../data/result/airpods/'


    def initUI(self):
        self.setWindowTitle('RealSense 相机显示')
        self.setGeometry(100, 100, 1200, 700)

        # 创建 RGB 和深度图像显示标签
        self.rgb_label = QLabel(self)
        self.depth_label = QLabel(self)

        # 图像显示布局
        image_layout = QHBoxLayout()
        image_layout.addWidget(self.rgb_label)
        image_layout.addWidget(self.depth_label)

        # 录制控制按钮
        self.start_record_button = QPushButton('开始录制', self)
        self.start_record_button.clicked.connect(self.start_recording)
        self.stop_record_button = QPushButton('停止录制', self)
        self.stop_record_button.clicked.connect(self.stop_recording)
        self.stop_record_button.setEnabled(False)


        # 添加返回初始位置按钮
        self.back_to_initial_button = QPushButton('返回初始', self)
        self.back_to_initial_button.clicked.connect(self.back_to_initial)
        # self.back_to_initial_button.setGeometry(50, 340, 150, 50)


        # 按钮布局
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.start_record_button)
        button_layout.addWidget(self.stop_record_button)
        button_layout.addWidget(self.back_to_initial_button)

        # 主布局
        main_layout = QVBoxLayout()
        main_layout.addLayout(image_layout)
        main_layout.addLayout(button_layout)
        self.setLayout(main_layout)

    def back_to_initial(self):
        print('back to initial')
        joint_positions = np.array([
            0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
        ])
        self.apply_action(joint_positions)
        time.sleep(2)

    def init_realsense(self):
        # 配置 RealSense 管道，同时获取 RGB 和深度流
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        # 启动管道
        self.pipeline.start(config)

        # 创建对齐对象，将深度帧对齐到 RGB 帧
        self.align = rs.align(rs.stream.color)

    def init_hand(self):
        self.hand = Hand(port='/dev/ttyACM0')  # COM* for Windows, ttyACM* or ttyUSB* for Linux
        self.hand.connect(goal_pwm=600)  # goal_pwm changes the speed, max 855
        self.action = np.zeros(11)
        # self.hand.home()  # home the hand

        self.back_to_initial()
        
        
        time.sleep(2)
        
        # exit()

    def init_timer(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(10)  # 大约 30 FPS

    def update_frame(self):
        # 获取一帧数据
        frames = self.pipeline.wait_for_frames()
        # 对齐深度帧到 RGB 帧
        aligned_frames = self.align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            return

        # 将 RGB 帧数据转换为 numpy 数组
        color_image = np.asanyarray(color_frame.get_data())
        # 将深度帧数据转换为 numpy 数组
        depth_image = np.asanyarray(depth_frame.get_data())
        # 将深度图像转换为伪彩色图像
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        
        self.color_image = color_image

        # 录制图像
        if self.is_recording:
            self.rgb_video_writer.write(color_image)
            self.depth_video_writer.write(depth_colormap)
            self.frame_count += 1

        # 显示 RGB 图像
        self.show_image(self.rgb_label, color_image)
        # 显示深度图像
        self.show_image(self.depth_label, depth_colormap)


    def show_image(self, label, image):
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_img = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        label.setPixmap(QPixmap.fromImage(q_img))

    def sample_hand_position(self):
        
        hand_pos = self.hand.getj()
        hand_cur = self.hand.get_current()
        
        hand_data_frame = [self.action, hand_pos, hand_cur]
        
        self.hand_data.append(hand_data_frame)
        # print(time.time())
        
        self.step()

    def start_recording(self):
        self.is_recording = True
        self.start_record_button.setEnabled(False)
        self.stop_record_button.setEnabled(True)
        self.frame_count = 0
        self.hand_data = []
        
        exp_dir_name = f"exp_{time.strftime('%Y%m%d_%H%M%S')}"  # 生成包含时间的实验目录名
        self.exp_dir = os.path.join(self.root_dir, exp_dir_name)
        os.makedirs(self.exp_dir, exist_ok=True)  # 创建实验目录
        print(f"实验目录已创建: {self.exp_dir}")

        # 初始化 RGB 视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.rgb_video_writer = cv2.VideoWriter(f'{self.exp_dir}/rgb_output.mp4', fourcc, 10.0, (640, 480))
        # 初始化深度视频写入器
        self.depth_video_writer = cv2.VideoWriter(f'{self.exp_dir}/depth_output.mp4', fourcc, 10.0, (640, 480))

        # 启动手部数据采样定时器
        self.hand_timer.start(100)  # 0.1 秒间隔
        
        self.sample_index = 0

    def stop_recording(self):
        self.is_recording = False
        self.start_record_button.setEnabled(True)
        self.stop_record_button.setEnabled(False)
        # 释放视频写入器
        if self.rgb_video_writer:
            self.rgb_video_writer.release()
        if self.depth_video_writer:
            self.depth_video_writer.release()
        # 停止手部数据采样定时器
        self.hand_timer.stop()
        # 保存手部数据
        np.save(f'{self.exp_dir}/hand_data.npy', np.array(self.hand_data))
        print('exp dir: ', self.exp_dir)
        print(f"录制完成，共 {self.frame_count} 帧，手部数据已保存到 hand_data.npy")
        print('hand traj len', len(self.hand_data))
        
        self.back_to_initial()
        

    def closeEvent(self, event):
        # 停止 RealSense 管道
        self.pipeline.stop()
        # 释放视频写入器
        if self.rgb_video_writer:
            self.rgb_video_writer.release()
        if self.depth_video_writer:
            self.depth_video_writer.release()
        # 停止手部数据采样定时器
        self.hand_timer.stop()
        event.accept()

    def apply_action(self, action):
        
        # self.joint_sign = np.array([1, 1, 1,
        #               1, -1, -1, -1,
        #               -1, -1, 1, 1])
        self.action = action[:11]
        self.hand.setj(self.joint_sign * self.action)


    def step(self):
        pass


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = RealSenseApp()
    ex.show()
    sys.exit(app.exec_())
