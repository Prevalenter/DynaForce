import sys
sys.path.append('../../..')

from libgex.libgex.libgx11 import Hand
import time
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QFileDialog, QLabel, QLineEdit
from PyQt5.QtCore import QTimer

class RecorderApp(QWidget):
    def __init__(self, save_path='test_traj.npy'):
        super().__init__()
        self.initUI()
        self.hand = Hand(port='/dev/ttyACM0')  # COM* for Windows, ttyACM* or ttyUSB* for Linux
        self.hand.connect(goal_pwm=20)  # goal_pwm changes the speed, max 855
        self.hand.home()  # home the hand
        self.is_recording = False
        self.is_replaying = False
        self.save_path = save_path
        self.joint_idx = 2
        self.exp_idx = 4
        self.interal = 0.02 # 0.1
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.record_step)
        self.replay_timer = QTimer(self)
        self.replay_timer.timeout.connect(self.replay_step)
        self.traj = []
        self.replay_traj = []
        self.replay_index = 0

    def initUI(self):
        self.record_button = QPushButton('开始记录', self)
        self.record_button.clicked.connect(self.toggle_recording)
        self.record_button.setGeometry(50, 50, 150, 50)

        self.select_button = QPushButton('选择轨迹文件', self)
        self.select_button.clicked.connect(self.select_traj_file)
        self.select_button.setGeometry(50, 120, 150, 50)

        self.replay_button = QPushButton('开始回放', self)
        self.replay_button.clicked.connect(self.toggle_replay)
        self.replay_button.setGeometry(50, 190, 150, 50)

        # 添加用于显示文件路径的标签
        self.file_path_label = QLabel(self)
        self.file_path_label.setGeometry(50, 260, 200, 30)
        self.file_path_label.setText("未选择文件")

        # 添加 PWM 输入框
        self.pwm_input = QLineEdit(self)
        self.pwm_input.setGeometry(50, 300, 100, 30)
        self.pwm_input.setPlaceholderText("输入 PWM 值")

        # 添加更改 PWM 按钮
        self.change_pwm_button = QPushButton('更改 PWM', self)
        self.change_pwm_button.clicked.connect(self.change_pwm)
        self.change_pwm_button.setGeometry(160, 300, 100, 30)

        self.setGeometry(300, 300, 300, 380)
        self.setWindowTitle('记录器')
        self.show()

    def toggle_recording(self):
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        self.traj = []
        self.is_recording = True
        self.record_button.setText('停止记录')
        self.start_time = time.time()
        self.timer.start(int(self.interal * 1000))

    def stop_recording(self):
        self.is_recording = False
        self.record_button.setText('开始记录')
        self.timer.stop()
        traj = np.array(self.traj)
        np.save(self.save_path, traj)
        self.traj = []
        print(f"Done, using:: {time.time() - self.start_time} sec")

    def record_step(self):
        traj_i = self.hand.getj()
        print(len(self.traj), traj_i)
        self.traj.append(traj_i)

    def select_traj_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择轨迹文件", "", "Numpy Files (*.npy)")
        if file_path:
            self.replay_traj = np.load(file_path)
            print(f"已选择轨迹文件: {file_path}")
            # 更新标签文本显示所选文件路径
            self.file_path_label.setText(f"已选文件: {file_path}")
        else:
            self.file_path_label.setText("未选择文件")

    def toggle_replay(self):
        if self.is_replaying:
            self.stop_replay()
        else:
            if self.replay_traj is not None:
                self.start_replay()
            else:
                print("请先选择轨迹文件")

    def start_replay(self):
        self.is_replaying = True
        self.replay_button.setText('停止回放')
        self.replay_index = 0
        self.replay_timer.start(int(self.interal * 1000))

    def stop_replay(self):
        self.is_replaying = False
        self.replay_button.setText('开始回放')
        self.replay_timer.stop()

    def replay_step(self):
        if self.replay_index < len(self.replay_traj):
            joint_positions = self.replay_traj[self.replay_index]
            # 假设 Hand 类有 setj 方法用于设置关节位置
            self.hand.setj(joint_positions)
            print(f"回放步骤: {self.replay_index}, 位置: {joint_positions}")
            self.replay_index += 1
        else:
            self.stop_replay()
            print("回放完成")

    def change_pwm(self):
        pwm_text = self.pwm_input.text()
        try:
            pwm_value = int(pwm_text)
            if 0 <= pwm_value <= 855:
                self.hand.init_config(goal_pwm=pwm_value)
                print(f"PWM 值已更新为: {pwm_value}")
            else:
                print("PWM 值需在 0 到 855 之间")
        except ValueError:
            print("请输入有效的整数")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = RecorderApp()
    sys.exit(app.exec_())  # PyQt5 使用 exec_() 方法
