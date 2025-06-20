import sys
import rospy
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QSlider, QLabel, QWidget
from PyQt5.QtCore import Qt
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped
import tf
import numpy as np


import rospy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped
import tf
import numpy as np

import sys
sys.path.append('../../../')
from libgx.libgx.libgx11 import Hand


class JointControl(QMainWindow):
    def __init__(self):
        super().__init__()

        self.hand = Hand(port='/dev/ttyACM0')  # COM* for windows
        self.hand.connect()
        print('the gx11pro is ready!')

        self.joint_sign = [1, 1, 1,
                      1, -1, -1, -1,
                      -1, -1, 1, 1]


        rospy.init_node('joint_control_node', anonymous=True)
        self.joint_pub = rospy.Publisher('/hand/joint_states', JointState, queue_size=10)
        self.tf_broadcaster = tf.TransformBroadcaster()
        self.initUI()
        self.start_ros()

        # Create a publisher for the joint states
        self.joint_state_pub = rospy.Publisher('joint_states', JointState, queue_size=10)


    def initUI(self):
        self.setWindowTitle('Robot Joint Control')

        centralWidget = QWidget(self)
        self.setCentralWidget(centralWidget)
        layout = QVBoxLayout()
        centralWidget.setLayout(layout)

        self.sliders = {}
        self.labels = {}

        for i in range(11):
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(-180)
            slider.setMaximum(180)
            slider.setValue(0)
            slider.valueChanged.connect(self.update_joints)

            self.sliders[f'Joint_{i + 1}'] = slider

            layout.addWidget(QLabel(f'Joint {i + 1}'))
            layout.addWidget(slider)

        self.update_joints()

    def update_joints(self):
        joint_state = JointState()
        joint_state.header.stamp = rospy.Time.now()
        joint_state.name = [f'joint{i + 1}' for i in range(11)]
        joint_state.position = [self.sliders[f'Joint_{i + 1}'].value()*(np.pi/180) for i in range(11)]

        print('joint_state: ', joint_state.position)

        self.joint_pub.publish(joint_state)

        for joint_idx in range(11):
            self.hand.motors[joint_idx].set_pos( self.joint_sign[joint_idx]*joint_state.position[joint_idx]*(180/np.pi) )


        transform = TransformStamped()

        # Set the header frame ID and timestamp
        transform.header.stamp = rospy.Time.now()
        transform.header.frame_id = "map"  # Parent frame
        transform.child_frame_id = "hand_/base_link"  # Child frame
        #
        # # Set the translation
        transform.transform.translation.x = 0.08 #+ anlge
        transform.transform.translation.y = 0.05
        transform.transform.translation.z = 0
        #
        # # Set the rotation
        quat = tf.transformations.quaternion_from_euler(-np.pi/9, 0, np.pi)  # Roll, Pitch, Yaw
        transform.transform.rotation.x = quat[0]
        transform.transform.rotation.y = quat[1]
        transform.transform.rotation.z = quat[2]
        transform.transform.rotation.w = quat[3]
        # Publish the transform
        self.tf_broadcaster.sendTransformMessage(transform)


    def start_ros(self):
        self.timer = self.startTimer(100)  # 100 ms
        self.update_joints()

    def timerEvent(self, event):
        self.update_joints()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = JointControl()
    ex.show()
    sys.exit(app.exec_())
