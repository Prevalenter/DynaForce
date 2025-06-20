from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QSlider, QLabel, QWidget, QPushButton, QFileDialog
from PyQt5.QtCore import Qt
import numpy as np
from geometry_msgs.msg import TransformStamped
import rospy
import tf
import sympy
from functools import partial

from sensor_msgs.msg import JointState

symbol_q_list = sympy.symbols('q1 q2 q3 q4 q5 q6 q7')
initial_zero = np.array([0, 0, 0])

class Finger:
    def __init__(self, robotdef):
        self.sym_kine = robotdef.geo.T
        self.dof = robotdef.dof
  
    def forward_kine(self, q):
        rst = []
        for i in range(self.dof):
            rst_i = self.sym_kine[i].subs({
                symbol_q_list[i]:q[i] for i in range(self.dof)
            })
            rst.append(np.array(rst_i))
        return np.array(rst)

class Hand:
    def __init__(self, robotdef_list):
        self.finger_dof = [rbt.dof for rbt in robotdef_list]
        self.dof = sum(self.finger_dof)
        self.finger_list = [Finger(rbt) for rbt in robotdef_list]

    def forward_kine(self, q):
        rst = []
        cur_idx = 0
        for idx, finger in enumerate(self.finger_list):
            
            cur_dof = self.finger_dof[idx]
            print(cur_idx, cur_idx+cur_dof)
            rst_i = self.finger_list[idx].forward_kine(q[cur_idx:cur_idx+cur_dof])

            rst.append(rst_i)
            cur_idx += cur_dof
        # breakpoint()
        return np.concatenate(rst)

class HandVisual(QMainWindow):
    def __init__(self, robotdef_list):
        super().__init__()
        # self.finger_dof = [rbt.dof for rbt in robotdef_list]
        # self.dof = sum(self.finger_dof)

        # self.finger_list = [Finger(rbt) for rbt in robotdef_list]
        self.hand = Hand(robotdef_list)
        self.ros_init()
        self.initUI()

    def ros_init(self):
        rospy.init_node('tf_transform_control')
        self.tf_broadcaster = tf.TransformBroadcaster()
        self.joint_pub = rospy.Publisher('/hand/joint_states', JointState, queue_size=10)

    def initUI(self):
        self.setWindowTitle('TF Transform Control')

        centralWidget = QWidget(self)
        self.setCentralWidget(centralWidget)
        layout = QVBoxLayout()
        centralWidget.setLayout(layout)

        self.sliders = []
        self.labels = []


        # for finger in self.sym_kine:
        for i in range(self.hand.dof):
            # slider with value tickes

            slider = QSlider(Qt.Horizontal)
            slider.setTickPosition(QSlider.TicksBelow)
            # 设置刻度间隔
            slider.setTickInterval(20)
            slider.setMinimum(-100)
            slider.setMaximum(100)
            slider.setMinimum(-100)
            slider.setMaximum(100)
            self.sliders.append(slider)
            label = f'Joint {i+1}'
            layout.addWidget(QLabel(label))
            layout.addWidget(slider)
            slider.valueChanged.connect(self.update_transform)
        self.save_button = QPushButton('Save Transform', self)
        self.save_button.clicked.connect(self.save_transform)
        layout.addWidget(self.save_button)
        self.update_transform()

        self.load_transform()
  
    def load_transform(self):
        pass

    def save_transform(self):
        pass

    def update_transform(self):
        # Extract values from sliders
        slider_values= [s.value() / 40.0 for s in self.sliders]
        print(slider_values)
  
        rst = self.hand.forward_kine(slider_values)
        print('rst: ', rst.shape)
  

        for i in range(11):
            pos_i = rst[i, :3, -1]

            quat_i = tf.transformations.quaternion_from_matrix(rst[i])
   
            if i<3:
                self.set_joint_tf(f'joint_{i+1}', pos_i, quat=quat_i)
            else:
                self.set_joint_tf(f'joint_{i+1}', pos_i, quat=quat_i, frame_id='hand_/base_link1')
  
        # update the rviz joint state
        joint_state = JointState()
        joint_state.header.stamp = rospy.Time.now()
        joint_state.name = [f'joint{i + 1}' for i in range(11)]
        # joint_state.position = [slider_values[i] if i<6 else 0 for i in range(11)]
        joint_state.position = [slider_values[i] if i<11 else 0 for i in range(11)]

        print('joint_state: ', joint_state.position)

        self.joint_pub.publish(joint_state)


        transform = TransformStamped()

        # Set the header frame ID and timestamp
        transform.header.stamp = rospy.Time.now()
        transform.header.frame_id = "map"  # Parent frame
        transform.child_frame_id = "hand_/base_link"  # Child frame
        #
        # # Set the translation
        transform.transform.translation.x = 0 #+ anlge
        transform.transform.translation.y = 0
        transform.transform.translation.z = 0
        #
        # # Set the rotation
        quat = tf.transformations.quaternion_from_euler(0, 0, 0)  # Roll, Pitch, Yaw
        transform.transform.rotation.x = quat[0]
        transform.transform.rotation.y = quat[1]
        transform.transform.rotation.z = quat[2]
        transform.transform.rotation.w = quat[3]
        # Publish the transform
        self.tf_broadcaster.sendTransformMessage(transform)



        # Set the header frame ID and timestamp
        transform.header.stamp = rospy.Time.now()
        transform.header.frame_id = "map"  # Parent frame
        transform.child_frame_id = "hand_/base_link1"  # Child frame
        #
        # # Set the translation
        transform.transform.translation.x = 0 #+ anlge
        transform.transform.translation.y = 0
        transform.transform.translation.z = 0
        #
        # # Set the rotation
        quat = tf.transformations.quaternion_from_euler(np.pi/2, 0, np.pi/2)  # Roll, Pitch, Yaw
        transform.transform.rotation.x = quat[0]
        transform.transform.rotation.y = quat[1]
        transform.transform.rotation.z = quat[2]
        transform.transform.rotation.w = quat[3]
        # Publish the transform
        self.tf_broadcaster.sendTransformMessage(transform)


    def set_joint_tf(self, child_frame_name, xyz, quat=None, frame_id='map'):
        print('set_joint_tf', child_frame_name, xyz)
        transform = TransformStamped()

        # Set the header frame ID and timestamp
        transform.header.stamp = rospy.Time.now()
        transform.header.frame_id = frame_id  # Parent frame
        transform.child_frame_id = child_frame_name  # Child frame

        # # Set the translation
        transform.transform.translation.x = xyz[0]
        transform.transform.translation.y = xyz[1]
        transform.transform.translation.z = xyz[2]
        #
        # # Set the rotation
        # quat = tf.transformations.quaternion_from_euler(-np.pi / 9, 0, np.pi)  # Roll, Pitch, Yaw
        if quat is not None:
            transform.transform.rotation.x = quat[0]
            transform.transform.rotation.y = quat[1]
            transform.transform.rotation.z = quat[2]
            transform.transform.rotation.w = quat[3]
        else:
            transform.transform.rotation.x = 0
            transform.transform.rotation.y = 0
            transform.transform.rotation.z = 0
            transform.transform.rotation.w = 1
        # Publish the transform
        self.tf_broadcaster.sendTransformMessage(transform)
  
  

if __name__ =="__main__":
    import sys
    sys.path.append('../../..')
    from utils.identification import gene_robot
 
    hand_tyep = 'gx11'

    if hand_tyep == 'gx11pm':
        rbt1 = gene_robot.get_robot('../../../data/model/gx11pm_finger1.pkl')
        rbt2 = gene_robot.get_robot('../../../data/model/gx11pm_finger2.pkl')
        rbt3 = gene_robot.get_robot('../../../data/model/gx11pm_finger3.pkl')
 
    elif hand_tyep == 'gx11':
        rbt1 = gene_robot.get_robot('../../../data/model/gx11_finger1.pkl')
        rbt2 = gene_robot.get_robot('../../../data/model/gx11_finger2.pkl')
        rbt3 = gene_robot.get_robot('../../../data/model/gx11_finger3.pkl')
  
    app = QApplication(sys.argv)
    rbt_vis = HandVisual([rbt1, rbt2, rbt3])
    # ex = TransformControl()
    rbt_vis.show()
    sys.exit(app.exec_())

