

import rospy
import tf
import tf2_ros
import geometry_msgs.msg

import numpy as np

from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped

class ValveRviz:
    def __init__(self):
        rospy.init_node('valve_rviz')
        # self.joint_names = ['valve_/base_link', 'valve_/valve_link']
        self.joint_pub = rospy.Publisher('/valve/joint_states', JointState, queue_size=10)
        self.tf_broadcaster = tf.TransformBroadcaster()
        
        self.pub_valve_base_link()
    
    def publish_valve(self, valve_name, valve_pos):
        pass

    def pub_valve_base_link(self):

        transform = TransformStamped()
        
        # Set the header frame ID and timestamp
        transform.header.stamp = rospy.Time.now()
        transform.header.frame_id = "map"  # Parent frame
        transform.child_frame_id = "valve_/base_link"  # Child frame

        # # Set the translation
        transform.transform.translation.x = 0.0 #+ anlge
        transform.transform.translation.y = 0.0
        transform.transform.translation.z = 0.0

        # # Set the rotation
        quat = tf.transformations.quaternion_from_euler(-np.pi/2, 0, 0)  # Roll, Pitch, Yaw
        transform.transform.rotation.x = quat[0]
        transform.transform.rotation.y = quat[1]
        transform.transform.rotation.z = quat[2]
        transform.transform.rotation.w = quat[3]
        
        # Publish the transform
        self.tf_broadcaster.sendTransformMessage(transform)
        
    def pub_joint_pos(self, joint_pos):

        joint_state = JointState()
        joint_state.header.stamp = rospy.Time.now()
        joint_state.name = ['joint1']
        joint_state.position = joint_pos
        
        self.joint_pub.publish(joint_state)
        self.pub_valve_base_link()

    
if __name__ == '__main__':
    valve_rviz = ValveRviz()
    i = 0
    while True:
        valve_rviz.pub_valve_base_link()
        
        valve_rviz.pub_joint_pos([i*0.02])
        
        rospy.sleep(0.1)
        
        i += 1
    rospy.spin()
