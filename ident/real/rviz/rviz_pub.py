import sys

import rospy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped, Point
from tf import TransformListener
import tf
import numpy as np
import time

from visualization_msgs.msg import Marker, MarkerArray

import sys
sys.path.append('../../../../')

class rviz_pub:
    def __init__(self):
        rospy.init_node('joint_control_node', anonymous=True)
        self.joint_pub = rospy.Publisher('/hand/joint_states', JointState, queue_size=10)
        self.tf_broadcaster = tf.TransformBroadcaster()
        self.marker_pub = rospy.Publisher('/visualization_marker', Marker, queue_size=50)
        # self.marker_array_pub = rospy.Publisher('/visualization_marker_array', MarkerArray, queue_size=50)

        self.tf_listener = TransformListener()  # 添加TF监听器

        # self.init_maker_array()
        self.line_pub = rospy.Publisher('/visualization_marker_line', Marker, queue_size=50)

    def init_maker_array(self):
        # create marker array
        link_idx_list = ['base_link']
        link_idx_list.extend([i for i in range(15)])
        
        stl_dir = 'package://GX11promax/meshes'
        
        self.marker_array = []
        for idx, link_idx in enumerate(link_idx_list):
            marker = Marker()
            if link_idx == 'base_link':
                marker.header.frame_id = "hand_/base_link"
            else:
                marker.header.frame_id = f"hand_/Link{link_idx}"
            marker.id = idx
            marker.type = Marker.MESH_RESOURCE
            marker.mesh_use_embedded_materials = False
            marker.action = Marker.MODIFY
            if link_idx == 'base_link':
                marker.mesh_resource = f"{stl_dir}/base_link.STL"
            else:
                marker.mesh_resource = f"{stl_dir}/Link{link_idx}.STL"
            marker.scale.x = 0
            marker.scale.y = 0
            marker.scale.z = 0
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            self.marker_array.append(marker)

    def pub_marker_array(self):
        pass

    def pub_joint_state(self, pos=None):
        joint_state = JointState()
        joint_state.header.stamp = rospy.Time.now()
        joint_state.name = [f'joint{i + 1}' for i in range(11)]
        if pos is None:
            joint_state.position = [0]*11
        else:
            joint_state.position = pos

        # print('joint_state: ', joint_state.position)

        self.joint_pub.publish(joint_state)
        self.pub_hand_base_link()

    def pub_hand_base_link(self):
        
        transform = TransformStamped()
        
        # Set the header frame ID and timestamp
        transform.header.stamp = rospy.Time.now()
        transform.header.frame_id = "map"  # Parent frame
        transform.child_frame_id = "hand_/base_link"  # Child frame

        # # Set the translation
        transform.transform.translation.x = 0.0 #+ anlge
        transform.transform.translation.y = 0.0
        transform.transform.translation.z = 0.0

        # # Set the rotation
        # if quat is None:
        quat = tf.transformations.quaternion_from_euler(np.pi, 0, 0)  # Roll, Pitch, Yaw
        transform.transform.rotation.x = quat[0]
        transform.transform.rotation.y = quat[1]
        transform.transform.rotation.z = quat[2]
        transform.transform.rotation.w = quat[3]
        
        # Publish the transform
        self.tf_broadcaster.sendTransformMessage(transform)

    # change the link color in rviz via pub
    def change_link_color(self, link_idx, color, visable=True):
        # Create a Marker message
        marker = Marker()
        marker.header.stamp = rospy.Time.now()
        if link_idx == 'base_link':
            marker.header.frame_id = "hand_/base_link"
        else:
            marker.header.frame_id = f"hand_/Link{link_idx}"
        # marker.header.frame_id = f"hand_/Link{link_idx}"
        marker.ns = f"link_color_{link_idx}"
        
        marker.id = link_idx if link_idx != 'base_link' else 100
        marker.type = Marker.MESH_RESOURCE
        marker.mesh_use_embedded_materials = False
        marker.action = Marker.MODIFY
        marker.lifetime = rospy.Duration(0.3)  # 0 means forever
        
        # stl_dir = '/home/gx4070/data/lx/HandOrientation/deploy/dev_ws/src/asset/GX11promax/meshes'
        stl_dir = 'package://GX11promax/meshes'
        
        if link_idx == 'base_link':
            marker.mesh_resource = f"{stl_dir}/base_link.STL"
        else:
            marker.mesh_resource = f"{stl_dir}/Link{link_idx}.STL"

        if visable:
            print(f"Checking mesh file: {marker.mesh_resource}")
        
        # Set the marker pose (position and orientation)
        marker.pose.position.x = 0.0 if visable else 5
        marker.pose.position.y = 0.0 
        marker.pose.position.z = 0.0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        
        marker.scale.x = 1
        marker.scale.y = 1
        marker.scale.z = 1
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = 1.0  # Alpha
        # Publish the Marker message
        self.marker_pub.publish(marker)

    def vis_collision_links(self, collision_links):
        # print('-'*60)
        for idx, vis in enumerate(self.get_gx11_vis_link(collision_links)):
            # print(idx, vis)
            if vis == 1:
                self.change_link_color(idx, [0.0, 0.0, 1, 1])
            # elif vis == 0:
            #     self.change_link_color(idx, [0.3, 0.3, 0.3, 1], visable=False)
            # else:
            #     self.change_link_color(idx, [0.3, 0.3, 0.3, 1])

    def get_gx11_vis_link(self, links=[]):
        viz_link_visable_list = [0]*15
        viz_link_visable_list[0] = -1
        vis_map = [
            # finger 1
            [1], [2], [3, 12],
            # finger 2
            [4], [5], [6], [7, 13],
            # finger 3
            [8], [9], [10], [11, 14]
        ]
        for link in links:
            for vis_link in vis_map[link]:
                viz_link_visable_list[vis_link] = 1
        return viz_link_visable_list
    
    def vis_force(self, link_name="hand_/Link12", force=[1, 0, 0], scale=0.1, lifetime=1.0, start_bias=[0.0, 0.0, -0.02], finger_idx=0):
        # pub an arrow in rviz
        marker = Marker()
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = link_name
        marker.ns = f"force_{finger_idx}"
        marker.id = 100
        marker.type = Marker.ARROW  # 使用箭头类型
        marker.action = Marker.MODIFY
        marker.lifetime = rospy.Duration(lifetime)

        # 设置箭头的起点和终点
        start_point = Point()
        start_point.x = start_bias[0]
        start_point.y = start_bias[1]
        start_point.z = start_bias[2]
        
        end_point = Point()
        end_point.x = force[0] * scale + start_bias[0] # 缩小力向量以便可视化
        end_point.y = force[1] * scale + start_bias[1]
        end_point.z = force[2] * scale + start_bias[2]
        
        marker.points = [start_point, end_point]

        # 设置箭头的尺寸
        marker.scale.x = 0.005  # 箭头尾部宽度
        marker.scale.y = 0.01  # 箭头头部宽度
        marker.scale.z = 0.02  # 箭头长度（如果使用CUBE类型）

        # 设置箭头颜色（红色）
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0  # 不透明度

        # 发布Marker
        self.line_pub.publish(marker)

    def get_relative_position(self, parent_link, child_link):
        try:
            # 等待并获取两个link之间的变换
            self.tf_listener.waitForTransform(parent_link, child_link, rospy.Time(0), rospy.Duration(1.0))
            (trans, rot) = self.tf_listener.lookupTransform(parent_link, child_link, rospy.Time(0))
            return trans, rot  # 返回相对位置
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logwarn(f"Failed to get transform from {parent_link} to {child_link}: {e}")
            return None

if __name__ == '__main__':
    window = rviz_pub()

    # window.change_link_color('base_link', [0.3, 0.3, 0.3, 0.01])
    # for i in range(1, 15):
    #     window.change_link_color(i, [0.3, 0.3, 0.3, 0.01])
    t = 0
    while not rospy.is_shutdown():
        window.pub_hand_base_link()
        window.pub_joint_state([np.pi/10, 0, 0,
                    0, 0, 0, 0,
                    0, 0, 0, 0])
        
        # window.vis_collision_links([2, 6, 10])
        window.vis_force()

        print(t)
        t += 1
        rospy.sleep(0.1)
        # time.sleep(0.2)
    # window.pub_hand_base_link()
    # rospy.spin()

