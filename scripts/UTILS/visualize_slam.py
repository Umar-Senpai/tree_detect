#! /usr/bin/env python

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
# import tf
import numpy as np

from tf_transformations import quaternion_matrix, quaternion_about_axis

class visualize(Node):

    def __init__(self):
        super().__init__('slam_visualize')
        self.frame_id = "odom"

        self.points_topic = str(self.declare_parameter("points_topic").value)
        self.cylinder_topic = str(self.declare_parameter("cylinder_topic").value)
        self.robot_vis_topic = str(self.declare_parameter("robot_vis_topic").value)
        self.landmark_vis_topic = str(self.declare_parameter("landmark_vis_topic").value)
        self.points_pub = self.create_publisher(Marker, self.points_topic, 2)
        self.cylinder_pub = self.create_publisher(Marker, self.cylinder_topic, 2)
        self.robot_cov = self.create_publisher(Marker, self.robot_vis_topic, 2)
        self.marker_array_pub = self.create_publisher(MarkerArray, self.landmark_vis_topic, 2)

        self.id_cyl = 0
        self.id_pt = 0
        
        self.origin, self.xaxis, self.yaxis, self.zaxis = (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)

        self.marker_array = MarkerArray()

    def reset_ids(self):
        self.id_cyl = 0
        self.id_pt = 0
        
    def draw_cylinder(self, point):

        marker = Marker()

        marker.header.frame_id = self.frame_id
        marker.header.stamp = self.get_clock().now().to_msg()

        # set shape, Arrow: 0; Cube: 1 ; Sphere: 2 ; Cylinder: 3
        marker.type = 3
        marker.id = self.id_cyl

        # Set the scale of the marker
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 5.0

        # Set the color
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        # Set the pose of the marker
        marker.pose.position.x = point[0, 0]
        marker.pose.position.y = point[1, 0]
        marker.pose.position.z = 0.0

        # print("publishing landmark")

        self.cylinder_pub.publish(marker)
        self.id_cyl += 1

    def draw_point(self, point):

        marker = Marker()

        marker.header.frame_id = self.frame_id
        marker.header.stamp = self.get_clock().now().to_msg()

        # set shape, Arrow: 0; Cube: 1 ; Sphere: 2 ; Cylinder: 3
        marker.type = 3
        marker.id = self.id_pt

        # Set the scale of the marker
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3

        # Set the color
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        # Set the pose of the marker
        marker.pose.position.x = point[0, 0]
        marker.pose.position.y = point[1, 0]
        marker.pose.position.z = point[2, 0]
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        self.points_pub.publish(marker)
        self.id_pt += 1

    def draw_ellipse(self, pose, cov, id = 0, p = 0.95, color = (0.0, 0.0, 1.0, 1.0), publish = True):

        s = -2 * np.log(1 - p)

        lambda_, v = np.linalg.eig(s*cov)

        lambda_ = np.sqrt(lambda_)

        angle = np.arccos(v[0, 0])

        qz = quaternion_about_axis(angle, self.zaxis)

        marker = Marker()

        marker.header.frame_id = self.frame_id
        marker.header.stamp = self.get_clock().now().to_msg()

        # set shape, Arrow: 0; Cube: 1 ; Sphere: 2 ; Cylinder: 3
        marker.type = 2
        marker.id = id

        # Set the scale of the marker
        marker.scale.x = lambda_[0]
        marker.scale.y = lambda_[1]
        marker.scale.z = 0.001

        # Set the color
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = 0.5

        # Set the pose of the marker
        marker.pose.position.x = pose[0, 0]
        marker.pose.position.y = pose[1, 0]
        marker.pose.position.z = 0.0
        marker.pose.orientation.x = qz[0]
        marker.pose.orientation.y = qz[1]
        marker.pose.orientation.z = qz[2]
        marker.pose.orientation.w = qz[3]

        if publish:
            self.robot_cov.publish(marker)

        else:
            return marker

    def draw_ellipses(self):
        self.marker_array_pub.publish(self.marker_array)
        self.marker_array.markers = []

    def get_transform(self, from_, to_, tf_buffer):
        transform = tf_buffer.lookup_transform(to_, from_, rclpy.time.Time())
        trans = transform.transform.translation
        rot = transform.transform.rotation
        trans = [trans.x, trans.y, trans.z]
        rot = [rot.x, rot.y, rot.z, rot.w]
        T =  quaternion_matrix(rot)
        T[0, 3] = trans[0]
        T[1, 3] = trans[1]
        T[2, 3] = trans[2]
        return T

