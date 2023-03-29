#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
import sys, os
import numpy as np
import time
from detector import detection_model, MetadataCatalog, Visualizer
from std_msgs.msg import String
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
from measurement3d_2d import compute_line, compute_projected_point, compute_measurements
from tf_transformations import euler_from_matrix
from tf2_ros.transform_listener import TransformListener
from tf2_ros.buffer import Buffer
from tf_transformations import quaternion_matrix, quaternion_about_axis

 
# Importing Visualization UTILS
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(os.path.join(parent, 'UTILS')) 
from visualize import visualize


class DetectionNode(Node):

    def __init__(self, model_name, display_enabled = False, sim_enabled = False): # COMPLETED
        super().__init__('tree_detect')
        # ROS communication utilities
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        rgb_topic = str(self.declare_parameter("rgb_topic").value)
        depth_topic = str(self.declare_parameter("depth_topic").value)
        measurement_topic = str(self.declare_parameter("measurement_topic").value)
        confidence = float(self.declare_parameter("detection_confidence").value)
        sampling_t = float(self.declare_parameter("sampling_t").value)
        self.maximum_depth = float(self.declare_parameter("maximum_depth").value)
        self.below_cam_th = float(self.declare_parameter("below_cam_th").value)
        self.inclination_th = np.deg2rad(float(self.declare_parameter("inclination_th").value))

        # Define machine learning model for detecting tree trunks
        self.DL_MODEL = detection_model(model_name, confidence)
        # Create a cv_bridge instance to convert from Image msg to CV img
        self.bridge = CvBridge()
        # Subscription to both RGB and DEPTH images
        self.rgb_sub = self.create_subscription(CompressedImage, rgb_topic, self.rgb_callback, 1)
        self.depth_sub = self.create_subscription(Image, depth_topic, self.depth_callback, 1) # real robot
        # Synchronous routine for computing measruments
        # self.detect_routine = rclpy.timer.Timer(rclpy.duration.Duration(sampling_t), self.inference_callback) # Run Inference at 10 Hz
        self.detect_routine = self.create_timer(sampling_t  , self.inference_callback)
        # Publisher for resulting measurements
        self.measurement_pub = self.create_publisher(Float32MultiArray, measurement_topic, 1)
        self.measurement_msg = Float32MultiArray()

        # Publisher for Debuggin purposes
        self.rgb_detection_pub = self.create_publisher(Image, "/rgb_keypoints", 1)
        self.depth_detection_pub = self.create_publisher(Image, "/depth_keypoints", 1)
        self.visualizer = visualize()

        # Image Storage
        self.rgb_img = None
        self.depth_img = None
        self.rgb_img_shape = None
        self.depth_img_shape = None
        
        self.display_enabled = display_enabled
        self.sim_enabled = sim_enabled
        self.rgb_keypoints_list = []
        self.depth_keypoints_list = []

        # Intrinsic Camera Parameters
        self.fx_rgb = float(self.declare_parameter("fx_rgb").value)
        self.fy_rgb = float(self.declare_parameter("fy_rgb").value)
        self.cx_rgb = float(self.declare_parameter("cx_rgb").value)
        self.cy_rgb = float(self.declare_parameter("cy_rgb").value)

        self.fx_depth = float(self.declare_parameter("fx_depth").value)
        self.fy_depth = float(self.declare_parameter("fy_depth").value)
        self.cx_depth = float(self.declare_parameter("cx_depth").value)
        self.cy_depth = float(self.declare_parameter("cy_depth").value)

        self.rgb_M = np.array([[self.fx_rgb, 0.0, self.cx_rgb],
                               [0.0, self.fy_rgb, self.cy_rgb],
                               [0.0, 0.0, 1.0]])

        self.depth_M = np.array([[self.fx_depth, 0.0, self.cx_depth],
                                 [0.0, self.fy_depth, self.cy_depth],
                                 [0.0, 0.0, 1.0]])

        self.camera_wrt_base_link = np.array([0.4, 0, 0.1])

        self.rgb2depth = self.depth_M@np.linalg.pinv(self.rgb_M)


    def rgb_callback(self, data): # COMPLETED
        '''
        Callback used to store the most recent RGB image
        '''
        try:
            self.rgb_img = self.bridge.compressed_imgmsg_to_cv2(data)
            if self.rgb_img_shape is None:
                self.rgb_img_shape = self.rgb_img.shape
        except CvBridgeError as e:
            print(e)

    def depth_callback(self, data): # COMPLETED
        '''
        Callback used to store the most recent depth image
        '''
        try:
            self.depth_img = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
            if self.depth_img_shape is None:
                self.depth_img_shape = self.depth_img.shape
        except CvBridgeError as e:
            print(e)

    def inference_callback(self): # STILL DEVELOPMENT, REQUIRES CLEANING!!!!
        
        if self.rgb_img is None or self.depth_img is None:
            return None 

        # Grabbing the latest images received by the system
        rgb_img = self.rgb_img
        depth_img = self.depth_img
        # print("BEFORE INSTANCE")
        # Execute inference with the DL model for deteting tree trunk keyoints
        time1 = time.time()
        outputs_pred = self.DL_MODEL.predict(rgb_img)
        print("inference time", time.time() - time1, "s")
        # Get detected instaces (Three keypoints along the trunk)

        instances = outputs_pred['instances'].get('pred_keypoints').to("cpu").numpy()
        # print("OUTSIDE INSTANCE", instance[0])
        # Iterate over the instances 
        for instance in instances:
            # print("INSIDE INSTANCE FOUND")
            # instance[:, 0] are the x along the width
            # instance[:, 1] are the y along the height
            # Getting coordinate indices from the rgb camera
            # for the keypoints
            ij = instance[[0, 3, 4], :]

            # Converting to depth img coordinates
            depth_ij = self.rgb2depth@np.vstack((ij[:, 0:2].T, np.ones(3)))
            depth_ij = np.abs(np.floor(depth_ij[0:2, :].T)).astype(np.uint32)

            if self.display_enabled:
                self.rgb_keypoints_list.append(ij.astype(np.uint32))
                self.depth_keypoints_list.append(depth_ij)

            # Set z as a column vector and convert it to meters
            z = depth_img[depth_ij[:, 1], depth_ij[:, 0]]

            if (z > 0.0).all() and (z < self.maximum_depth).all(): # if no measurements with zero values proceed to compute 3d points fro the depth measurements
                
                # Compute 3D point wrt to the camera frame given the depth measurements
                # Convert to meters
                Pc = (np.vstack(((ij[:, 0] - self.cx_rgb)/self.fx_rgb, (ij[:, 1] - self.cy_rgb)/self.fy_rgb, np.ones(3)))*z)
                # print('---', Pc)
                # Make the Points Homogeneous
                Pc = np.vstack((Pc, np.ones(3))) # [x, y, z, 1]'

                if (Pc[1, :] < self.below_cam_th).all(): # Discard points that are 0.3 meters below the camera frame (0.3 units in the positive y axis of the camera_depth_optical_frame)
                        
                    # Compute measurement given the detected instances
                    p0, d = compute_line(Pc.T[:, 0:3]) # Points in [x, y, z] format
                    projected_point_c = compute_projected_point(p0, d)
                    distance, angle, _, angle_y, _ = compute_measurements(projected_point_c, d)
                    # Only consider lines with less than 0.1 rad difference wrt the camera's y axis
                    if angle_y[0, 0] < self.inclination_th:
                        # Publish Measurment to SLAM node
                        self.measurement_msg.data = [distance, angle]
                        self.measurement_pub.publish(self.measurement_msg)

                        if self.sim_enabled:
                            self.point3Dvis(Pc)                 # Display Detected 3D points (for simulation only)
                            self.cylinder3Dvis(distance, angle) # Display detected landmark according to the range and angle measured

        if self.display_enabled:
            # Show estimated tree trunk keypoints
            self.show_keypoints(rgb_img, depth_img)
            

    def point3Dvis(self, Pc):
        # Ground Truth Transform for Visualization and debugging (MEasured 3D points)
        T = self.visualizer.get_transform('oakd_camera_rgb_camera_optical_frame', 'odom', self.tf_buffer)
        Pw = T@Pc
        self.visualizer.draw_point(Pw[:, [0]])
        self.visualizer.draw_point(Pw[:, [1]])
        self.visualizer.draw_point(Pw[:, [2]])

    def get_transform(self, from_, to_):
        transform = self.tf_buffer.lookup_transform(to_, from_, rclpy.time.Time())
        trans = transform.transform.translation
        rot = transform.transform.rotation
        trans = [trans.x, trans.y, trans.z]
        rot = [rot.x, rot.y, rot.z, rot.w]
        T =  quaternion_matrix(rot)
        T[0, 3] = trans[0]
        T[1, 3] = trans[1]
        T[2, 3] = trans[2]
        return T

    def cylinder3Dvis(self, distance, angle):
        # Ground Truth Transform for Visualization and debugging (Measured 3D landmark)
        # T = self.visualizer.get_transform('base_link', 'odom', self.tf_buffer)
        # _, _ , yaw = euler_from_matrix(T[0:3, 0:3])
        # lx = T[0, 3] + (distance + self.camera_wrt_base_link[0])*np.cos(angle + yaw)
        # ly = T[1, 3] + (distance + self.camera_wrt_base_link[0])*np.sin(angle + yaw) 

        # T = self.visualizer.get_transform('oakd_camera_rgb_camera_optical_frame', 'odom', self.tf_buffer)
        # T1 = self.visualizer.get_transform('base_link', 'odom', self.tf_buffer)

        T = self.get_transform('oakd_camera_rgb_camera_optical_frame', 'odom')
        T1 = self.get_transform('base_link', 'odom')
        print("---", T1[0:2, 3])
        _, _ , yaw = euler_from_matrix(T1[0:3, 0:3])
        lx = T[0, 3] + (distance)*np.cos(angle + yaw)
        ly = T[1, 3] + (distance)*np.sin(angle + yaw) 
        self.visualizer.draw_cylinder(np.asarray([[lx],[ly]]))

    def show_keypoints(self, rgb_img, depth_img) :

        for i in range(len(self.rgb_keypoints_list)):

            cv2.circle(rgb_img, (self.rgb_keypoints_list[i][0, 0], self.rgb_keypoints_list[i][0, 1]), 5, (255, 0, 0), 2)
            cv2.circle(rgb_img, (self.rgb_keypoints_list[i][1, 0], self.rgb_keypoints_list[i][1, 1]), 5, (255, 0, 0), 2)
            cv2.circle(rgb_img, (self.rgb_keypoints_list[i][2, 0], self.rgb_keypoints_list[i][2, 1]), 5, (255, 0, 0), 2)

            cv2.circle(depth_img, (self.depth_keypoints_list[i][0, 0], self.depth_keypoints_list[i][0, 1]), 5, (255, 255, 255), 2)
            cv2.circle(depth_img, (self.depth_keypoints_list[i][1, 0], self.depth_keypoints_list[i][1, 1]), 5, (255, 255, 255), 2)
            cv2.circle(depth_img, (self.depth_keypoints_list[i][2, 0], self.depth_keypoints_list[i][2, 1]), 5, (255, 255, 255), 2)

        self.rgb_detection_pub.publish(self.bridge.cv2_to_imgmsg(rgb_img, encoding="rgb8"))
        self.depth_detection_pub.publish(self.bridge.cv2_to_imgmsg(depth_img, encoding="passthrough"))
        self.rgb_keypoints_list = []; self.depth_keypoints_list = []


def main(args): # COMPLETED
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_name = os.path.join(current_dir,"R-50_RGB_60k.pth")
    # model_name = os.path.join(current_dir,"ResNext-101_fold_01.pth")
    rclpy.init(args=args)
    # 'detect_node', anonymous=True
    display_enabled = 'True' == args[1]
    sim_enabled = 'True' == args[2]
    ic = DetectionNode(model_name, display_enabled, sim_enabled)
    print("Model", model_name, "Loaded")
    print("NODE STARTED -- 'detect_node.py'")
    try:
        rclpy.spin(ic)
    except KeyboardInterrupt:
        print("Shutting down")

if __name__ == '__main__':
    main(sys.argv)
