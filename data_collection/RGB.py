import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, DurabilityPolicy, ReliabilityPolicy, QoSProfile, HistoryPolicy

import tf2_geometry_msgs
from tf2_ros import TransformListener, Buffer, LookupException, ConnectivityException, ExtrapolationException

from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import PoseStamped
from capella_ros_msg.msg import GarbageDetect

import cv2
import time
import math
import numpy as np
import os
from functools import partial

from ultralytics import YOLO

from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup


class Person_Count(Node):
    def __init__(self):
        super().__init__('garbage_detection_monocular')

        # YOLO模型加载
        self.last_save_time = 0.0
        self.model = YOLO(os.path.join(os.path.dirname(__file__),
                                       '/capella/lib/python3.10/site-packages/garbage_detection_node/garbage_detection_yolo11_324.pt'))
        self.get_logger().info("YOLO Model initialization successful !")

        # 相机配置字典
        self.cameras = {
            'front': {
                'topic': '/rgb_camera_front/compressed',
                'frame_id': 'front_camera_color_frame',
                'fx': 454.58425,
                'fy': 452.05297,
                'cx': 309.77324,
                'cy': 280.9,
                'theta': 7.0,  # 俯仰角度
                'height': 0.905,  # 相机高度
                'horizontal_fov_deg': 90.0,
                'v_fov_deg': 73.0,
            },
            'back': {
                'topic': '/rgb_camera_back/compressed',
                'frame_id': 'back_camera_color_frame',
                'fx': 1312.81579,
                'fy': 1319.68694,
                'cx': 681.62665,
                'cy': 508.38222,
                'theta': 1.7,
                'height': 0.635,
                'horizontal_fov_deg': 60.0,
                'v_fov_deg': 58.0,
            },
            'left': {
                'topic': '/rgb_camera_left/compressed',
                'frame_id': 'left_camera_color_frame',
                'fx': 454.58425,
                'fy': 452.05297,
                'cx': 309.77324,
                'cy': 280.9,
                'theta': 5.0,
                'height': 0.8,
                'horizontal_fov_deg': 90.0,
                'v_fov_deg': 73.0,
            },
            'right': {
                'topic': '/rgb_camera_right/compressed',
                'frame_id': 'right_camera_color_frame',
                'fx': 454.58425,
                'fy': 452.05297,
                'cx': 309.77324,
                'cy': 280.9,
                'theta': 5.0,
                'height': 0.8,
                'horizontal_fov_deg': 90.0,
                'v_fov_deg': 73.0,
            },
        }

        # QoS配置
        qos_ = QoSProfile(depth=1)
        qos_.reliability = ReliabilityPolicy.BEST_EFFORT

        # 为每个相机创建订阅
        self.camera_subscriptions = {}

        for camera_name, cam_config in self.cameras.items():
            callback_group = MutuallyExclusiveCallbackGroup()

            self.camera_subscriptions[camera_name] = self.create_subscription(
                CompressedImage,
                cam_config['topic'],
                partial(self.camera_callback, camera_name=camera_name),
                qos_,
                callback_group=callback_group,
            )
            self.get_logger().info(f"Subscribed to {cam_config['topic']}")

        # 发布器
        self.pose_publisher = self.create_publisher(GarbageDetect, '/garbage_cord', 1)

        # TF监听器
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # 检测参数
        self.detecting_conf_threshold = 0.4
        self.max_detecting_distance = 500.0  # 最大检测距离（米）

    def reverse_compressed_image(self, compressed_msg, camera_name: str) -> tuple:
        """
        解码压缩图像
        :param compressed_msg: CompressedImage, 压缩图像消息
        :param camera_name: str, 相机名称

        return: tuple, (img: np.ndarray, current_time: rclpy.time.Time)
        """
        current_time = rclpy.time.Time.from_msg(compressed_msg.header.stamp)
        img = cv2.imdecode(np.frombuffer(compressed_msg.data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            self.get_logger().warn(f"[{camera_name}] 压缩图像解码失败.", once=True)
            return None, None
        self.get_logger().info(f"[{camera_name}] 压缩图像解码成功.", once=True)
        return img, current_time

    def calculate_position_from_bbox(self, bbox, cam_config, img_width, img_height, class_id):
        """
        从检测框计算相机坐标系下的3D位置
        :param bbox: [x1, y1, x2, y2] 检测框
        :param cam_config: 相机配置字典
        :param img_width: 图像宽度    像素
        :param img_height: 图像高度    像素
        :param class_id: 垃圾类别ID
        :return: (x, y, z) 相机坐标系下的位置，或None
        """
        x1, y1, x2, y2 = bbox

        # 检测框中心和尺寸
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        box_height = y2 - y1
        box_width = x2 - x1

        # 检测框底部y坐标
        y_bottom = y2

        # 方法1: 基于相机内参和俯仰角的几何估算
        theta_rad = np.deg2rad(cam_config['theta'])
        k = (y_bottom - cam_config['cy']) / cam_config['fy']
        X_visual = cam_config['height'] * (np.cos(theta_rad) - k * np.sin(theta_rad)) / \
                   (np.sin(theta_rad) + k * np.cos(theta_rad))

        # 计算水平角度
        deviation_rad = np.deg2rad(
            ((cam_config['cx'] - (img_width / 2)) / img_width) * cam_config['horizontal_fov_deg'])
        theta_x_center_rad = np.deg2rad((0.5 - center_x / img_width) * cam_config['horizontal_fov_deg']) + deviation_rad
        Y_visual = X_visual * np.tan(theta_x_center_rad)

        # 距离检查
        distance = np.sqrt(X_visual ** 2 + Y_visual ** 2)
        if distance > self.max_detecting_distance:
            self.get_logger().info(f"Object too far: {distance:.2f}m", throttle_duration_sec=2.0)
            return None

        # 返回相机坐标系下的位置 (x向前, y向左, z向上)
        return X_visual, Y_visual, 0.0

    def save_image_every_1s(self, color_data, save_dir):

        now = time.time()
        if now - self.last_save_time < 3.0:
            return

        os.makedirs(save_dir, exist_ok=True)

        filename = time.strftime("%Y%m%d_%H%M%S", time.localtime(now)) + ".jpg"
        cv2.imwrite(os.path.join(save_dir, filename), color_data)

        self.last_save_time = now

    def camera_callback(self, msg, camera_name: str):
        """
        单目压缩图像回调函数
        :param msg: CompressedImage消息
        :param camera_name: 相机名称 ('front', 'back', 'left', 'right')
        """
        if len(msg.data) == 0:
            self.get_logger().warn(f"[{camera_name}] 摄像头无数据!", throttle_duration_sec=2.0)
            return

        start_time = time.perf_counter()

        # 获取相机配置
        cam_config = self.cameras[camera_name]

        # 解码压缩图像
        img, current_time = self.reverse_compressed_image(msg, camera_name)
        if img is None:
            return
        self.save_image_every_1s(img, save_dir='/capella/lib/python3.10/site-packages/garbage_detection_node/RGB_data')

def main(args=None):
    rclpy.init(args=args)
    garbage_detection_demonstration = GarbageDetectionDemonstration()
    multi_executor = MultiThreadedExecutor(num_threads=4)
    multi_executor.add_node(garbage_detection_demonstration)
    multi_executor.spin()
    garbage_detection_demonstration.destroy_node()
    multi_executor.shutdown()


if __name__ == '__main__':
    main()
