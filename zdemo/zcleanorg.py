# -*- coding: UTF-8 -*-
# 控制清洁机器人清扫指定位置的垃圾

import cv2
import time
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy import clock
import numpy as np
from rclpy.qos import qos_profile_sensor_data, DurabilityPolicy, ReliabilityPolicy, QoSProfile, HistoryPolicy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.action import ActionServer, GoalResponse, CancelResponse

from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_geometry_msgs import do_transform_pose_stamped
from geometry_msgs.msg import PoseStamped, TransformStamped, PoseArray, Pose
import math
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateThroughPoses
from nav2_msgs.srv import ClearEntireCostmap
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Float32, Bool
from capella_ros_msg.action import SearchGarbage
from capella_ros_msg.msg import DeviceStatus, GarbageDetect
# 透视相机检测人的相关依赖
import threading
import copy


class CleanGarbage(Node):

    def __init__(self):
        super().__init__('clean_garbage_node')

        callback_gp1 = MutuallyExclusiveCallbackGroup()
        callback_gp2 = MutuallyExclusiveCallbackGroup()
        callback_gp3 = MutuallyExclusiveCallbackGroup()
        callback_gp4 = MutuallyExclusiveCallbackGroup()
        callback_gp5 = MutuallyExclusiveCallbackGroup()
        callback_gp6 = MutuallyExclusiveCallbackGroup()
        callback_gp7 = MutuallyExclusiveCallbackGroup()
        qos_ = QoSProfile(depth=1)
        qos_.durability = DurabilityPolicy.TRANSIENT_LOCAL

        # subscribe garbage detect topic
        '''数据接收接口'''
        self.garbage_detect_sub = self.create_subscription(GarbageDetect, '/garbage_cord',
                                                           self.garbage_detect_sub_callback, 1,
                                                           callback_group=callback_gp1)  #
        self.garbage_detect_sub  # prevent unused variable warning

        self.device_status_sub = self.create_subscription(DeviceStatus, '/device_status',
                                                          self.device_status_sub_callback, 1,
                                                          callback_group=callback_gp2)  #
        self.device_status_sub  # prevent unused variable warning

        self.local_costmap_sub = self.create_subscription(OccupancyGrid, '/local_costmap/costmap',
                                                          self.local_costmap_sub_callback, qos_,
                                                          callback_group=callback_gp3)  #
        self.local_costmap_sub  # prevent unused variable warning

        self.map_sub = self.create_subscription(OccupancyGrid, '/global_costmap/costmap', self.map_sub_callback, qos_,
                                                callback_group=callback_gp4, )  #
        self.map_sub  # prevent unused variable warning

        self.manual_control = self.create_subscription(Bool, '/clean_garbage_manual', self.manual_control_callback,
                                                       qos_, callback_group=callback_gp6)  #
        self.manual_control  # prevent unused variable warning
        self.clean_action_succed_sub = self.create_subscription(Bool, '/clean_action_success', self.clean_action_succeed_callback,
                                                       qos_, callback_group=callback_gp7)  #
        self.clean_action_succed_sub  # prevent unused variable warning
        '''数据发送接口'''
        self.vacuum_cleaner_control_pub = self.create_publisher(Bool, '/vacuum_cleaner_ctr_proxy', 10)  #
        self.vacuum_cleaner_control_pub  # prevent unused variable warning

        # self.rolling_brush_control_pub = self.create_publisher(Bool,'/rolling_brush_ctr_proxy',10)#
        # self.rolling_brush_control_pub  # prevent unused variable warning
        #
        # self.suction_sewage_control_pub = self.create_publisher(Bool,'/suction_sewage_ctr_proxy',10)#
        # self.suction_sewage_control_pub  # prevent unused variable warning

        self.rolling_brush_pushrod_control_pub = self.create_publisher(Bool, '/rolling_brush_pushrod_ctr_proxy', 10)
        self.rolling_brush_pushrod_control_pub  # prevent unused variable warning

        self.rolling_brush_motor_control_pub = self.create_publisher(Bool, '/rolling_brush_motor_ctr_proxy', 10)
        self.rolling_brush_motor_control_pub  # prevent unused variable warning

        self.suction_sewage_pushrod_control_pub = self.create_publisher(Bool, '/suction_sewage_pushrod_ctr_proxy', 10)
        self.suction_sewage_pushrod_control_pub  # prevent unused variable warningAdd commentMore actions

        self.suction_sewage_motor_control_pub = self.create_publisher(Bool, '/suction_sewage_motor_ctr_proxy', 10)
        self.suction_sewage_motor_control_pub  # prevent unused variable warning

        self.clear_water_control_pub = self.create_publisher(Bool, '/clear_water_ctr_proxy', 10)
        self.clear_water_control_pub  # 新增喷水

        self.whirl_control_pub = self.create_publisher(Twist, '/cmd_vel_nav', 1)  #
        self.whirl_control_pub  # prevent unused variable warning

        self.clear_lcost_map_client = self.create_client(ClearEntireCostmap,
                                                         '/local_costmap/clear_entirely_local_costmap')
        self.clear_lcost_map_client

        self.clear_gcost_map_client = self.create_client(ClearEntireCostmap,
                                                         '/global_costmap/clear_entirely_global_costmap')
        self.clear_gcost_map_client

        self.goal_test_pub = self.create_publisher(PoseArray, '/garbage_test', 10)

        # clean garbage action server
        self.send_goal_action = ActionClient(self, NavigateThroughPoses, '/navigate_through_poses')

        # rate
        self.rate_ = self.create_rate(10.0)

        # 创建tf2监听器
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # update_image_thread = threading.Thread(target=self.update_image)
        # update_image_thread.start()

        # action
        action_server_feedback_qos = QoSProfile(depth=1)
        action_server_feedback_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.faction_server = ActionServer(self, SearchGarbage, '/search_garbage',
                                           self.action_server_callback,
                                           goal_callback=self.action_goal_callback,
                                           cancel_callback=self.action_cancel_callback,
                                           callback_group=callback_gp5,
                                           feedback_pub_qos_profile=action_server_feedback_qos)  #
        self._reset_variable()

        # 定义参数
        self.declare_parameter('goal_num', 3)  # 机器人在清理垃圾时行走的goal的数量，3或者4（在垃圾附近有障碍物时使用）
        self.declare_parameter('goal_num_angle', math.pi / 4)  # 弧度，机器人与垃圾障碍物的角度大于该角度时，使用goal_num
        self.declare_parameter('goal_cut', 1.5)  # m，垃圾前一个goal距离垃圾goal的欧式距离
        self.declare_parameter('goal_addition', 2.0)  # m，机器人清扫完垃圾的终点goal距离垃圾goal的欧式距离，干垃圾为0.5倍goal_addition
        self.declare_parameter('map_resolution', 0.05)  # m/pixel，地图分辨率
        self.declare_parameter('min_gap_to_clean', 0.8)  # m，如果垃圾距离障碍物小于这个距离就不再清扫
        self.declare_parameter('backward_distance', 0.3)  # m，当机器人收到垃圾清扫命令但没看到垃圾时需要后退的距离
        self.declare_parameter('backward_speed', 0.1)  # m/s，当机器人收到垃圾清扫命令但没看到垃圾时需要后退的速度
        self.declare_parameter('robot_radius', 1.3)  # m，机器人的半径，在旋转寻找垃圾时使用
        self.declare_parameter('garbage_inflation_radius',
                               0.5)  # m，以垃圾为圆心，garbage_inflation_radius半径内没有障碍物就只发两个点清扫垃圾，goal_num就不起作用
        self.declare_parameter('whirl_speed', math.pi / 10)  # radian/s，机器人原地旋转寻找垃圾的速度

        #===================================================================================
        self.action_callback_init_time = None  
        self.is_init_blocking = False  
        # =================================================================================
        # 加载参数
        self.goal_num = self.get_parameter('goal_num').get_parameter_value().integer_value
        self.goal_num_angle = self.get_parameter('goal_num_angle').get_parameter_value().double_value
        self.goal_cut = self.get_parameter('goal_cut').get_parameter_value().double_value
        self.goal_addition = self.get_parameter('goal_addition').get_parameter_value().double_value
        self.map_resolution = self.get_parameter('map_resolution').get_parameter_value().double_value
        self.min_gap_to_clean = self.get_parameter('min_gap_to_clean').get_parameter_value().double_value
        self.backward_distance = self.get_parameter('backward_distance').get_parameter_value().bool_value
        self.backward_speed = self.get_parameter('backward_speed').get_parameter_value().double_value
        self.robot_radius = self.get_parameter('robot_radius').get_parameter_value().double_value
        self.garbage_inflation_radius = self.get_parameter(
            'garbage_inflation_radius').get_parameter_value().double_value
        self.whirl_speed = self.get_parameter('whirl_speed').get_parameter_value().double_value

        # 显示参数
        self.get_logger().info("loading parameters : ")
        self.get_logger().info(f"goal_num : {self.goal_num} ")
        self.get_logger().info(f"goal_num_angle : {self.goal_num_angle} °")
        self.get_logger().info(f"goal_cut : {self.goal_cut} m")
        self.get_logger().info(f"goal_addition : {self.goal_addition} m")
        self.get_logger().info(f"map_resolution : {self.map_resolution} pixel/m")
        self.get_logger().info(f"min_gap_to_clean : {self.min_gap_to_clean} m")
        self.get_logger().info(f"backward_distance : {self.backward_distance} m")
        self.get_logger().info(f"backward_speed : {self.backward_speed} m/s")
        self.get_logger().info(f"robot_radius : {self.robot_radius} m")
        self.get_logger().info(f"garbage_inflation_radius : {self.garbage_inflation_radius} m")
        self.get_logger().info(f"whirl_speed : {self.whirl_speed} r/s")

        self.get_logger().info('initialize program complete')
        self.action_success_sign = False

        # setup self.logger for logging garbage coordinates
        self.logger = self.get_logger()

    def garbage_detect_sub_callback(self, msg):
        if msg.class_id == 2:
            return
        if self.start_clean_signal:

            if self.manual_control_signal is None:
                if time.time() - self.receive_signal_time < 0.5:
                    self.get_logger().info('sleep 0.5s', throttle_duration_sec=1)
                elif self.can_it_rotate or self.garbage_positions is not None or self.action_request_goal is not None:
                    if self.map is not None:
                        '''获取机器人位姿'''
                        base_link_t = None
                        try:
                            base_link_t = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
                            self.receive_tf2_time = time.time()
                        except Exception as ex:
                            self.get_logger().info(f'get tf failed:{ex}')

                        if base_link_t is not None:
                            '''识别到一堆以后，就把它的历史数据把它弄进去，然后取最后一帧的垃圾位置作为它的真正的位置'''
                            self.garbage_history_list.append(
                                [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.class_id])
                            '''这是将历史数据复制过去  平滑垃圾位置——如果当前帧检测不到垃圾或检测噪声，就用历史可靠数据代替'''
                            base_link_x = base_link_t.transform.translation.x
                            base_link_y = base_link_t.transform.translation.y
                            base_link_r = base_link_t.transform.rotation
                            tf2_map_base_link = TransformStamped()
                            tf2_map_base_link.child_frame_id = 'base_link'
                            tf2_map_base_link.header.frame_id = 'map'
                            tf2_map_base_link.header.stamp = self.get_clock().now().to_msg()
                            tf2_map_base_link.transform.translation.x = np.float(base_link_x)
                            tf2_map_base_link.transform.translation.y = np.float(base_link_y)
                            tf2_map_base_link.transform.rotation = base_link_r
                            
                            # 超过一秒没检测到垃圾就恢复旋转速度
                            if time.time() - self.receive_valid_data_time > 1:
                                self.whirl_speed = math.pi / 10
                                
                            # 计算机器人在map下的欧拉角   把随后一个历史数据的垃圾拿出来
                            roll, pitch, yaw = self.quart_to_rpy(base_link_r.x, base_link_r.y, base_link_r.z,
                                                                 base_link_r.w)
                            garbage_history_list_ = np.array(self.garbage_history_list)
                            garbage_history_list_ = garbage_history_list_[garbage_history_list_[:, -1] != -1]  # -1为空数据
                            
                            
                            '''将历史最新数据赋值给msg'''
                            if len(garbage_history_list_) > 0:
                                msg.pose.pose.position.x = garbage_history_list_[-1][0]
                                msg.pose.pose.position.y = garbage_history_list_[-1][1]
                                msg.class_id = int(garbage_history_list_[-1][2])
                            else:
                                msg.pose.pose.position.x = 0.0
                                msg.pose.pose.position.y = 0.0
                                msg.class_id = -1
                                
                                
                            if self.action_request_goal is not None and self.action_request_goal.pose.pose.position.x > 2.2 and not self.is_approach_garbage:
                                self.get_logger().info('接近垃圾......', throttle_duration_sec=2)
                                if msg.pose.pose.position.x == 0.0:
                                    if self.approach_garbage_start_time is None:
                                        '''获取机器人当前位姿'''
                                        try:
                                            base_link_g = self.tf_buffer.lookup_transform('map', 'base_link',
                                                                                          rclpy.time.Time.from_msg(
                                                                                              self.action_request_goal.pose.header.stamp),
                                                                                          rclpy.duration.Duration(
                                                                                              seconds=0.5))
                                            _, _, base_link_history_angle = self.quart_to_rpy(
                                                base_link_g.transform.rotation.x, base_link_g.transform.rotation.y,
                                                base_link_g.transform.rotation.z, base_link_g.transform.rotation.w)
                                            _, _, base_link_now_angle = self.quart_to_rpy(base_link_r.x, base_link_r.y,
                                                                                          base_link_r.z, base_link_r.w)
                                            angle_now_history = self.angle_diff(base_link_history_angle,
                                                                                base_link_now_angle, False)
                                        except Exception as ex:
                                            angle_now_history = 0.0
                                        # 获取垃圾相对于机器人的角度和距离
                                        # self.get_logger().info(f'angle_now_history: {angle_now_history}',throttle_duration_sec=2)
                                        _, _, self.approach_garbage_angle = self.quart_to_rpy(
                                            self.action_request_goal.pose.pose.orientation.x,
                                            self.action_request_goal.pose.pose.orientation.y,
                                            self.action_request_goal.pose.pose.orientation.z,
                                            self.action_request_goal.pose.pose.orientation.w)
                                        self.approach_garbage_angle = self.approach_garbage_angle + angle_now_history  # 把垃圾的姿态修正为与机器人相同姿态
                                        self.approach_garbage_dist = self.action_request_goal.pose.pose.position.x
                                        self.approach_garbage_start_time = time.time()
                                    elif time.time() - self.receive_signal_time > 5.5:
                                        self.is_exit = True
                                        self.get_logger().info('接近垃圾超时。', throttle_duration_sec=2)

                                else:
                                    _, _, self.approach_garbage_angle = self.quart_to_rpy(msg.pose.pose.orientation.x,
                                                                                          msg.pose.pose.orientation.y,
                                                                                          msg.pose.pose.orientation.z,
                                                                                          msg.pose.pose.orientation.w)
                                    self.approach_garbage_dist = msg.pose.pose.position.x
                                    self.approach_garbage_start_time = time.time()
                                self.get_logger().info(
                                    f'angle:{self.approach_garbage_angle}, dist:{self.approach_garbage_dist}。',
                                    throttle_duration_sec=1)

                                # 旋转
                                if self.approach_garbage_angle > 10 / 180 * math.pi or self.approach_garbage_angle < -10 / 180 * math.pi:
                                    self.get_logger().info(f'接近垃圾旋转:{self.approach_garbage_angle}。',throttle_duration_sec=2)
                                    self.get_logger().info(f'现在机器人和垃圾的角度是: {self.approach_garbage_angle:.4f} rad ({self.approach_garbage_angle * 180 / math.pi:.2f}°)', throttle_duration_sec=2)
                                    whirl_speed = math.pi / 20
                                    whirl_time = abs(self.approach_garbage_angle) / whirl_speed
                                    if time.time() - self.approach_garbage_start_time < whirl_time:
                                        if self.approach_garbage_angle > 10 / 180 * math.pi:
                                            whirl_control_msg = Twist()
                                            whirl_control_msg.angular.z = whirl_speed
                                            self.whirl_control_pub.publish(whirl_control_msg)
                                        elif self.approach_garbage_angle < -(10 / 180 * math.pi):
                                            whirl_control_msg = Twist()
                                            whirl_control_msg.angular.z = -whirl_speed
                                            self.whirl_control_pub.publish(whirl_control_msg)
                                    else:
                                        self.get_logger().info('接近垃圾超时。', throttle_duration_sec=2)

                                # 前进
                                elif self.approach_garbage_dist > 2.3: 
                                # elif self.approach_garbage_dist > 2.1:     
                                    self.get_logger().info(f'接近垃圾前进:{self.approach_garbage_dist}。',throttle_duration_sec=2)
                                    forward_speed = 0.25
                                    forward_time = self.approach_garbage_dist / forward_speed
                                    if time.time() - self.approach_garbage_start_time < forward_time:
                                        whirl_control_msg = Twist()
                                        whirl_control_msg.linear.x = forward_speed
                                        self.whirl_control_pub.publish(whirl_control_msg)
                                        # self.rate_.sleep()
                                    else:
                                        # self.is_approach_garbage = True
                                        self.get_logger().info('接近垃圾超时。', throttle_duration_sec=2)

                                else:
                                    self.is_approach_garbage = True
                                    recorder_garbage_map_goal = self.base_link_pose_to_map(tf2_map_base_link,
                                                                                           msg.pose.pose.position.x,
                                                                                           msg.pose.pose.position.y)
                                    recorder_garbage_map_x, recorder_garbage_map_y = recorder_garbage_map_goal.pose.position.x, recorder_garbage_map_goal.pose.position.y
                                    self.garbage_recorder = [recorder_garbage_map_x, recorder_garbage_map_y,
                                                             msg.class_id]
       
                                    for i in range(10):
                                        whirl_control_msg = Twist()
                                        whirl_control_msg.linear.x = 0.0
                                        self.whirl_control_pub.publish(whirl_control_msg)
                                        self.rate_.sleep()
                                    self.get_logger().info('接近垃圾完成。', throttle_duration_sec=2)

                            elif (msg.class_id != -1 or len(self.garbage_position_list) > 0) and not self.goal_accept:
                                # self.get_logger().info('-----receive valid data-----',once=True)
                                garbage_dist = 999.9
                                if not self.is_confirm_garbage_location:
                                    if self.garbage_recorder is None:
                                        recorder_garbage_map_goal = self.base_link_pose_to_map(tf2_map_base_link,
                                                                                               msg.pose.pose.position.x,
                                                                                               msg.pose.pose.position.y)
                                        recorder_garbage_map_x, recorder_garbage_map_y = recorder_garbage_map_goal.pose.position.x, recorder_garbage_map_goal.pose.position.y
                                        self.garbage_recorder = [recorder_garbage_map_x, recorder_garbage_map_y,
                                                                 msg.class_id]
                                        
          
                                    current_garbage_map_goal = self.base_link_pose_to_map(tf2_map_base_link,
                                                                                          msg.pose.pose.position.x,
                                                                                          msg.pose.pose.position.y)
                                    current_garbage_map_x, current_garbage_map_y = current_garbage_map_goal.pose.position.x, current_garbage_map_goal.pose.position.y
                                    garbage_dist = math.dist([current_garbage_map_x, current_garbage_map_y],
                                                             [self.garbage_recorder[0], self.garbage_recorder[1]])

                                if msg.class_id != -1 and garbage_dist < 1.0 and self.garbage_recorder[
                                    -1] == msg.class_id and not self.is_confirm_garbage_location:
                                    self.garbage_recorder = [current_garbage_map_x, current_garbage_map_y, msg.class_id]
  
                                    self.last_valid_data = msg.pose.pose

                                    self.receive_valid_data_time = time.time()
                                    self.whirl_speed = max(self.whirl_speed / 3, math.pi / 30)
                                    garbage_to_robot_angle = np.arctan(
                                        msg.pose.pose.position.y / msg.pose.pose.position.x)
                                
                                    self.work_status = 'work'
                                    if garbage_to_robot_angle > 4 / 180 * math.pi:
                                        self.get_logger().info(f'=====================================rotating......', throttle_duration_sec=5)
                                        whirl_control_msg = Twist()
                                        # whirl_control_msg.angular.z = math.pi / 45
                                        whirl_control_msg.angular.z = 0.075
                                        self.whirl_control_pub.publish(whirl_control_msg)

                                    elif garbage_to_robot_angle < - 4 / 180 * math.pi:
                                        self.get_logger().info(f'=====================================rotating......', throttle_duration_sec=5)
                                        whirl_control_msg = Twist()
                                        whirl_control_msg.angular.z = -0.075
                                        # whirl_control_msg.angular.z = -math.pi / 45
                                        self.whirl_control_pub.publish(whirl_control_msg)

                                    else:
                                        self.get_logger().info(f'旋转完成，现在垃圾与机器人的角度差: {garbage_to_robot_angle:.4f} rad ({garbage_to_robot_angle * 180 / math.pi:.2f}°)', throttle_duration_sec=2)
                                        if self.garbage_position_list_add_time is None:
                                            self.garbage_position_list_add_time = time.time()
                                            self.garbage_position_list.append(
                                                [msg.pose.pose.position.x, msg.pose.pose.position.y,
                                                 garbage_to_robot_angle, msg.class_id])

                                        elif time.time() - self.garbage_position_list_add_time < 1.5:
                                            self.garbage_position_list.append(
                                                [msg.pose.pose.position.x, msg.pose.pose.position.y,
                                                 garbage_to_robot_angle, msg.class_id])

                                        elif len(self.garbage_position_list) > 0:
                                            self.is_confirm_garbage_location = True
                                            self.get_logger().info('garbage position is confirm.')
                                        else:
                                            self.get_logger().info('garbage_position_list is None exit.')
                                            self.is_exit = True
                                elif self.is_confirm_garbage_location:
                                    if self.garbage_positions is None:
                                        self.get_logger().info(
                                            f'len(garbage_position_list)-->{len(self.garbage_position_list)}',
                                            throttle_duration_sec=5)
                                        garbage_position_list_ = np.array(self.garbage_position_list)
                                        garbage_x_base, garbage_y_base = np.median(
                                            garbage_position_list_[garbage_position_list_[:, 0] > 0][:, :2], axis=0)
                                        garbage_to_robot_angle = math.atan(garbage_y_base / garbage_x_base)

                                        id_unique = np.unique(garbage_position_list_[:, 3], return_counts=True)
                                        c_id = id_unique[0][np.argmax(id_unique[1])]
                                        if c_id == 1:
                                            self.clean_mode = 1
                                            goal_addition = self.goal_addition / 2
                                        elif c_id == 2:
                                            self.clean_mode = 2
                                            goal_addition = self.goal_addition / 4
                                        else:
                                            self.clean_mode = None

                                        # 都开
                                        if self.clean_mode == 0:
                                            if not self.control_cleaning_tools_is_pub:
                                                self._control_cleaning_tools(vacuum_control_msg=True,
                                                                             rolling_brush_control_msg=True,
                                                                             suction_sewage_control_msg=True,
                                                                             clear_water_control_msg=True)
                                                self.control_cleaning_tools_is_pub = True

                                        elif self.clean_mode == 1:  # 湿垃圾
                                            if not self.control_cleaning_tools_is_pub:
                                                self._control_cleaning_tools(vacuum_control_msg=False,
                                                                             rolling_brush_control_msg=True,
                                                                             suction_sewage_control_msg=True,
                                                                             clear_water_control_msg=True)
                                                self.control_cleaning_tools_is_pub = True

                                        elif self.clean_mode == 2:  # 干垃圾
                                            if not self.control_cleaning_tools_is_pub:
                                                self._control_cleaning_tools(vacuum_control_msg=True,
                                                                             rolling_brush_control_msg=False,
                                                                             suction_sewage_control_msg=False,
                                                                             clear_water_control_msg=False)
                                                self.control_cleaning_tools_is_pub = True

                                        else:
                                            self.get_logger().info(f'unknown garbage class: {msg.class_id}')
                                            self.is_exit = True
                                        '''-------------------------------------路径规划-----------------------------'''
                                        if (self.clean_mode == 0 and self.clean_tools_state['roll_pushrod'] and
                                            self.clean_tools_state['roll_motor'] and self.clean_tools_state[
                                                'vacuum'] and self.clean_tools_state['suct_pushrod'] and
                                            self.clean_tools_state['suct_motor']) \
                                                or (self.clean_mode == 1 and self.clean_tools_state['suct_pushrod'] and
                                                    self.clean_tools_state['suct_motor'] and self.clean_tools_state[
                                                        'roll_pushrod'] and self.clean_tools_state['roll_motor']) \
                                                or (self.clean_mode == 2 and self.clean_tools_state['vacuum']):
                                            if self.start_clean_point == None:
                                                self.start_clean_point = [base_link_x, base_link_y, yaw]
                                            self.get_logger().info('start clean', )  # throttle_duration_sec=5)
                                            send_goal_msg = NavigateThroughPoses.Goal()
                                            # 行为树
                                            send_goal_msg.behavior_tree = '/capella/share/capella_ros_launcher/behavior_trees/navigate_through_poses_w_replanning_and_recovery_clean_garbage.xml'
                                            # 垃圾附近没有障碍物
                                            garbage_map_goal = self.base_link_pose_to_map(tf2_map_base_link,
                                                                                          garbage_x_base,
                                                                                          garbage_y_base)
                                            garbage_map_x, garbage_map_y = garbage_map_goal.pose.position.x, garbage_map_goal.pose.position.y
                                            obstacle_map = self._get_obstacle_map(garbage_map_x, garbage_map_y,
                                                                                  self.garbage_inflation_radius)
                                            # have obstacle
                                            if not (obstacle_map > 0).all():
                                                map_where_is_zero = np.argwhere(obstacle_map == 0)
                                                points_dist = np.sqrt(np.sum(np.power(np.array([[int(
                                                    obstacle_map.shape[0] / 2), int(
                                                    obstacle_map.shape[1] / 2)]]) - map_where_is_zero, 2), axis=-1))
                                                min_distance_obstacle = map_where_is_zero[
                                                    points_dist == np.min(points_dist)]
                                                #
                                                if len(min_distance_obstacle) == 1:
                                                    # self.get_logger().info(f'obstacle = 1')
                                                    min_obstacle_pixel = min_distance_obstacle[0]
                                                    # pixel to m
                                                    min_obstacle_x = garbage_map_x + (min_obstacle_pixel[1] - int(
                                                        obstacle_map.shape[1] / 2)) * self.map_resolution
                                                    min_obstacle_y = garbage_map_y - (min_obstacle_pixel[0] - int(
                                                        obstacle_map.shape[0] / 2)) * self.map_resolution
                                                    y_gap = min_obstacle_y - garbage_map_y
                                                    x_gap = min_obstacle_x - garbage_map_x
                                                    if x_gap == 0.0:
                                                        k = np.inf
                                                    else:
                                                        k = y_gap / x_gap
                                                    k = np.arctan(-(1 / k) if k != 0.0 else np.inf)
                                                    garbage_obstacle_dist = math.dist([garbage_map_x, garbage_map_y],
                                                                                      [min_obstacle_x, min_obstacle_y])

                                                else:
                                                    # find min max
                                                    x_min = min_distance_obstacle[
                                                        np.argmin(min_distance_obstacle[:, 0])]
                                                    x_max = min_distance_obstacle[
                                                        np.argmax(min_distance_obstacle[:, 0])]
                                                    y_min = min_distance_obstacle[
                                                        np.argmin(min_distance_obstacle[:, 1])]
                                                    y_max = min_distance_obstacle[
                                                        np.argmax(min_distance_obstacle[:, 1])]
                                                    min_distance_obstacle = np.array([x_min, x_max, y_min, y_max])
                                                    dist = -1.0
                                                    points = []
                                                    # 和下面的for循环一样的功能
                                                    for i in range(len(min_distance_obstacle)):
                                                        for j in range(i + 1, len(min_distance_obstacle)):
                                                            dist_temp = np.sqrt(np.sum(np.power(
                                                                min_distance_obstacle[i] - min_distance_obstacle[j],
                                                                2)))
                                                            if dist < dist_temp:
                                                                dist = dist_temp
                                                                points = [i, j]

                                                    point1 = min_distance_obstacle[points[0]]
                                                    point2 = min_distance_obstacle[points[1]]
                                                    # pixel to m
                                                    point1_obstacle_x = garbage_map_x + (point1[1] - int(
                                                        obstacle_map.shape[1] / 2)) * self.map_resolution
                                                    point1_obstacle_y = garbage_map_y - (point1[0] - int(
                                                        obstacle_map.shape[0] / 2)) * self.map_resolution
                                                    point2_obstacle_x = garbage_map_x + (point2[1] - int(
                                                        obstacle_map.shape[1] / 2)) * self.map_resolution
                                                    point2_obstacle_y = garbage_map_y - (point2[0] - int(
                                                        obstacle_map.shape[0] / 2)) * self.map_resolution
                                                    # ax+by+c=0
                                                    a = point2_obstacle_y - point1_obstacle_y
                                                    b = point1_obstacle_x - point2_obstacle_x
                                                    c = point2_obstacle_x * point1_obstacle_y - point1_obstacle_x * point2_obstacle_y
                                                    # point to line dist
                                                    garbage_obstacle_dist = abs(
                                                        a * garbage_map_x + b * garbage_map_y + c) / math.sqrt(
                                                        a ** 2 + b ** 2)
                                                    # k
                                                    y_gap = point2_obstacle_y - point1_obstacle_y
                                                    x_gap = point2_obstacle_x - point1_obstacle_x
                                                    if x_gap == 0.0:
                                                        k = np.inf
                                                    else:
                                                        k = y_gap / x_gap
                                                    k = np.arctan(k)

                                                self.get_logger().info(
                                                    f'garbage_obstacle_dist: {garbage_obstacle_dist}')
                                                if garbage_obstacle_dist > self.min_gap_to_clean:
                                                    # 筛选k
                                                    k_pi = k - math.pi if k > 0 else k + math.pi
                                                    if self.angle_diff(k, yaw) > self.angle_diff(k_pi, yaw):
                                                        k = k_pi
                                                    garbage_map_x0 = garbage_map_x - math.cos(k) * self.goal_cut
                                                    garbage_map_y0 = garbage_map_y - math.sin(k) * self.goal_cut
                                                    garbage_map_x1 = garbage_map_x + math.cos(k) * goal_addition
                                                    garbage_map_y1 = garbage_map_y + math.sin(k) * goal_addition
                                                    if self.angle_diff(k, yaw) >= self.goal_num_angle:
                                                        if self.goal_num == 4:
                                                            # 第四个点
                                                            k001 = k + math.pi / 2
                                                            if k001 > math.pi:
                                                                k001 = -math.pi - (math.pi - k001)

                                                            k002 = k001 - math.pi if k001 > 0 else k001 + math.pi

                                                            k00 = k001 if self.angle_diff(k001, yaw) < self.angle_diff(
                                                                k002, yaw) else k002

                                                            garbage_map_x00 = garbage_map_x0 - math.cos(
                                                                k00) * self.goal_cut
                                                            garbage_map_y00 = garbage_map_y0 - math.sin(
                                                                k00) * self.goal_cut

                                                            point_list = [[garbage_map_x0, garbage_map_y0, k],
                                                                          [garbage_map_x, garbage_map_y, k],
                                                                          [garbage_map_x1, garbage_map_y1, k]]
                                                        else:
                                                            point_list = [[garbage_map_x, garbage_map_y, k],
                                                                          [garbage_map_x1, garbage_map_y1, k]]
                                                    else:
                                                        point_list = [[garbage_map_x, garbage_map_y, k],
                                                                      [garbage_map_x1, garbage_map_y1, k]]

                                                    last_point_x, last_point_y, last_point_k = self.optimize_goal(
                                                        *point_list[-1])

                                                    point_list[-1] = [last_point_x, last_point_y, last_point_k]
                                                    last_goal_to_garbage_dist = math.dist(
                                                        [point_list[-2][0], point_list[-2][1]],
                                                        [point_list[-1][0], point_list[-1][1]])
                                                    last_goal_to_garbage_angle_diff = self.angle_diff(
                                                        point_list[-1][-1], point_list[-2][-1])

                                                    if last_goal_to_garbage_dist < 0.9 and last_goal_to_garbage_angle_diff > 10 / 180 * math.pi:
                                                        point_list[-1][0] = point_list[-1][0] + math.cos(
                                                            point_list[-1][-1]) * (
                                                                                        goal_addition - last_goal_to_garbage_dist + (
                                                                                            1.3 - last_goal_to_garbage_dist))
                                                        point_list[-1][1] = point_list[-1][1] + math.sin(
                                                            point_list[-1][-1]) * (
                                                                                        goal_addition - last_goal_to_garbage_dist + (
                                                                                            1.3 - last_goal_to_garbage_dist))

                                                    for goal in point_list:
                                                        goal1 = PoseStamped()
                                                        goal1.header.frame_id = 'map'
                                                        goal1.header.stamp = self.get_clock().now().to_msg()
                                                        goal1.pose.position.x = goal[0]
                                                        goal1.pose.position.y = goal[1]
                                                        q = self.rpy_to_quart(0.0, 0.0, goal[2])
                                                        goal1.pose.orientation.x = q[0]
                                                        goal1.pose.orientation.y = q[1]
                                                        goal1.pose.orientation.z = q[2]
                                                        goal1.pose.orientation.w = q[3]

                                                        send_goal_msg.poses.append(goal1)

                                                else:
                                                    self.is_exit = True
                                                    self.get_logger().info('too close to the obstacle,can not clear.',
                                                                           throttle_duration_sec=1)
                                                    self.get_logger().info(
                                                        f'garbage_base_x_y: {garbage_x_base},{garbage_y_base}',
                                                        throttle_duration_sec=1)
                                                    self.get_logger().info(
                                                        f'garbage_map_x_y: {garbage_map_x},{garbage_map_y}',
                                                        throttle_duration_sec=1)
                                                    if len(min_distance_obstacle) == 1:
                                                        self.get_logger().info(
                                                            f'min_obstacle_x_y: {min_obstacle_x},{min_obstacle_y}',
                                                            throttle_duration_sec=1)

                                                    else:
                                                        self.get_logger().info(
                                                            f'min_obstacle_x_y 1: {point1_obstacle_x},{point1_obstacle_y}',
                                                            throttle_duration_sec=1)
                                                        self.get_logger().info(
                                                            f'min_obstacle_x_y 2: {point2_obstacle_x},{point2_obstacle_y}',
                                                            throttle_duration_sec=1)

                                                # break

                                            # no obstacle
                                            else:
                                                goal_1 = self.base_link_pose_to_map(tf2_map_base_link, garbage_x_base,
                                                                                    garbage_y_base,
                                                                                    yaw=garbage_to_robot_angle)
                                                goal_2 = self.base_link_pose_to_map(tf2_map_base_link,
                                                                                    garbage_x_base + goal_addition,
                                                                                    garbage_y_base,
                                                                                    yaw=garbage_to_robot_angle)
                                                _, _, goal_2_k = self.quart_to_rpy(goal_2.pose.orientation.x,
                                                                                   goal_2.pose.orientation.y,
                                                                                   goal_2.pose.orientation.z,
                                                                                   goal_2.pose.orientation.w)
                                                last_point_x, last_point_y, last_point_k = self.optimize_goal(
                                                    goal_2.pose.position.x, goal_2.pose.position.y, goal_2_k)

                                                goal_2.pose.position.x = last_point_x
                                                goal_2.pose.position.y = last_point_y
                                                goal_2.pose.orientation.x, goal_2.pose.orientation.y, goal_2.pose.orientation.z, goal_2.pose.orientation.w = self.rpy_to_quart(
                                                    0.0, 0.0, last_point_k)
                                                last_goal_to_garbage_dist = math.dist([last_point_x, last_point_y],
                                                                                      [goal_1.pose.position.x,
                                                                                       goal_1.pose.position.y])
                                                last_goal_to_garbage_angle_diff = self.angle_diff(last_point_k,
                                                                                                  self.quart_to_rpy(
                                                                                                      goal_1.pose.orientation.x,
                                                                                                      goal_1.pose.orientation.y,
                                                                                                      goal_1.pose.orientation.z,
                                                                                                      goal_1.pose.orientation.w)[
                                                                                                      -1])
                                                self.get_logger().info(
                                                    f'last_goal_to_garbage_dist addition dist: {last_goal_to_garbage_dist}, angle_diff{last_goal_to_garbage_angle_diff}',
                                                    throttle_duration_sec=1)
                                                if last_goal_to_garbage_dist < 0.9 and last_goal_to_garbage_angle_diff > 10 / 180 * math.pi:
                                                    goal_2.pose.position.x = goal_2.pose.position.x + math.cos(
                                                        last_point_k) * (goal_addition - last_goal_to_garbage_dist + (
                                                                1.3 - last_goal_to_garbage_dist))
                                                    goal_2.pose.position.y = goal_2.pose.position.y + math.sin(
                                                        last_point_k) * (goal_addition - last_goal_to_garbage_dist + (
                                                                1.3 - last_goal_to_garbage_dist))

                                                send_goal_msg.poses.append(goal_1)
                                                send_goal_msg.poses.append(goal_2)

                                            if len(send_goal_msg.poses) > 0:
                                                self.garbage_positions = send_goal_msg

                                                # log all target goal coordinates for this garbage
                                                points = [(p.pose.position.x, p.pose.position.y) for p in send_goal_msg.poses]
                                                self.logger.info(f"垃圾目标点设置：{points}")

                                                goal_test_msg = PoseArray()
                                                goal_test_msg.header.frame_id = 'map'
                                                goal_test_msg.header.stamp = self.get_clock().now().to_msg()
                                                for i in send_goal_msg.poses:
                                                    goal_test_msg.poses.append(i.pose)
                                                self.goal_test_pub.publish(goal_test_msg)
                                                self.send_goal_future = self.send_goal_action.send_goal_async(
                                                    self.garbage_positions, feedback_callback=self.feedback_callback)
                                                self.send_goal_future.add_done_callback(self.goal_response_callback)
                                            else:
                                                self.get_logger().info('no goal exit.')
                                                self.is_exit = True
                                                self.start_clean_signal = False
                                        else:
                                            if self.open_clean_tools_time is None:
                                                self.open_clean_tools_time = None
                                            elif time.time() > self.open_clean_tools_time > 30:
                                                self.is_exit = True
                                                self.get_logger().info('打开清洁工具超时，即将退出。。。。。。')

                                    else:
                                        self.get_logger().info('request navigate_to_pose action.',
                                                               throttle_duration_sec=5)
                                        self.send_goal_future = self.send_goal_action.send_goal_async(
                                            self.garbage_positions, feedback_callback=self.feedback_callback)
                                        self.send_goal_future.add_done_callback(self.goal_response_callback)

                                else:
                                    # 连续10s没检测到垃圾退出
                                    if time.time() - self.receive_valid_data_time > 10:
                                        self.get_logger().info('检测到的垃圾不匹配，即将退出。。。。。。', throttle_duration_sec=1)
                                        self.is_exit = True

                            elif (self.action_request_goal is not None) and (self.last_valid_data is not None) and (
                            not self.goal_accept):
                                self.get_logger().info('-----backward-----', once=True)
                                if time.time() - self.receive_valid_data_time > 10.0 or self.has_backward:
                                    self.get_logger().info('未检测到垃圾，即将退出。。。。。。', throttle_duration_sec=1)
                                    self.is_exit = True
                                else:
                                    # if self.can_it_rotate and self.can_it_backward:
                                    self.get_logger().info(
                                        f'开始后退。。。。。。,whirl_speed: {self.whirl_speed}, backward_speed: {self.backward_speed}',
                                        once=True)
                                    r, p, last_garbage_angle = self.quart_to_rpy(self.last_valid_data.orientation.x,
                                                                                 self.last_valid_data.orientation.y,
                                                                                 self.last_valid_data.orientation.z,
                                                                                 self.last_valid_data.orientation.w)
                                    last_garbage_distance = self.backward_distance
                                    # 转正
                                    whirl_time = abs(last_garbage_angle) / self.whirl_speed
                                    start_time = time.time()
                                    while time.time() - start_time < whirl_time:
                                        whirl_control_msg = Twist()
                                        whirl_control_msg.angular.z = self.whirl_speed
                                        self.whirl_control_pub.publish(whirl_control_msg)
                                        self.rate_.sleep()
                                    # 后退
                                    back_time = last_garbage_distance / self.backward_speed
                                    start_time = time.time()
                                    while time.time() - start_time < back_time:
                                        whirl_control_msg = Twist()
                                        whirl_control_msg.linear.x = -self.backward_speed
                                        self.whirl_control_pub.publish(whirl_control_msg)
                                        self.rate_.sleep()

                                    self.has_backward = True
                                    self.get_logger().info(
                                        f'can_it_rotate:{self.can_it_rotate}, can_it_backward:{self.can_it_backward}')


                            elif self.work_status == 'idle':
                                self.get_logger().info('-----self.work_status == "idle"-----', once=True)
                                if self.action_request_goal is None:
                                    self.get_logger().info('searching garbage......', once=True)
                                    whirl_control_msg = Twist()
                                    whirl_control_msg.angular.z = self.whirl_speed
                                    self.whirl_control_pub.publish(whirl_control_msg)
                                else:
                                    if time.time() - self.receive_valid_data_time > 5.0:
                                        self.is_exit = True
                                        self.get_logger().info('no garbage,exit......')


                            else:
                                self.get_logger().info('-----else-----', once=True)
                                if self.reached_goal_time is not None:
                                    self.get_logger().info('reached goal,wait......', once=True)
                                    if (time.time() - self.reached_goal_time > 0.5 and self.clean_mode == 1) or (
                                            time.time() - self.reached_goal_time > 10.0 and self.clean_mode == 2):
                                        self.get_logger().info('clean complete.')
                                        self.is_exit = True

                                        self._control_cleaning_tools(vacuum_control_msg=False,
                                                                     rolling_brush_control_msg=False,
                                                                     clear_water_control_msg=False)
                                        self.clean_success = True

                                        while self.clean_tools_state['vacuum'] or self.clean_tools_state[
                                            'roll_pushrod'] or self.clean_tools_state['roll_motor']:
                                            time.sleep(0.5)
                                            self.get_logger().info('closing clean tools.', once=True)

                                elif not self.goal_accept:
                                    self.work_status = 'idle'
                        else:
                            if time.time() - self.receive_tf2_time > 10:
                                self.get_logger().info('no map to base_link tf2 exit.')
                                self.is_exit = True
                            else:
                                self.get_logger().info('no map to base_link tf2.', throttle_duration_sec=3)
                    else:
                        if time.time() - self.receive_signal_time > 10:
                            self.get_logger().info('map is None exit.')
                            self.is_exit = True
                        else:
                            self.get_logger().info('map is None', throttle_duration_sec=3)
                else:
                    if time.time() - self.can_not_rorate_time > 30:
                        self.get_logger().info('can not rotate exit.')
                        self.is_exit = True
                    else:
                        self.get_logger().info('can not rotate.', throttle_duration_sec=5)
            else:
                if self.manual_control_signal:
                    self.get_logger().info('manual control', throttle_duration_sec=5)
                else:
                    self.get_logger().info('manual control exit', throttle_duration_sec=5)
                    self.is_exit = True
        else:
            self.get_logger().info('等待开始清洁指令......', throttle_duration_sec=10)

    def device_status_sub_callback(self, msg):
        """设备状态回调"""
        self.clean_tools_state = {
            'vacuum': msg.vacuum_cleaner_stu,
            'roll_pushrod': msg.roll_pushrod_stu,
            'roll_motor': msg.roll_motor_stu,
            'suct_pushrod': msg.suct_pushrod_stu,
            'suct_motor': msg.suct_motor_stu,

            # 'rolling_brush': msg.roll_cleaner_stu,
            # 'suction_sewage': msg.suct_water_cleaner_stu
        }

    def angle_diff(self, a, b, use_abs=True):
        error = a - b
        if error < -math.pi:
            error = error + 2 * math.pi  # 小于-pi加2pi
        elif error >= math.pi:
            error = 2 * math.pi - error  # 大于pi减2pi
        else:
            pass
        if use_abs:
            return abs(error)
        else:
            return error

    def quart_to_rpy(self, x, y, z, w):
        r = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        p = math.asin(2 * (w * y - x * z))
        y = math.atan2(2 * (w * z + x * y), 1 - 2 * (z * z + y * y))

        return r, p, y

    def rpy_to_quart(self, r, p, y):
        sinp = math.sin(p / 2)
        siny = math.sin(y / 2)
        sinr = math.sin(r / 2)

        cosp = math.cos(p / 2)
        cosy = math.cos(y / 2)
        cosr = math.cos(r / 2)

        w = cosr * cosp * cosy + sinr * sinp * siny
        x = sinr * cosp * cosy - cosr * sinp * siny
        y = cosr * sinp * cosy + sinr * cosp * siny
        z = cosr * cosp * siny - sinr * sinp * cosy
        return [x, y, z, w]

    def convert_tf_coordinate(self, x, y, angle):
        map_x = (math.cos(angle) * x) - (math.sin(angle) * (y))
        map_y = (math.sin(angle) * x) + (math.cos(angle) * (y))

        return map_x, map_y

    def base_link_pose_to_map(self, tf2_map_base_link, x, y, z=0.0, roll=0.0, pitch=0.0, yaw=0.0, ):
        goal_template = PoseStamped()
        goal_template.header.frame_id = 'base_link'
        goal_template.header.stamp = self.get_clock().now().to_msg()
        goal_template.pose.position.x = x
        goal_template.pose.position.y = y
        goal_template.pose.position.z = z
        angle_quart = self.rpy_to_quart(roll, pitch, yaw)
        goal_template.pose.orientation.x = angle_quart[0]
        goal_template.pose.orientation.y = angle_quart[1]
        goal_template.pose.orientation.z = angle_quart[2]
        goal_template.pose.orientation.w = angle_quart[3]
        goal1 = do_transform_pose_stamped(goal_template, tf2_map_base_link)
        return goal1

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.feedback_number_of_poses_remaining = feedback.number_of_poses_remaining
        self.feedback_distance_remaining = feedback.distance_remaining
        if self.feedback_number_of_poses_remaining == 1:
            if self.clean_tools_state['roll_pushrod'] and self.clean_tools_state['roll_motor']:
                time.sleep(3.5)
                self._control_cleaning_tools(rolling_brush_control_msg=False)
                self.get_logger().info('closing rolling brush', once=True)

        self.get_logger().info(f'feedback number_of_poses_remaining: {self.feedback_number_of_poses_remaining}',
                               throttle_duration_sec=1)
        self.get_logger().info(f'feedback distance_remaining: {self.feedback_distance_remaining}',
                               throttle_duration_sec=1)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.goal_accept = False
            self.get_logger().info('request navigate_through_poses goal fail :(')
        else:
            self.get_logger().info('request navigate_through_poses goal success :)')
            if self.action_goal_timeout is None:
                self.action_goal_timeout = time.time()
            self.goal_accept = True
            self.goal_handle = goal_handle
            get_result_future = goal_handle.get_result_async()
            get_result_future.add_done_callback(self.get_result_callback)

    def cancel_done_callback(self, future):
        cancel_response = future.result()
        if len(cancel_response.goals_canceling) > 0:
            self.get_logger().info('goal cancel successfully :)')
        else:
            self.get_logger().info('goal cancel failed :(')

    def action_server_callback(self, goal_handle):
        self.get_logger().info(f'receive clean garbage signal.,frame_id:{goal_handle.request.pose.header.frame_id}')
        if goal_handle.request.pose.header.frame_id == 'base_link':
            self.action_request_goal = goal_handle.request
            self.last_valid_data = self.action_request_goal.pose.pose
        request = ClearEntireCostmap.Request()
        self.clear_cost_map_future = self.clear_lcost_map_client.call_async(request)
        self.clear_cost_map_future.add_done_callback(self.clear_lcost_map_callback)
        self.clear_cost_map_future = self.clear_gcost_map_client.call_async(request)
        self.clear_cost_map_future.add_done_callback(self.clear_gcost_map_callback)

        self._reset_variable(is_action_reset=True)
        
       # =======================================================================
        self.is_init_blocking = True
        self.action_callback_init_time = time.time()
        
        
        #self.get_logger().info('4秒阻塞......')
        

        #blocking_duration =4.0
        blocking_duration = 0.01
        # start_time = time.time()sss
        # while time.time() - start_time < blocking_duration:
        #     elapsed_time = time.time() - start_time
        #     remaining_time = blocking_duration - elapsed_time
        #     time.sleep(0.1)
        time.sleep(blocking_duration)
        self.is_init_blocking = False
        # =======================================================================

        feedback_msg = SearchGarbage.Feedback()
        goal_handle.publish_feedback(feedback_msg)

        while True:
            if goal_handle.is_cancel_requested:
                self.get_logger().info('exit: receive cancel signal.')
                break
            if self.is_exit == True:
                self.get_logger().info('exit: is_exit is True.')
                break
            if ((
                    time.time() - self.receive_signal_time > 60 and self.manual_control_signal is None) and self.work_status == 'idle'):
                self.get_logger().info('exit: receive_signal_time > 60.')
                break
            if (self.action_success_sign):
                self.get_logger().info('exit:  Recieve action_success_sign .')
                break
            if (time.time() - self.receive_signal_time > 180):
                self.get_logger().info('exit: receive_signal_time > 180.')
                break
            if self.action_goal_timeout is not None:
                
                if time.time() - self.action_goal_timeout > 15:
                    self.get_logger().info('exit: action_goal_timeout > 15.')
                    break                
                
                # if time.time() - self.action_goal_timeout > 20:
                #     self.get_logger().info('exit: action_goal_timeout > 20.')
                #     break
            time.sleep(0.5)
        try:
            # self.get_logger().info(f'---------->clean all time: {time.time() - self.action_goal_timeout}')
            self.send_goal_future.result().cancel_goal_async()
        except Exception as ex:
            pass

        self._reset_variable()
        self.action_success_sign = False

        self._control_cleaning_tools(vacuum_control_msg=False, rolling_brush_control_msg=False,
                                     clear_water_control_msg=False)

        while self.clean_tools_state['vacuum'] or self.clean_tools_state['roll_pushrod'] or self.clean_tools_state[
            'roll_motor']:  # or self.clean_tools_state['suction_sewage']:
            time.sleep(0.5)
            self.get_logger().info('closing clean tools', once=True)

        result = SearchGarbage.Result()
        if self.clean_success:
            goal_handle.succeed()
        else:
            goal_handle.abort()
        self.get_logger().info('exit clean program.')

        return result

    def action_goal_callback(self, handle):
        self.get_logger().info('receive goal callback signal.')
        self.work_status = 'idle'
        return GoalResponse.ACCEPT

    def action_cancel_callback(self, handle):
        self.get_logger().info('receive cancel signal.')
        self.is_exit = True
        try:
            self.send_goal_future.result().cancel_goal_async()
        except Exception as ex:
            pass
        return CancelResponse.ACCEPT

    def get_result_callback(self, future):
        result = future.result().result
        if self.feedback_number_of_poses_remaining <= 1.0:
            self.reached_goal_time = time.time()
            self.get_logger().info(f'receive action result: {result}')
            self.get_logger().info('导航目标成功完成')
        else:
            self.get_logger().info('导航目标未完成')

    def manual_control_callback(self, msg):
        self.manual_control_signal = msg.data
        self.get_logger().info(f'receive manual_control_signal msg: {msg.data}.', throttle_duration_sec=1)

    def _control_cleaning_tools(self, vacuum_control_msg=None, rolling_brush_control_msg=None,
                                suction_sewage_control_msg=None, clear_water_control_msg=None):
        if vacuum_control_msg is not None:  # 干垃圾
            msg = Bool()
            msg.data = vacuum_control_msg
            self.vacuum_cleaner_control_pub.publish(msg)
            self.get_logger().info(f'control vacuum: {vacuum_control_msg}', throttle_duration_sec=1)
        if rolling_brush_control_msg is not None:  # 湿垃圾
            msg = Bool()
            msg.data = rolling_brush_control_msg
            self.rolling_brush_pushrod_control_pub.publish(msg)
            self.rolling_brush_motor_control_pub.publish(msg)
            self.get_logger().info(f'control rolling_brush: {rolling_brush_control_msg}', throttle_duration_sec=1)
        if suction_sewage_control_msg is not None:  # 湿垃圾
            msg = Bool()
            msg.data = suction_sewage_control_msg
            # self.suction_sewage_control_pub.publish(msg)
            self.suction_sewage_pushrod_control_pub.publish(msg)
            self.suction_sewage_motor_control_pub.publish(msg)
            self.get_logger().info(f'control suction_sewage: {suction_sewage_control_msg}', throttle_duration_sec=1)
        if clear_water_control_msg is not None:
            msg = Bool()
            msg.data = clear_water_control_msg
            self.clear_water_control_pub.publish(msg)
            self.get_logger().info(f'control clear water: {clear_water_control_msg}', throttle_duration_sec=1)

    def local_costmap_sub_callback(self, msg):
        if self.start_clean_signal:
            self.get_logger().info('receive local_cost_map msg.', once=True)
            # whirl
            map = np.array(msg.data)
            map[map == -1] = 50
            map = (100 - map) / 100 * 255
            map = map.reshape((msg.info.height, msg.info.width)).astype(np.uint8)[::-1]
            local_cost_map_mask = np.zeros_like(map)
            robot_radius_pixel = int(self.robot_radius // msg.info.resolution)
            cv2.circle(local_cost_map_mask, (int(msg.info.width // 2), int(msg.info.height // 2)), robot_radius_pixel,
                       (255, 255, 255), -1)
            local_cost_map_mask_inverse = 255 - local_cost_map_mask
            obstacle_map = cv2.bitwise_and(map, local_cost_map_mask) + local_cost_map_mask_inverse
            self.can_it_rotate = (obstacle_map > 0).all()
            if self.can_it_rotate:
                self.can_not_rorate_time = time.time()
            # backward
            local_cost_map_mask = np.zeros_like(map)
            backward_distance_pixel = int(self.backward_distance // msg.info.resolution)
            cv2.circle(local_cost_map_mask, (int(msg.info.width // 2), int(msg.info.height // 2)),
                       backward_distance_pixel, (255, 255, 255), -1)
            local_cost_map_mask_inverse = 255 - local_cost_map_mask
            obstacle_map = cv2.bitwise_and(map, local_cost_map_mask) + local_cost_map_mask_inverse
            self.can_it_backward = (obstacle_map > 0).all()

    def map_sub_callback(self, msg):
        # if self.map is None:
        if self.start_clean_signal:
            self.get_logger().info('receive map msg.', throttle_duration_sec=3)
            map = np.array(msg.data)
            map[map == -1] = 50
            map = (100 - map) / 100 * 255
            map[map != 0] = 255
            self.map = np.ascontiguousarray(map.reshape((msg.info.height, msg.info.width)).astype(np.uint8)[::-1])
            self.origin = [msg.info.origin.position.x, msg.info.origin.position.y]

    def clear_lcost_map_callback(self, future):
        response = future.result()
        response = response
        self.get_logger().info(f'clear local cost map')

    def clear_gcost_map_callback(self, future):
        response = future.result()
        response = response
        self.get_logger().info(f'clear global cost map')

    def _reset_variable(self, is_action_reset=False):
        if is_action_reset:
            # variable
            self.receive_signal_time = time.time()
            self.receive_valid_data_time = time.time()
            self.can_not_rorate_time = time.time()
            self.receive_tf2_time = time.time()
            self.can_it_rotate = False
            self.can_it_backward = False
            self.has_backward = False
            self.garbage_recorder = None
            self.start_clean_point = None
            self.work_status = 'idle'
            self.garbage_positions = None
            self.goal_accept = False
            self.feedback_number_of_poses_remaining = 999
            self.feedback_distance_remaining = 999
            self.clean_success = False
            self.reached_goal_time = None
            self.open_clean_tools_time = None
            self.is_confirm_garbage_location = False
            self.is_approach_garbage = False
            self.approach_garbage_start_time = None
            self.approach_garbage_angle = 0.0
            self.approach_garbage_dist = 0.0
            self.approach_garbage_has_whirl = False
            self.approach_garbage_has_forward = False
            self.manual_control_signal = None
            self.action_goal_timeout = None
            
        # =======================================================================    
            self.is_init_blocking = False
            self.action_callback_init_time = None
        # =======================================================================
            self.garbage_position_list.clear()
            self.garbage_history_list.clear()
            self.whirl_speed = math.pi / 10
            self.vacuum_cleaner_control_is_pub = False
            self.rolling_brush_control_is_pub = False
            self.suction_sewage_control_is_pub = False
            self.start_clean_signal = True

        else:
            self.start_clean_signal = False
            time.sleep(1.0)
            self.garbage_position_list = deque(maxlen=5)
            self.garbage_history_list = deque(maxlen=5)
            self.is_exit = False
            self.control_cleaning_tools_is_pub = False
            self.clean_tools_state = {
                'vacuum': 0,
                'roll_pushrod': 0,
                'roll_motor': 0,
                'suct_pushrod': 0,
                'suct_motor': 0,
                'clear_water': 0
            }
            self.clean_mode = None
            self.map = None
            self.origin = []
            self.action_request_goal = None
            self.last_valid_data = None
            self.garbage_position_list_add_time = None

            # variable
            self.receive_signal_time = time.time()
            self.receive_valid_data_time = time.time()
            self.can_not_rorate_time = time.time()
            self.receive_tf2_time = time.time()
            self.can_it_rotate = False
            self.can_it_backward = False
            self.has_backward = False
            self.garbage_recorder = None
            self.start_clean_point = None
            self.work_status = 'idle'
            self.garbage_positions = None
            self.goal_accept = False
            self.feedback_number_of_poses_remaining = 999
            self.feedback_distance_remaining = 999
            self.clean_success = False
            self.reached_goal_time = None
            self.open_clean_tools_time = None
            self.is_confirm_garbage_location = False
            self.is_approach_garbage = False
            self.approach_garbage_start_time = None
            self.approach_garbage_angle = 0.0
            self.approach_garbage_dist = 0.0
            self.approach_garbage_has_whirl = False
            self.approach_garbage_has_forward = False
            self.manual_control_signal = None
            self.action_goal_timeout = None
            # =======================================================================
            self.is_init_blocking = False
            self.action_callback_init_time = None
            # =======================================================================


    def optimize_goal(self, x, y, k):
        self.get_logger().info(f'optimize end goal.')
        last_point_x, last_point_y, last_point_k = x, y, k
        obstacle_map = self._get_obstacle_map(last_point_x, last_point_y)
        if not (obstacle_map > 0).all():
            self.get_logger().info(f'find obstacle.')
            # 找到离垃圾最近的障碍物，这个要快一点
            map_where_is_zero = np.argwhere(obstacle_map == 0)
            points_dist = np.sqrt(np.sum(np.power(
                np.array([[int(obstacle_map.shape[0] / 2), int(obstacle_map.shape[1] / 2)]]) - map_where_is_zero, 2),
                                         axis=-1))
            min_distance_obstacle = map_where_is_zero[points_dist == np.min(points_dist)]
            #
            if len(min_distance_obstacle) == 1:
                # self.get_logger().info(f'obstacle = 1')
                min_obstacle_pixel = min_distance_obstacle[0]
                # pixel to m
                min_obstacle_x = last_point_x + (
                            min_obstacle_pixel[1] - int(obstacle_map.shape[1] / 2)) * self.map_resolution
                min_obstacle_y = last_point_y - (
                            min_obstacle_pixel[0] - int(obstacle_map.shape[0] / 2)) * self.map_resolution
                y_gap = min_obstacle_y - last_point_y
                x_gap = min_obstacle_x - last_point_x
                if x_gap == 0.0:
                    k = np.inf
                else:
                    k = y_gap / x_gap
                k = np.arctan(-(1 / k) if k != 0.0 else np.inf)
                last_point_obstacle_dist = math.dist([last_point_x, last_point_y], [min_obstacle_x, min_obstacle_y])
                k_pi = k - math.pi if k > 0 else k + math.pi
                if self.angle_diff(k, last_point_k) > self.angle_diff(k_pi, last_point_k):
                    k = k_pi
                kk1 = k + math.pi / 2
                if kk1 > math.pi:
                    kk1 = kk1 - (math.pi * 2)
                last1_point_x = last_point_x - math.cos(kk1) * abs(1.3 - last_point_obstacle_dist)
                last1_point_y = last_point_y - math.sin(kk1) * abs(1.3 - last_point_obstacle_dist)

                last2_point_x = last_point_x + math.cos(kk1) * abs(1.3 - last_point_obstacle_dist)
                last2_point_y = last_point_y + math.sin(kk1) * abs(1.3 - last_point_obstacle_dist)

                last_point_obstacle_dist1 = math.dist([last1_point_x, last1_point_y], [min_obstacle_x, min_obstacle_y])
                last_point_obstacle_dist2 = math.dist([last2_point_x, last2_point_y], [min_obstacle_x, min_obstacle_y])
                if last_point_obstacle_dist1 > last_point_obstacle_dist2:
                    last_point_x = last1_point_x
                    last_point_y = last1_point_y
                    last_point_k = k

                else:
                    last_point_x = last2_point_x
                    last_point_y = last2_point_y
                    last_point_k = k


            else:
                # find min max
                x_min = min_distance_obstacle[np.argmin(min_distance_obstacle[:, 0])]
                x_max = min_distance_obstacle[np.argmax(min_distance_obstacle[:, 0])]
                y_min = min_distance_obstacle[np.argmin(min_distance_obstacle[:, 1])]
                y_max = min_distance_obstacle[np.argmax(min_distance_obstacle[:, 1])]
                min_distance_obstacle = np.array([x_min, x_max, y_min, y_max])
                dist = -1.0
                points = []
                # 和下面的for循环一样的功能
                for i in range(len(min_distance_obstacle)):
                    for j in range(i + 1, len(min_distance_obstacle)):
                        dist_temp = np.sqrt(np.sum(np.power(min_distance_obstacle[i] - min_distance_obstacle[j], 2)))
                        if dist < dist_temp:
                            dist = dist_temp
                            points = [i, j]

                point1 = min_distance_obstacle[points[0]]
                point2 = min_distance_obstacle[points[1]]
                # pixel to m
                point1_obstacle_x = last_point_x + (point1[1] - int(obstacle_map.shape[1] / 2)) * self.map_resolution
                point1_obstacle_y = last_point_y - (point1[0] - int(obstacle_map.shape[0] / 2)) * self.map_resolution
                point2_obstacle_x = last_point_x + (point2[1] - int(obstacle_map.shape[1] / 2)) * self.map_resolution
                point2_obstacle_y = last_point_y - (point2[0] - int(obstacle_map.shape[0] / 2)) * self.map_resolution
                # ax+by+c=0
                a = point2_obstacle_y - point1_obstacle_y
                b = point1_obstacle_x - point2_obstacle_x
                c = point2_obstacle_x * point1_obstacle_y - point1_obstacle_x * point2_obstacle_y
                # point to line dist
                last_point_obstacle_dist = abs(a * last_point_x + b * last_point_y + c) / math.sqrt(a ** 2 + b ** 2)
                # k
                y_gap = point2_obstacle_y - point1_obstacle_y
                x_gap = point2_obstacle_x - point1_obstacle_x
                # self.get_logger().info(f'{point1_obstacle_x,point1_obstacle_y,point2_obstacle_x,point2_obstacle_y}')
                if x_gap == 0.0:
                    k = np.inf
                else:
                    k = y_gap / x_gap
                k = np.arctan(k)
                # 筛选k
                k_pi = k - math.pi if k > 0 else k + math.pi
                if self.angle_diff(k, last_point_k) > self.angle_diff(k_pi, last_point_k):
                    k = k_pi
                kk1 = k + math.pi / 2
                if kk1 > math.pi:
                    kk1 = kk1 - (math.pi * 2)
                last1_point_x = last_point_x - math.cos(kk1) * abs(1.3 - last_point_obstacle_dist)
                last1_point_y = last_point_y - math.sin(kk1) * abs(1.3 - last_point_obstacle_dist)

                last2_point_x = last_point_x + math.cos(kk1) * abs(1.3 - last_point_obstacle_dist)
                last2_point_y = last_point_y + math.sin(kk1) * abs(1.3 - last_point_obstacle_dist)
                # point to line dist
                last_point_obstacle_dist1 = abs(a * last1_point_x + b * last1_point_y + c) / math.sqrt(a ** 2 + b ** 2)
                last_point_obstacle_dist2 = abs(a * last2_point_x + b * last2_point_y + c) / math.sqrt(a ** 2 + b ** 2)
                if last_point_obstacle_dist1 > last_point_obstacle_dist2:
                    last_point_x = last1_point_x
                    last_point_y = last1_point_y
                    last_point_k = k

                else:
                    last_point_x = last2_point_x
                    last_point_y = last2_point_y
                    last_point_k = k
        return last_point_x, last_point_y, last_point_k

    def _get_obstacle_map(self, x, y, obstacle_r=1.3):
        self.get_logger().info(f'optimize end goal.')
        last_point_inflation_radius_pixel = int(obstacle_r / self.map_resolution)
        last_point_x, last_point_y = x, y
        last_point_picture_left_bottom_x = last_point_x - self.origin[0]
        last_point_picture_left_bottom_y = last_point_y - self.origin[1]
        last_point_picture_x_pixel = last_point_picture_left_bottom_x / self.map_resolution
        last_point_picture_y_pixel = self.map.shape[0] - (last_point_picture_left_bottom_y / self.map_resolution)
        y1 = int(last_point_picture_y_pixel - last_point_inflation_radius_pixel)
        y2 = int(last_point_picture_y_pixel + last_point_inflation_radius_pixel)
        x1 = int(last_point_picture_x_pixel - last_point_inflation_radius_pixel)
        x2 = int(last_point_picture_x_pixel + last_point_inflation_radius_pixel)
        map_last_point = self.map[y1:y2, x1:x2]

        map_mask = np.zeros_like(map_last_point)
        cv2.circle(map_mask, (int(map_last_point.shape[1] / 2), int(map_last_point.shape[0] / 2)),
                   last_point_inflation_radius_pixel, (255, 255, 255), -1)
        map_mask_inverse = 255 - map_mask
        obstacle_map = cv2.bitwise_and(map_last_point, map_mask) + map_mask_inverse

        return obstacle_map
    def clean_action_succeed_callback(self, msg):
        self.action_success_sign = msg.data


def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = CleanGarbage()
    executor_ = MultiThreadedExecutor()
    rclpy.spin(minimal_subscriber, executor=executor_)

    minimal_subscriber.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()