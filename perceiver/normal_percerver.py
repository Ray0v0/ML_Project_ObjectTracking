import numpy as np
from dataclasses import dataclass
from typing import Optional
import cv2
from ultralytics import YOLO
import time
from carla import Location, Rotation, Transform

@dataclass
class PerceptionInfo:
    """感知信息数据类，存储处理后的感知结果"""
    # 前车信息
    lead_vehicle_pose: Optional[np.ndarray] = None  # 前车位置姿态 [x, y, heading]
    lead_vehicle_velocity: Optional[np.ndarray] = None  # 前车速度 [vx, vy]
    # 车辆位置信息
    pose_to_follow: Optional[object] = None  # 前车位置信息
    velocity_to_follow: Optional[object] = None  # 前车速度信息
    pose_follow: Optional[object] = None  # 后车位置信息
    velocity_follow: Optional[object] = None  # 后车速度信息
    # 相对信息
    relative_distance: Optional[float] = None  # 与前车的相对距离
    relative_heading: Optional[float] = None  # 与前车的相对朝向

class NormalPerceiver:
    """标准感知器类"""
    
    def __init__(self):
        """初始化感知器"""
        self.info = PerceptionInfo()
        # 加载预训练的YOLOv8模型
        self.yolo_model = YOLO('yolov8n.pt')
        
        # 更新相机参数以提高准确性
        self.camera_matrix = np.array([
            [1000, 0, 960],
            [0, 1000, 540],
            [0, 0, 1]
        ])
        
        # 添加状态估计相关变量
        self.prev_position = None
        self.prev_timestamp = None
        self.position_history = []  # 添加位置历史记录
        self.velocity_history = []  # 添加速度历史记录
        self.history_max_size = 5   # 历史记录最大长度
        self.min_detection_confidence = 0.6  # 提高检测置信度阈值

    def update_state_history(self, position, velocity):
        """更新状态历史记录"""
        self.position_history.append(position)
        self.velocity_history.append(velocity)
        
        # 保持历史记录在指定长度内
        if len(self.position_history) > self.history_max_size:
            self.position_history.pop(0)
        if len(self.velocity_history) > self.history_max_size:
            self.velocity_history.pop(0)

    def get_smoothed_state(self):
        """获取平滑后的状态估计"""
        if not self.position_history or not self.velocity_history:
            return None, None
            
        # 使用加权平均进行平滑
        weights = np.exp(np.linspace(-1, 0, len(self.position_history)))
        weights = weights / np.sum(weights)
        
        smoothed_position = np.average(self.position_history, axis=0, weights=weights)
        smoothed_velocity = np.average(self.velocity_history, axis=0, weights=weights)
        
        return smoothed_position, smoothed_velocity

    def process_camera_input(self, camera_image):
        """处理CARLA相机输入，只检测最相关的一辆车"""
        if camera_image is None:
            return

        try:
            image = camera_image.copy()
            results = self.yolo_model(image, conf=self.min_detection_confidence)
            
            if len(results) > 0:
                result = results[0]
                boxes = result.boxes
                
                # 只关注车辆类别
                vehicle_classes = [2]  # 只检测轿车类别
                
                # 找到最相关的一辆车
                best_vehicle = None
                best_score = -1
                
                for box in boxes:
                    if int(box.cls) in vehicle_classes:
                        box_data = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf)
                        
                        # 计算框的中心点和面积
                        center_x = (box_data[0] + box_data[2]) / 2
                        center_y = (box_data[1] + box_data[3]) / 2
                        box_area = (box_data[2] - box_data[0]) * (box_data[3] - box_data[1])
                        
                        # 计算到图像中心的距离
                        image_center_x = image.shape[1] / 2
                        image_center_y = image.shape[0] / 2
                        center_dist = np.sqrt((center_x - image_center_x)**2 + (center_y - image_center_y)**2)
                        
                        # 评分标准：优先选择图像中心、较大且置信度高的车辆
                        score = confidence * (1 - center_dist/1000) * (box_area / (image.shape[0] * image.shape[1]))
                        
                        if score > best_score:
                            best_score = score
                            best_vehicle = box_data
                
                # 如果找到合适的车辆，处理其信息
                if best_vehicle is not None:
                    box = best_vehicle
                    # 改进的距离估算
                    box_height = box[3] - box[1]
                    box_width = box[2] - box[0]
                    aspect_ratio = box_width / box_height
                    
                    # 考虑实际车辆尺寸和透视效果
                    VEHICLE_HEIGHT = 1.5  # 米
                    VEHICLE_WIDTH = 1.8   # 米
                    
                    # 使用高度和宽度的组合来估算距离
                    distance_by_height = (VEHICLE_HEIGHT * self.camera_matrix[1,1]) / box_height
                    distance_by_width = (VEHICLE_WIDTH * self.camera_matrix[0,0]) / box_width
                    relative_distance = (distance_by_height + distance_by_width) / 2
                    
                    # 计算相对位置
                    box_center_x = (box[0] + box[2]) / 2
                    image_center_x = image.shape[1] / 2
                    
                    # 修改横向偏移计算
                    # 图像坐标系：左上角为原点，向右为正
                    # 车辆坐标系：前方为x轴正方向，左侧为y轴正方向
                    normalized_offset = (box_center_x - image_center_x) / image_center_x
                    
                    # 关键修改：反转符号！
                    # 当目标在图像右侧时(normalized_offset > 0)，y应该为负
                    # 当目标在图像左侧时(normalized_offset < 0)，y应该为正
                    lateral_offset = -normalized_offset * relative_distance * 0.05
                    
                    # 设置前车位置
                    x = relative_distance
                    y = lateral_offset  # 现在符号是正确的了
                    heading = 0.0
                    
                    # 更新感知信息
                    self.info.lead_vehicle_pose = np.array([x, y, heading])
                    
                    # 更新相对距离和朝向信息
                    self.info.relative_distance = relative_distance
                    self.info.relative_heading = heading
                    
                    # 打印调试信息
                    print(f"Perception Debug:")
                    print(f"Box center: {box_center_x:.1f}, Image center: {image_center_x:.1f}")
                    print(f"Normalized offset: {normalized_offset:.3f}")
                    print(f"Forward distance: {relative_distance:.2f}m")
                    print(f"Lateral offset: {lateral_offset:.2f}m")
                    print(f"Final pose: x={x:.2f}m, y={y:.2f}m, heading={np.degrees(heading):.1f}°")
                    
                    # 增强的可视化
                    if hasattr(self, 'debug_visualization') and self.debug_visualization:
                        self._draw_debug_info(image, box, relative_distance, heading)
                    
        except Exception as e:
            print(f"处理相机图像时出错: {str(e)}")

    def _draw_debug_info(self, image, box, distance, heading):
        """绘制调试信息"""
        cv2.rectangle(image, 
                    (int(box[0]), int(box[1])), 
                    (int(box[2]), int(box[3])), 
                    (0, 255, 0), 2)
        
        info_text = [
            f'Distance: {distance:.2f}m',
            f'Heading: {np.degrees(heading):.1f}deg',
        ]
        
        y_offset = int(box[1] - 10)
        for text in info_text:
            cv2.putText(image, text, 
                      (int(box[0]), y_offset), 
                      cv2.FONT_HERSHEY_SIMPLEX, 
                      0.9, (0, 255, 0), 2)
            y_offset -= 25
            
        cv2.imshow('Vehicle Detection', image)
        cv2.waitKey(1)

    def process_map_info(self, map_data):
        """处理地图信息
        Args:
            map_data: 地图数据
        """
        # TODO: 实现地图信息处理逻辑
        pass
    
    def process_ego_state(self, pose, velocity):
        """处理自车状态
        Args:
            pose: 自车位置姿态 (carla.Transform)
            velocity: 自车速度 (carla.Location)
        """
        if pose is None or velocity is None:
            return
            
        try:
            # 将前车位置转换到全局坐标系
            if self.info.lead_vehicle_pose is not None:
                # 从 Transform 对象获取正确的位置和朝向
                ego_location = pose.location
                ego_rotation = pose.rotation
                ego_heading = np.radians(ego_rotation.yaw)  # CARLA使用度数，需要转换为弧度
                
                relative_x = self.info.lead_vehicle_pose[0]
                relative_y = self.info.lead_vehicle_pose[1]
                
                # 坐标转换
                global_x = ego_location.x + relative_x * np.cos(ego_heading) - relative_y * np.sin(ego_heading)
                global_y = ego_location.y + relative_x * np.sin(ego_heading) + relative_y * np.cos(ego_heading)
                global_heading = ego_heading + self.info.lead_vehicle_pose[2]
                
                # 更新前车全局位置
                self.info.lead_vehicle_pose = np.array([global_x, global_y, global_heading])
                
                # 估算前车速度
                current_position = np.array([global_x, global_y])
                current_time = time.time()
                
                if self.prev_position is not None and self.prev_timestamp is not None:
                    dt = current_time - self.prev_timestamp
                    if dt > 0:
                        velocity_estimate = (current_position - self.prev_position) / dt
                        self.info.lead_vehicle_velocity = velocity_estimate
                
                self.prev_position = current_position
                self.prev_timestamp = current_time
                
        except Exception as e:
            print(f"处理自车状态时出错: {str(e)}")
    
    def perceive(self, velocity_follow=None, pose_follow=None, 
                velocity_to_follow=None, pose_to_follow=None, map=None, camera_image=None):
        """主要感知处理函数"""
        try:
            # 重置前一帧的信息
            self.info = PerceptionInfo()
            
            # # 如果有真实的前车信息，优先使用
            # if pose_to_follow is not None:
            #     self.info.pose_to_follow = pose_to_follow
            #     self.info.velocity_to_follow = velocity_to_follow
            
            # 存储后车数据
            self.info.pose_follow = pose_follow
            self.info.velocity_follow = velocity_follow
            
            # 处理相机输入获取前车信息
            if camera_image is not None:
                self.process_camera_input(camera_image)
                
                # 如果通过相机检测到前车，且没有真实前车信息，则使用检测结果
                if self.info.lead_vehicle_pose is not None and pose_to_follow is None:
                    x, y, heading = self.info.lead_vehicle_pose
                    location = Location(x=x, y=y, z=0)
                    rotation = Rotation(yaw=np.degrees(heading))
                    self.info.pose_to_follow = Transform(location, rotation)
                    
                    if self.info.lead_vehicle_velocity is not None:
                        vx, vy = self.info.lead_vehicle_velocity
                        self.info.velocity_to_follow = Location(x=vx, y=vy, z=0)
            
            # 处理地图信息
            self.process_map_info(map)
            
            # 处理自车状态
            if pose_follow is not None and velocity_follow is not None:
                self.process_ego_state(pose_follow, velocity_follow)
            
            return self.info
            
        except Exception as e:
            print(f"感知处理出错: {str(e)}")
            return PerceptionInfo()
