import numpy as np
import carla
from controller.traditional_controller import TraditionalController
from dataclasses import dataclass

@dataclass
class ControllerParams:
    """控制器参数配置"""
    # 距离控制参数
    desired_distance: float = 10.0  # 减小期望跟车距离
    max_throttle: float = 0.7  # 降低最大油门
    max_brake: float = 1.0
    safe_distance: float = 5.0
    
    # 速度控制参数
    pid_velocity: dict = None
    
    # 转向控制参数
    max_steer_angle: float = 70.0
    pid_heading: dict = None
    
    def __post_init__(self):
        # 调整PID参数
        if self.pid_velocity is None:
            self.pid_velocity = {
                'Kp': 0.3,  # 降低比例系数
                'Ki': 0.05,  # 降低积分系数
                'Kd': 0.2   # 增加微分系数
            }
        if self.pid_heading is None:
            self.pid_heading = {
                'Kp': 0.3,  # 大幅降低转向的比例系数
                'Ki': 0.0,  # 移除积分项，避免累积误差
                'Kd': 0.1
            }
class NormalController(TraditionalController):
    """与NormalPerceiver配套的控制器"""
    
    def __init__(self, params: ControllerParams = None):
        """初始化控制器
        Args:
            params: 控制器参数
        """
        self.params = params if params else ControllerParams()
        
        # 初始化PID控制器状态
        self.velocity_error_sum = 0
        self.last_velocity_error = 0
        self.heading_error_sum = 0
        self.last_heading_error = 0
        
        # 控制输出
        self.throttle = 0
        self.brake = 0
        self.steer = 0
        
        # 调试信息
        self.debug_info = {}

    def predict_control(self, info) -> carla.VehicleControl:
        """预测控制指令
        Args:
            info: 感知器提供的信息
        Returns:
            carla.VehicleControl: 控制指令
        """
        if info.relative_distance is None:
            # 如果没有检测到前车，则停车
            return self._emergency_stop()
            
        # 计算纵向控制
        self._longitudinal_control(info)
        
        # 计算横向控制
        self._lateral_control(info)
        
        # 打印调试信息
        self._print_debug_info()
        
        return carla.VehicleControl(
            throttle=self.throttle,
            steer=self.steer,
            brake=self.brake
        )

    # 修改横向控制方法
    def _lateral_control(self, info):
        """横向控制：控制车辆转向"""
        if info.relative_heading is None:
            self.steer = 0
            return
            
        # 计算目标航向角
        if info.lead_vehicle_pose is not None:
            # 使用反正切计算期望航向角
            dx = info.relative_distance
            dy = info.lead_vehicle_pose[1]  # 横向偏移
            target_heading = np.arctan2(dy, dx)
            
            # 将弧度转换为度数
            heading_error = np.degrees(target_heading)
            
            # 添加死区，忽略小角度误差
            if abs(heading_error) < 2.0:
                heading_error = 0
                
            # PID控制
            p_term = self.params.pid_heading['Kp'] * heading_error
            # 移除积分项，避免累积误差
            d_term = self.params.pid_heading['Kd'] * (heading_error - self.last_heading_error)
            self.last_heading_error = heading_error
            
            # 计算转向控制
            steer_control = p_term + d_term
            
            # 平滑转向输出
            if abs(steer_control) < 0.1:
                steer_control = 0
                
            # 限制转向角度并添加平滑
            target_steer = np.clip(steer_control / self.params.max_steer_angle, -1.0, 1.0)
            self.steer = self.steer * 0.8 + target_steer * 0.2  # 添加平滑
            
            # 更新调试信息
            self.debug_info.update({
                'heading_error': heading_error,
                'target_heading': np.degrees(target_heading),
                'steer_control': steer_control
            })

    # 修改纵向控制方法
    def _longitudinal_control(self, info):
        """纵向控制：控制车辆加速和制动"""
        # 计算距离误差
        distance_error = info.relative_distance - self.params.desired_distance
        
        # 估计相对速度
        if hasattr(self, 'last_distance'):
            relative_velocity = (info.relative_distance - self.last_distance) * self.get_fps()
            # 添加速度平滑
            if hasattr(self, 'last_relative_velocity'):
                relative_velocity = 0.7 * self.last_relative_velocity + 0.3 * relative_velocity
            self.last_relative_velocity = relative_velocity
        else:
            relative_velocity = 0
        self.last_distance = info.relative_distance
        
        # 基于距离和速度的组合控制
        safe_distance_factor = max(0, min(1, (info.relative_distance - self.params.safe_distance) / 
                                        (self.params.desired_distance - self.params.safe_distance)))
        
        # 如果距离小于安全距离，立即制动
        if info.relative_distance < self.params.safe_distance:
            self.throttle = 0
            self.brake = self.params.max_brake
            return
            
        # 计算期望速度
        target_speed = relative_velocity * safe_distance_factor
        
        # PID控制
        p_term = self.params.pid_velocity['Kp'] * distance_error
        self.velocity_error_sum = np.clip(self.velocity_error_sum + distance_error, -10, 10)  # 限制积分项
        i_term = self.params.pid_velocity['Ki'] * self.velocity_error_sum
        d_term = self.params.pid_velocity['Kd'] * relative_velocity
        
        # 计算控制输出
        control = p_term + i_term - d_term  # 注意这里减去d_term，因为正的相对速度应该降低控制输出
        
        # 平滑控制输出
        if abs(control) < 0.1:
            control = 0
        
        # 确定油门和刹车
        if control > 0:
            self.throttle = min(control * safe_distance_factor, self.params.max_throttle)
            self.brake = 0
        else:
            self.throttle = 0
            self.brake = min(-control, self.params.max_brake)
            
    def _emergency_stop(self):
        """紧急停车"""
        self.throttle = 0
        self.brake = 1.0
        self.steer = 0
        return carla.VehicleControl(throttle=0, steer=0, brake=1.0)

    def _print_debug_info(self):
        """打印调试信息"""
        print("\r", end="")
        print(f"距离误差: {self.debug_info.get('distance_error', 0):.2f}m | ", end="")
        print(f"相对速度: {self.debug_info.get('relative_velocity', 0):.2f}m/s | ", end="")
        print(f"航向误差: {self.debug_info.get('heading_error', 0):.2f}° | ", end="")
        print(f"油门: {self.throttle:.2f} | 刹车: {self.brake:.2f} | 转向: {self.steer:.2f}", end="")