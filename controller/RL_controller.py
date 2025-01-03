import torch
import numpy as np
import carla
from controller.DDPG import DDPG
from manager.pose_manager import PoseManager
from collections import deque
from manager.vec3d_utils import get_magnitude, get_angle  # 假设这个方法用来计算向量的模长和夹角


class RLController:
    def __init__(self, state_dim, action_dim, action_bound, gamma=0.99, lr_a=0.0001, lr_c=0.001, tau=0.001,
                 batch_size=64, memory_capacity=10000, pretrain=False, daf_controller=None):
        # 初始化 DDPG 控制器，传入必要的参数
        self.ddpg = DDPG(state_dim, action_dim, action_bound, replacement={'name': 'soft', 'tau': tau},
                         memory_capacity=memory_capacity, gamma=gamma, lr_a=lr_a, lr_c=lr_c, batch_size=batch_size)
        self.memory_capacity = memory_capacity
        self.batch_size = batch_size

        # 如果 pretrain 为 True，则进行预训练
        if pretrain and daf_controller:
            self.pretrain(daf_controller)

    def pretrain(self, daf_controller):
        """预训练阶段，使用 DAFController 进行基本的跟车训练"""
        print("开始预训练...")
        for _ in range(1000):  # 训练1000步，或者根据需要的次数调整
            # 获取前后车的状态信息
            state = self.get_initial_state()
            action = daf_controller.predict_control(state)
            next_state, reward, done = self.step(action)  # 执行动作
            # 存储经验
            self.store_transition(state, action, reward, next_state)
            # 进行预训练
            self.learn()
            if done:
                break
        print("预训练结束")

    def store_transition(self, state, action, reward, next_state):
        """将状态、动作、奖励和下一个状态存储到经验回放池"""
        self.ddpg.store_transition(state, action, reward, next_state)

    def learn(self):
        """调用 DDPG 算法的学习过程"""
        self.ddpg.learn()

    def calculate_reward(self, info):
        """计算奖励函数"""
        reward = 0

        # 获取相对距离、速度和航向信息
        relative_distance = self.calculate_relative_distance(info)
        relative_velocity = self.calculate_relative_velocity(info)
        relative_heading = self.calculate_relative_heading(info)

        # 奖励逻辑，可以根据相对距离、速度等调整奖励策略
        if 5.0 <= relative_distance <= 10.0:
            reward += 1
        elif relative_distance < 5.0:
            reward -= 5
        else:
            reward -= 1

        if abs(relative_velocity) > 10.0:
            reward -= 10

        # 可以根据其他策略进一步调整奖励
        return reward

    def predict_control(self, info):
        """根据当前状态通过 DDPG 控制器预测后车控制"""
        state = np.array([
            self.calculate_relative_distance(info),  # 相对距离
            self.calculate_relative_velocity(info),  # 相对速度
            self.calculate_relative_heading(info)  # 相对航向
        ])

        # 使用 DDPG 控制器预测动作
        action = self.ddpg.choose_action(state)

        # 返回后车控制指令
        return carla.VehicleControl(throttle=action[0], brake=action[1], steer=action[2])

    def save_model(self, episode):
        """保存训练好的模型"""
        self.ddpg.save_model(episode)

    def load_model(self, actor_path, critic_path):
        """加载保存的模型"""
        self.ddpg.load_model(actor_path, critic_path)

    def calculate_relative_distance(self, info):
        """计算前车与后车之间的相对距离"""
        return PoseManager.get_distance(info.pose_follow, info.pose_to_follow)

    def calculate_relative_velocity(self, info):
        """计算前车和后车之间的相对速度"""
        velocity_follow = np.array([info.velocity_follow.x, info.velocity_follow.y, info.velocity_follow.z])
        velocity_to_follow = np.array([info.velocity_to_follow.x, info.velocity_to_follow.y, info.velocity_to_follow.z])
        return np.linalg.norm(velocity_follow - velocity_to_follow)

    def calculate_relative_heading(self, info):
        """计算前车和后车之间的相对航向角"""
        vec_to_follow = carla.Vector3D(info.pose_to_follow.location.x - info.pose_follow.location.x,
                                       info.pose_to_follow.location.y - info.pose_follow.location.y,
                                       info.pose_to_follow.location.z - info.pose_follow.location.z)
        vec_forward = carla.Vector3D(info.pose_follow.location.x - info.pose_follow.location.x,
                                     info.pose_follow.location.y - info.pose_follow.location.y,
                                     info.pose_follow.location.z - info.pose_follow.location.z)
        return get_angle(vec_to_follow, vec_forward)

    def get_initial_state(self):
        """获取初始状态"""
        return np.array([0, 0, 0])  # 这里返回一个示例状态

    def step(self, action,info ):
        """模拟与环境的交互"""
        # 1. 执行动作：控制车辆
        vehicle_follow_control = carla.VehicleControl(throttle=action[0], brake=action[1], steer=action[2])
        vehicle_follow.apply_control(vehicle_follow_control)

        # 2. 获取新的位置和速度
        pose_follow_new = vehicle_follow.get_transform()
        velocity_follow_new = vehicle_follow.get_velocity()

        # 3. 计算新的状态
        next_state = np.array([
            self.calculate_relative_distance(info_follow),
            self.calculate_relative_velocity(info_follow),
            self.calculate_relative_heading(info_follow)
        ])

        # 4. 计算奖励
        reward = self.calculate_reward(info_follow)

        # 5. 判断是否完成任务
        done = False
        if pose_follow_new.location.distance(pose_to_follow.location) < 5:
            done = True  # 如果后车与前车足够接近，结束本次任务

        # 6. 返回新的状态、奖励和是否结束
        return next_state, reward, done
