# 导入carla包
import sys
import glob
import os
import time
import traceback
import numpy as np
import cv2  # 在文件开头添加


#TODO: 改成自己的路径
try:
    sys.path.append(glob.glob('D:/Carla_0.9.8/WindowsNoEditor/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import random
from manager.sync_carla_manager import SyncCarlaManager
from manager.pose_manager import PoseManager
from manager.display_manager import DisplayManager
from controller.DDPG import DDPG
from controller.daf_controller import DAFController
from controller.carla_auto_pilot import CarlaAutoPilot
from controller.path_follower import PathFollower
from controller.manual_controller import ManualController
from controller.follow_track_controller import FollowTrackController
from controller.normal_controller import NormalController
from controller.RL_controller import RLController
from perceiver.god_perceiver import GodPerceiver
from perceiver.blind_perceiver import BlindPerceiver
from perceiver.normal_percerver import NormalPerceiver

def start(controller_to_follow, controller_follow, perceiver_to_follow, perceiver_follow):
    actor_list = []

    try:
        client = carla.Client('127.0.0.1', 2000)
        client.set_timeout(2.0)
        world = client.get_world()
        blueprint_library = world.get_blueprint_library()

        # 生成前车
        pose_manager_to_follow = PoseManager()
        pose_to_follow = random.choice(world.get_map().get_spawn_points())
        bp_to_follow = blueprint_library.filter('model3')[0]
        vehicle_to_follow = world.spawn_actor(bp_to_follow, pose_to_follow)
        vehicle_to_follow.set_simulate_physics(True)
        actor_list.append(vehicle_to_follow)

        # 前车控制
        if isinstance(controller_to_follow, PathFollower):
            pose_manager_to_follow.load_history_from_file('path', controller_to_follow.file)
            pose_to_follow = pose_manager_to_follow.get_car_pose(0)
            vehicle_to_follow.set_transform(pose_to_follow)

        # 生成后车
        pose_follow = PoseManager.create_pose_in_front_of(pose_to_follow, -5, 0.1)
        bp_follow = blueprint_library.filter('jeep')[0]
        vehicle_follow = world.spawn_actor(bp_follow, pose_follow)
        actor_list.append(vehicle_follow)

        # 生成后车传感器
        bp_camera_rgb = blueprint_library.find('sensor.camera.rgb')
        camera_rgb = world.spawn_actor(bp_camera_rgb, carla.Transform(carla.Location(x=1.5, z=1.4, y=0)), attach_to=vehicle_follow)
        actor_list.append(camera_rgb)

        fps_max = 30
        display_manager = DisplayManager()
        display_manager.clock.tick(fps_max)

        if controller_to_follow.is_traditional_controller():
            controller_to_follow.register_display_manager(display_manager)

        if controller_follow.is_traditional_controller():
            controller_follow.register_display_manager(display_manager)

        frame_counter = -1
        with SyncCarlaManager(world, camera_rgb, fps=fps_max) as sync_mode:
            while True:
                frame_counter += 1
                display_manager.clock.tick(fps_max)

                if display_manager.should_quit():
                    return

                snapshot, image_rgb = sync_mode.tick(timeout=2.0)
                array = np.frombuffer(image_rgb.raw_data, dtype=np.uint8)
                array = array.reshape((image_rgb.height, image_rgb.width, 4))[:, :, :3]

                # 前车循迹
                if isinstance(controller_to_follow, PathFollower):
                    if frame_counter >= len(pose_manager_to_follow.history):
                        break
                    else:
                        pose_to_follow = pose_manager_to_follow.get_car_pose(frame_counter)
                        vehicle_to_follow.set_transform(pose_to_follow)

                # 获取两车位置与速度
                pose_follow = vehicle_follow.get_transform()
                pose_to_follow = vehicle_to_follow.get_transform()
                velocity_to_follow = vehicle_to_follow.get_velocity()
                velocity_follow = vehicle_follow.get_velocity()

                # 后车感知 - 使用转换后的数组
                info_follow = perceiver_follow.perceive(
                    velocity_follow=velocity_follow,
                    pose_follow=pose_follow,
                    velocity_to_follow=velocity_to_follow,
                    pose_to_follow=pose_to_follow,
                    map=map
                )
                # 计算当前状态（state）
                state = np.array([
                    controller_follow.calculate_relative_distance(info_follow),  # 相对距离
                    controller_follow.calculate_relative_velocity(info_follow),  # 相对速度
                    controller_follow.calculate_relative_heading(info_follow),  # 相对航向
                ])

                # 后车控制
                vehicle_follow_control = controller_follow.predict_control(info_follow)
                vehicle_follow.apply_control(vehicle_follow_control)
                action = vehicle_follow_control.action
                # 获取后车新的位置（pose）和速度（velocity）
                # **更新 info_follow**：这一步是关键，确保 `info_follow` 被更新以反映当前最新的状态
                info_follow.pose_follow = vehicle_follow.get_transform()
                info_follow.velocity_follow = vehicle_follow.get_velocity()
                # 计算下一状态（next_state）
                next_state = np.array([
                    controller_follow.calculate_relative_distance(info_follow),  # 相对距离
                    controller_follow.calculate_relative_velocity(info_follow),  # 相对速度
                    controller_follow.calculate_relative_heading(info_follow),  # 相对航向
                ])
                # 计算奖励
                reward = controller_follow.calculate_reward(info_follow)
                # 存储状态转换
                controller_follow.store_transition(state, action=None, reward=reward, next_state=next_state)
                # 学习（训练）
                controller_follow.learn()

                fps_current = round(1.0 / snapshot.timestamp.delta_seconds)
                display_manager.draw(image_rgb)
                display_manager.write_fps(fps_current)
                display_manager.flip()

                # 每隔一定的时间保存模型
                if frame_counter % 100 == 0:
                    controller_follow.save_model(frame_counter)

    except Exception as e:
        print(f"发生错误: {str(e)}")
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()
        for actor in actor_list:
            actor.destroy()

if __name__ == '__main__':
    for i in range(1, 21):
        file = 'ride' + str(i) + '.p'
        start(controller_to_follow=PathFollower(file),
              perceiver_to_follow=BlindPerceiver(),
              controller_follow=RLController(state_dim=3, action_dim=3, action_bound=[1.0, 1.0, 1.0]),  # 使用RLController
              perceiver_follow=GodPerceiver())
