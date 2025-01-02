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
    carla_path = 'E:/VIVADO/CARLA_0.9.8/WindowsNoEditor/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64')
    
    # 添加调试信息
    matching_files = glob.glob(carla_path)
    if not matching_files:
        print(f"在路径下未找到匹配的文件: {carla_path}")
    else:
        print(f"找到文件: {matching_files[0]}")
        sys.path.append(matching_files[0])
except IndexError:
    print("CARLA_0.9.8 not found")

import carla
import random

from manager.sync_carla_manager import SyncCarlaManager
from manager.pose_manager import PoseManager
from manager.display_manager import DisplayManager

from controller.daf_controller import DAFController
from controller.carla_auto_pilot import CarlaAutoPilot
from controller.path_follower import PathFollower
from controller.manual_controller import ManualController
from controller.follow_track_controller import FollowTrackController
from controller.normal_controller import NormalController

from perceiver.god_perceiver import GodPerceiver
from perceiver.blind_perceiver import BlindPerceiver
from perceiver.normal_percerver import NormalPerceiver


def start(controller_to_follow, controller_follow, perceiver_to_follow, perceiver_follow):

    actor_list = []

    try:
        # 初始化carla客户端
        client = carla.Client('127.0.0.1', 2000)
        client.set_timeout(2.0)
        world = client.get_world()
        map = world.get_map()
        blueprint_library = world.get_blueprint_library()

        # 生成前车
        pose_manager_to_follow = PoseManager()
        pose_to_follow = random.choice(world.get_map().get_spawn_points())
        bp_to_follow = blueprint_library.filter('model3')[0]
        bp_to_follow.set_attribute('color', '0,101,189')
        vehicle_to_follow = world.spawn_actor(
            bp_to_follow,
            pose_to_follow
        )
        vehicle_to_follow.set_simulate_physics(True)
        actor_list.append(vehicle_to_follow)
        vehicle_to_follow.set_autopilot(False)

        # 前车自动驾驶
        if type(controller_to_follow) is CarlaAutoPilot:
            vehicle_to_follow.set_autopilot(True)
        # 前车循迹
        elif type(controller_to_follow) is PathFollower:
            pose_manager_to_follow.load_history_from_file('path', controller_to_follow.file)
            pose_to_follow = pose_manager_to_follow.get_car_pose(0)
            vehicle_to_follow.set_transform(pose_to_follow)


        # 生成后车
        pose_follow = PoseManager.create_pose_in_front_of(pose_to_follow, -5, 0.1) # height=0.1防止spawn_collision
        bp_follow = blueprint_library.filter('jeep')[0]
        vehicle_follow = world.spawn_actor(
            bp_follow,
            pose_follow
        )
        vehicle_follow.set_simulate_physics(True)
        actor_list.append(vehicle_follow)


        # 生成后车传感器
        bp_collision_sensor = blueprint_library.find('sensor.other.collision')
        collision_sensor = world.spawn_actor(
            bp_collision_sensor,
            carla.Transform(),
            attach_to=vehicle_follow
        )
        # TODO: 处理碰撞传感器信息

        bp_camera_rgb = blueprint_library.find('sensor.camera.rgb')
        bp_camera_rgb.set_attribute('image_size_x', '800')
        bp_camera_rgb.set_attribute('image_size_y', '600')
        bp_camera_rgb.set_attribute('fov', '90')
        camera_rgb = world.spawn_actor(
            bp_camera_rgb,
            carla.Transform(carla.Location(x=1.5, z=1.4, y=0), carla.Rotation(pitch=0)),
            attach_to=vehicle_follow
        )
        actor_list.append(camera_rgb)

        # 设置最大帧率
        fps_max = 30

        # 初始化pygame
        display_manager = DisplayManager()
        display_manager.clock.tick(fps_max)

        if controller_to_follow.is_traditional_controller():
            controller_to_follow.register_display_manager(display_manager)

        if controller_follow.is_traditional_controller():
            controller_follow.register_display_manager(display_manager)

        # 帧计数器
        frame_counter = -1

        with SyncCarlaManager(world, camera_rgb, fps=fps_max) as sync_mode:
            while True:
                frame_counter += 1
                display_manager.clock.tick(fps_max)

                if display_manager.should_quit():
                    return

                # 获取当前帧世界信息与传感器信息
                snapshot, image_rgb = sync_mode.tick(timeout=2.0)

                # 将CARLA图像转换为NumPy数组
                array = np.frombuffer(image_rgb.raw_data, dtype=np.uint8)
                array = array.reshape((image_rgb.height, image_rgb.width, 4))
                array = array[:, :, :3]  # 只保留RGB通道
                
                # 添加以下代码来显示RGB相机图像
                cv2.imshow('RGB Camera View', array)
                cv2.waitKey(1)

                # 如果前车循迹
                if type(controller_to_follow) is PathFollower:
                    # 如果路径结束则退出
                    if frame_counter >= len(pose_manager_to_follow.history):
                        break
                    # 否则加载该路径下当前帧前车位置
                    else:
                        pose_to_follow = pose_manager_to_follow.get_car_pose(frame_counter)
                        vehicle_to_follow.set_transform(pose_to_follow)
                # 如果前车通过油门刹车转向控制
                elif controller_to_follow.is_traditional_controller():
                    info_to_follow = perceiver_to_follow.perceive()
                    vehicle_to_follow_control = controller_to_follow.predict_control(info_to_follow)
                    vehicle_to_follow.apply_control(vehicle_to_follow_control)

                # 获取两车位置
                pose_follow = vehicle_follow.get_transform()
                pose_to_follow = vehicle_to_follow.get_transform()

                # 获取前后车速度
                velocity_to_follow = vehicle_to_follow.get_velocity()
                velocity_follow = vehicle_follow.get_velocity()

                # 后车通过油门刹车转向控制
                assert(controller_follow.is_traditional_controller())

                # 后车感知 - 使用转换后的数组
                info_follow = perceiver_follow.perceive(
                    velocity_follow=velocity_follow, 
                    pose_follow=pose_follow, 
                    map=map,
                    camera_image=array.copy()  # 传入数组的副本
                )

                # 确保pose_to_follow不为None
                if info_follow.pose_to_follow is None:
                    print("未检测到前车，跳过当前帧")
                    continue

                # 后车控制
                vehicle_follow_control = controller_follow.predict_control(info_follow)
                vehicle_follow.apply_control(vehicle_follow_control)

                # 绘图 - 使用原始图像数据
                fps_current = round(1.0 / snapshot.timestamp.delta_seconds)
                display_manager.draw(image_rgb)  # 使用原始CARLA图像而不是NumPy数组
                display_manager.write_fps(fps_current)
                display_manager.flip()

    except Exception as e:
        print(f"发生错误: {str(e)}")
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()  # 关闭所有OpenCV窗口
        for actor in actor_list:
            actor.destroy()

if __name__ == '__main__':
    # 整个任务分为两个步骤
    # 第一步：感知，感知器perceiver将环境感知信息（Camera图像，地图信息，后车Pose，后车Velocity等）传入perceiver，perceiver输出一个数据传输对象info，包含经过深度处理的信息（前车Pose，前车Velocity等）
    # 第二步：控制，将对应的数据传输对象info传入控制器controller，控制器输出车辆控制

    # xxx_to_follow 表示被跟的车，即前车
    # xxx_follow 表示跟的车，即后车

    # 下面定义了前车和后车的perceiver与controller算法
    # 为了让前车能够循迹或使用carla的auto_pilot()方法，创建了两个特殊的controller：PathFollower和AutoPilotController
    # 由于前车不需要感知环境信息，BlindPerceiver()被传入
    # 出于可拓展性的考量，前车也可以使用其他自定义的自动寻路算法，只需要将对应的perceiver和controller传入即可

    # 后车的控制算法使用简化兼容版DAFController，感知算法没写，暂时使用全知全能的神GodPerceiver占位
    
    for i in range(1, 21):
        file = 'ride' + str(i) + '.p'
        start(controller_to_follow=PathFollower(file), perceiver_to_follow=BlindPerceiver(),
              controller_follow= NormalController(), perceiver_follow=NormalPerceiver())
