import os
import pickle
import math
import carla

# Pose即Carla中的Transform类
# 虽然官方叫Transform，但这个名字比较容易误解成一个相对的位移或变换
# 因此使用机器人领域的专有名词Pose表示一个actor的location和rotation信息
class PoseManager(object):
    def __init__(self):
        self.history = []

    # 为后续自定义路线预留接口
    def save_car_pose(self, pose):
        self.history.append(self.unzip(pose))

    # 获取history中第frame_index帧的车辆坐标
    def get_car_pose(self, frame_index):
        return self.zip(self.history[frame_index])

    # 为后续自定义路线预留接口
    def save_history_to_file(self, directory, filename):
        if not os.path.exists(directory):
            os.mkdir(directory)
        if len(self.history) > 0:
            pickle.dump(self.history,  open(os.path.join(directory, filename), "wb"))

    # 读取路线
    def load_history_from_file(self, directory, filename):
        self.history = pickle.load(open(os.path.join(directory, filename), "rb"))

    # 将location_and_rotation_list（格式为[location.x, location.y, location.z, rotation.pitch, rotation.yaw, rotation.roll]）打包为一个Pose
    @staticmethod
    def zip(location_and_rotation_list):
        location = carla.Location(location_and_rotation_list[0], location_and_rotation_list[1], location_and_rotation_list[2])
        rotation = carla.Rotation(location_and_rotation_list[3], location_and_rotation_list[4], location_and_rotation_list[5])
        return carla.Transform(location, rotation)

    # 从一个Pose中解包出location_and_rotation_list，格式参见上文
    @staticmethod
    def unzip(pose):
        return [pose.location.x, pose.location.y, pose.location.z,
                pose.rotation.pitch, pose.rotation.yaw, pose.rotation.roll]

    # 在Pose前方distance、高height处创建新Pose，方向与原Pose一致
    @staticmethod
    def create_pose_in_front_of(pose, distance, height=0):
        x_delta = math.cos(math.radians(pose.rotation.yaw)) * distance
        y_delta = math.sin(math.radians(pose.rotation.yaw)) * distance
        location = carla.Location(pose.location.x + x_delta, pose.location.y + y_delta, pose.location.z + height)
        rotation = carla.Rotation(pose.rotation.pitch, pose.rotation.yaw, pose.rotation.roll)
        return carla.Transform(location, rotation)

    # 获取两个Pose之间的距离
    @staticmethod
    def get_distance(pose1, pose2):
        return ((pose1.location.x - pose2.location.x) ** 2 +
                (pose1.location.y - pose2.location.y) ** 2 +
                (pose1.location.z - pose2.location.z) ** 2) ** 0.5