import os
import pickle
import math
import carla

class PoseManager(object):
    def __init__(self):
        self.history = []

    def save_car_pose(self, pose):
        self.history.append(self.unzip(pose))

    def get_car_pose(self, frame_index):
        return self.zip(self.history[frame_index])

    def save_history_to_file(self, directory, filename):
        if not os.path.exists(directory):
            os.mkdir(directory)
        if len(self.history) > 0:
            pickle.dump(self.history,  open(os.path.join(directory, filename), "wb"))

    def load_history_from_file(self, directory, filename):
        self.history = pickle.load(open(os.path.join(directory, filename), "rb"))


    @staticmethod
    def zip(location_and_rotation_list):
        location = carla.Location(location_and_rotation_list[0], location_and_rotation_list[1], location_and_rotation_list[2])
        rotation = carla.Rotation(location_and_rotation_list[3], location_and_rotation_list[4], location_and_rotation_list[5])
        return carla.Transform(location, rotation)

    @staticmethod
    def unzip(pose):
        return [pose.location.x, pose.location.y, pose.location.z,
                pose.rotation.pitch, pose.rotation.yaw, pose.rotation.roll]

    @staticmethod
    def create_pose_in_front_of(pose, distance, height=0):
        x_delta = math.cos(math.radians(pose.rotation.yaw)) * distance
        y_delta = math.sin(math.radians(pose.rotation.yaw)) * distance
        location = carla.Location(pose.location.x + x_delta, pose.location.y + y_delta, pose.location.z + height)
        rotation = carla.Rotation(pose.rotation.pitch, pose.rotation.yaw, pose.rotation.roll)
        return carla.Transform(location, rotation)

    @staticmethod
    def get_distance(pose1, pose2):
        return ((pose1.location.x - pose2.location.x) ** 2 +
                (pose1.location.y - pose2.location.y) ** 2 +
                (pose1.location.z - pose2.location.z) ** 2) ** 0.5