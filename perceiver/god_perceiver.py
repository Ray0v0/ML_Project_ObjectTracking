from dto.daf_info import DAFInfo
from dto.god_info import GodInfo
from manager.pose_manager import PoseManager
from manager.vec3d_utils import get_angle
import carla

# 啥信息都能有的无敌感知器
class GodPerceiver(object):
    def perceive(self, velocity_follow, pose_follow, velocity_to_follow, pose_to_follow, map):
        # return GodInfo(velocity_follow, pose_follow, velocity_to_follow, pose_to_follow, map)
        distance = PoseManager.get_distance(pose_follow, pose_to_follow)
        vec_between_cars = carla.Vector3D(pose_to_follow.location.x - pose_follow.location.x,
                                          pose_to_follow.location.y - pose_follow.location.y,
                                          pose_to_follow.location.z - pose_follow.location.z)
        pose_in_front_of_follow = PoseManager.create_pose_in_front_of(pose_follow, 1)
        vec_forward = carla.Vector3D(pose_in_front_of_follow.location.x - pose_follow.location.x,
                                     pose_in_front_of_follow.location.y - pose_follow.location.y,
                                     pose_in_front_of_follow.location.z - pose_follow.location.z)
        angle_between_cars = get_angle(vec_between_cars, vec_forward)

        return DAFInfo(distance, angle_between_cars), None