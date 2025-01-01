from controller.traditional_controller import TraditionalController
from manager.pose_manager import PoseManager
from manager.vec3d_utils import get_magnitude, get_angle
import carla

# 记录前车历史坐标点并循迹
class DAFFollowTrackController(TraditionalController):
    def __init__(self):
        self.car_length = 6
        self.intent_distance = self.car_length

        self.distance = 0
        self.prev_distance = 0
        self.dist_change = 0
        self.distance_emg = 0
        self.prev_distance_emg = 0
        self.dist_change_emg = 0

        self.a_secure = 2
        self.a_brake = 1

        self.angle = 0
        self.prev_angle = 0
        self.angle_change = 0

        self.look_forward = 30

        self.throttle = 0
        self.brake = 0
        self.steer = 0

        self.checkpoints = []
        self.check_distance = 5


    def predict_control(self, info):
        self.checkpoints.append(info.pose_to_follow)

        while len(self.checkpoints) > 1:
            if PoseManager.get_distance(info.pose_follow, self.checkpoints[0]) < self.check_distance:
                self.checkpoints.pop(0)
            else:
                break

        distance = PoseManager.get_distance(info.pose_follow, info.pose_to_follow)
        self.dist_change = self.distance - self.prev_distance
        self.prev_distance = self.distance
        self.distance = distance - self.intent_distance

        vec_to_next_checkpoint = carla.Vector3D(self.checkpoints[0].location.x - info.pose_follow.location.x,
                                          self.checkpoints[0].location.y - info.pose_follow.location.y,
                                          self.checkpoints[0].location.z - info.pose_follow.location.z)
        pose_in_front_of_follow = PoseManager.create_pose_in_front_of(info.pose_follow, 1)
        vec_forward = carla.Vector3D(pose_in_front_of_follow.location.x - info.pose_follow.location.x,
                                     pose_in_front_of_follow.location.y - info.pose_follow.location.y,
                                     pose_in_front_of_follow.location.z - info.pose_follow.location.z)
        self.angle = get_angle(vec_to_next_checkpoint, vec_forward)
        self.angle_change = self.angle - self.prev_angle
        self.prev_angle = self.angle

        self.decide_steer()
        self.decide_throttle_and_brake()

        print("\rthrottle: %.2f \tsteer: %.2f \tbrake: %.2f" %(self.throttle, self.steer, self.brake), end='')

        return carla.VehicleControl(throttle=self.throttle, steer=self.steer, brake=self.brake)

    # 决策油门与刹车
    def decide_throttle_and_brake(self):
        if self.dist_change > 0 and self.distance > 0:
            # 加速！加速！加速！
            self.throttle = 1
            self.brake = 0
        else:
            # 考虑当前距离是否能够刹停
            dif_v = self.dist_change * self.get_fps()
            # 考虑刹车迟滞与反应时间
            distance_consider_delay = self.distance + dif_v
            if distance_consider_delay <= 0.01:
                a = 100
            else:
                a = (dif_v * dif_v) / (distance_consider_delay * 2)

            # 如果刹车所需加速度a小于a_brake，则适当松油门
            if a < self.a_brake:
                self.throttle = (self.a_brake - a) / self.a_brake
                self.brake = 0
            # 如果a大于a_brake但小于a_secure，则适当踩刹车
            elif a < self.a_secure:
                self.throttle = 0
                self.brake = (a - self.a_brake) / (self.a_secure - self.a_brake)
            # 否则踩死刹车
            else:
                self.throttle = 0
                self.brake = 1

    # 决策转向
    def decide_steer(self):
        self.steer = self.angle / 90
        if self.steer > 1:
            self.steer = 1
        if self.steer < -1:
            self.steer = -1

