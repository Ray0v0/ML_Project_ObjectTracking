# Distance-Angle-Fusion Controller
from controller.traditional_controller import TraditionalController
from manager.pose_manager import PoseManager
from manager.vec3d_utils import get_magnitude, get_angle
import carla

# 简化的daf_controller
class DAFController(TraditionalController):
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

        self.speed = 0
        self.throttle = 0
        self.brake = 0
        self.steer = 0

        self.emergency = False



    def predict_control(self, info):
        # 通过前后车坐标速度等信息计算距离和转向角（其实用不到这么多信息；或者说，如果这些信息已经知道了，照道理应该能写一个更好的控制算法）
        # 但因为懒，就这样子暂且兼容一下吧
        distance = PoseManager.get_distance(info.pose_follow, info.pose_to_follow)
        speed = get_magnitude(info.velocity_follow)
        vec_between_cars = carla.Vector3D(info.pose_to_follow.location.x - info.pose_follow.location.x,
                                          info.pose_to_follow.location.y - info.pose_follow.location.y,
                                          info.pose_to_follow.location.z - info.pose_follow.location.z)
        pose_in_front_of_follow = PoseManager.create_pose_in_front_of(info.pose_follow, 1)
        vec_forward = carla.Vector3D(pose_in_front_of_follow.location.x - info.pose_follow.location.x,
                                     pose_in_front_of_follow.location.y - info.pose_follow.location.y,
                                     pose_in_front_of_follow.location.z - info.pose_follow.location.z)
        angle_between_cars = get_angle(vec_between_cars, vec_forward)

        # print(vec_between_cars)
        # print(vec_forward)

        # print("distance: %.2f \tangle: %.2f" % (distance, angle_between_cars), end='')

        # self.emergency = self.whether_emergency(distance)

        self.speed = speed

        self.dist_change = self.distance - self.prev_distance
        self.prev_distance = self.distance
        self.distance = distance - self.intent_distance

        self.angle = angle_between_cars
        self.angle_change = self.angle - self.prev_angle
        self.prev_angle = self.angle

        self.decide_steer()
        self.decide_throttle_and_brake()
        self.fusion()

        # 如果遇到紧急情况，反打方向盘，踩死刹车（前车转弯容易使车距过近，误触AEB，导致跟丢，因此注释掉了）
        # if self.emergency:
        #     self.steer = -self.steer
        #     self.throttle = 0
        #     self.brake = 1

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
            # print("dif_v: %.2f \t a: %.2f \t" % (dif_v, a), end='')

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
        # self.steer = (self.angle + self.look_forward * self.angle_change) / 180.0
        # # 如果转角过小，则不需要修正方向
        # if abs(self.steer) < 0.04:
        #     self.steer = self.steer * abs(self.steer) * 25
        # print("angle: %.2f \t anlge_change: %.2f \t" % (self.angle, self.angle_change), end='')


    # 融合决策
    def fusion(self):
        pass
        # # 如果遇到转弯，则放慢速度增加车距
        # if abs(self.steer) > 0.1:
        #     if abs(self.steer) < 0.17:
        #         self.intent_distance = self.car_length * 1.5
        #     else:
        #         self.intent_distance = self.car_length * 1.6
        # # 否则逐步减少跟车距离
        # else:
        #     if self.intent_distance > self.car_length * 1.4:
        #         self.intent_distance -= 0.2

        # print("intend_dist: %.2f \t" % self.intent_distance, end='')

    # 主动AEB（已被注释）
    def whether_emergency(self, distance):
        self.distance_emg = distance - self.car_length
        self.dist_change_emg = self.distance_emg - self.prev_distance_emg
        dif_v = self.dist_change_emg * self.get_fps()
        distance_consider_delay = max(0.01, self.distance_emg + dif_v)
        a = (dif_v * dif_v) / (distance_consider_delay * 2)
        print("a_emg: %.2f \t" % a, end='')
        self.prev_distance_emg = self.distance_emg
        return a > self.a_secure