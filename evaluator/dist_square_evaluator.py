from evaluator.evaluator import Evaluator
from manager.pose_manager import PoseManager
from manager.vec3d_utils import get_magnitude


class DistSquareEvaluator(Evaluator):
    def evaluate(self, info):
        pose_to_follow = info[0]
        pose_follow = info[1]
        distance = PoseManager.get_distance(pose_to_follow, pose_follow) ** 2
        self.append_loss(distance)
        print(self.ride_evaluation[-1])
        

    def collision_occurred(self, info):
        speed = get_magnitude(info[0])
        self.append_loss(speed * 100)