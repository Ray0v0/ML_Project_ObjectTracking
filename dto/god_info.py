from dto.info import Info


class GodInfo(Info):
    def __init__(self, velocity_follow, pose_follow, velocity_to_follow, pose_to_follow, map, trust_worth=True):
        super().__init__(trust_worth)
        self.velocity_follow = velocity_follow
        self.pose_follow = pose_follow
        self.velocity_to_follow = velocity_to_follow
        self.pose_to_follow = pose_to_follow
        self.map = map
