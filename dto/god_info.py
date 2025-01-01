class GodInfo(object):
    def __init__(self, velocity_follow, pose_follow, velocity_to_follow, pose_to_follow, map):
        self.velocity_follow = velocity_follow
        self.pose_follow = pose_follow
        self.velocity_to_follow = velocity_to_follow
        self.pose_to_follow = pose_to_follow
        self.map = map