from dto.god_info import GodInfo

# 啥信息都能有的无敌感知器
class GodPerceiver(object):
    def perceive(self, velocity_follow, pose_follow, velocity_to_follow, pose_to_follow, map):
        return GodInfo(velocity_follow, pose_follow, velocity_to_follow, pose_to_follow, map)
