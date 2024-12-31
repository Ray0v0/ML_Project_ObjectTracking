from dto.dfa_info import DfaInfo


class GodPerceiver(object):
    def perceive(self, velocity_follow, pose_follow, velocity_to_follow, pose_to_follow):
        return DfaInfo(velocity_follow, pose_follow, velocity_to_follow, pose_to_follow)
