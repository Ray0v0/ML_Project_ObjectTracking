from dto.info import Info


class DAFInfo(Info):
    def __init__(self, distance, angle, trust_worth=True):
        super().__init__(trust_worth)
        self.distance = distance
        self.angle = angle