import datetime

class Evaluator:
    def __init__(self):
        self.ride_evaluation = []
        self.ride_index = -1
        pass

    def next_ride(self):
        self.ride_evaluation.append(0)

    def evaluate(self, info):
        pass

    def collision_occurred(self, info):
        pass

    def append_loss(self, loss):
        self.ride_evaluation[self.ride_index] += loss

    def save_evaluation(self):
        filename = self.generate_filename()
        with open(filename, 'w') as f:
            for item in self.ride_evaluation:
                f.write(str(item))
                f.write('\n')

    @staticmethod
    def generate_filename():
        now = datetime.datetime.now()
        formatted_time = now.strftime("%Y%m%d_%H%M%S")
        filename = f"evaluation\\{formatted_time}.txt"
        return filename