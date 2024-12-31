class TraditionalController(object):

    def register_display_manager(self, display_manager):
        self.display_manager = display_manager

    def get_fps(self):
        return self.display_manager.clock.get_fps()

    @staticmethod
    def is_traditional_controller():
        return True