from abc import abstractmethod, abstractstaticmethod


class Controller(object):
    @staticmethod
    @abstractmethod
    def is_traditional_controller():
        pass

    def close(self):
        pass