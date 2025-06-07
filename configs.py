import torch

class PreprocessConfig(object):
    def __init__(self):

        self.batch_size = 256
        self.time_step = 20
        self.stride = self.time_step // 2