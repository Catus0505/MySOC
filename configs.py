import torch

class PreprocessConfig(object):
    def __init__(self):

        self.batch_size = 64
        self.time_step = 20
        self.stride = self.time_step // 2


class PretrainConfig(object):
    def __init__(self):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = '10degC/10degC_US06.csv'
        self.num_epochs = 100


class TransferConfig(object):
    def __init__(self):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.source = '10degC/10degC_US06.csv'
        self.target = '10degC/10degC_LA92.csv'
        self.checkpoint = 'pretrain_model.pth'
        self.num_epochs = 100


class EvaluateConfig(object):
    def __init__(self):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = '10degC/10degC_US06.csv'
        self.checkpoint = 'pretrain_model.pth'