class Config:

    def __init__(self):
        self.input_shape = [128, 128, 128]
        self.dnn_shape = [128, 32, 16]
        self.filters = [16, 32, 32, 256]
        self.kernel_size = [4, 8, 4, 8]
        self.strides = [2, 4, 2, 8]
        self.padding = "valid"
        self.buffer_size = 20
        self.learning_rate = 5e-3
        self.train_episodes = 2000
        self.train_buffer_size = 10
