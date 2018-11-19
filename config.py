class Config:

    def __init__(self):
        # network params
        self.input_shape = [128, 128, 128, 1]
        self.dnn_shape = [256, 256, 16 * 7]
        self.filters = [16, 32, 32, 256]
        self.kernel_size = [4, 8, 4, 8]
        self.strides = [2, 4, 2, 8]
        self.padding = "SAME"

        # training params
        self.buffer_size = 20
        self.learning_rate = 5e-4
        self.train_episodes = 50000
        self.train_buffer_size = 20
        self.training_set_percent = 0.8

        # about model saving
        self.every_steps_save = 2000
        self.model_load_path = None
        self.model_load = False
