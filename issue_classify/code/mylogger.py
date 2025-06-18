import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

class CustomTensorBoardLogger(TensorBoardLogger):
    def __init__(self, save_dir, name="default", log_dir_name="custom_log_dir"):
        super().__init__(save_dir, name)
        self.log_dir_name = log_dir_name

    @property
    def log_dir(self):
        # Override the log_dir to use the custom log directory name
        return os.path.join(self.save_dir, self.name, self.log_dir_name)
    
# 使用实例
_logger = CustomTensorBoardLogger(save_dir="logs", log_dir_name="my_experiment")
