import os
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import Callback


class BestModelCallback(Callback):

    def __init__(self, filename='best-model-{epoch}-{loss:.2f}', dirpath="./run/models/"):
        super(BestModelCallback, self).__init__()  
        self.filename = filename
        self.dirpath  = dirpath
        os.makedirs(dirpath, exist_ok=True)
        self.best_model_path = None
        self.model_checkpoint = ModelCheckpoint(
            dirpath    = dirpath,
            filename   = filename,
            save_top_k = 1,
            verbose    = False,
            monitor    = "vae_loss",
            mode       = "min"
        )

    def on_train_epoch_end(self, trainer, pl_module):
        # save the best model
        self.model_checkpoint.on_train_epoch_end(trainer, pl_module)
        self.best_model_path = self.model_checkpoint.best_model_path

                
