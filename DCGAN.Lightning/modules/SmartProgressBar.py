
# ------------------------------------------------------------------
#     _____ _     _ _
#    |  ___(_) __| | | ___
#    | |_  | |/ _` | |/ _ \
#    |  _| | | (_| | |  __/
#    |_|   |_|\__,_|_|\___|                   GAN / SmartProgressBar
# ------------------------------------------------------------------
# Formation Introduction au Deep Learning  (FIDLE)
# CNRS/SARI/DEVLOG MIAI/EFELIA 2023 - https://fidle.cnrs.fr
# ------------------------------------------------------------------
# by JL Parouty (feb 2023) - PyTorch Lightning example

from lightning.pytorch.callbacks.progress.base import ProgressBarBase
from tqdm import tqdm
import sys

class SmartProgressBar(ProgressBarBase):

    def __init__(self, verbosity=2):
        super().__init__()
        self.verbosity = verbosity

    def disable(self):
        self.enable = False


    def setup(self, trainer, pl_module, stage):
        super().setup(trainer, pl_module, stage)
        self.stage = stage


    def on_train_epoch_start(self, trainer, pl_module):
        super().on_train_epoch_start(trainer, pl_module)
        if not self.enable : return

        if self.verbosity==2:
            self.progress=tqdm( total=trainer.num_training_batches,
                                desc=f'{self.stage} {trainer.current_epoch+1}/{trainer.max_epochs}', 
                                ncols=100, ascii= " >", 
                                bar_format='{l_bar}{bar}| [{elapsed}] {postfix}')



    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        if not self.enable : return

        if self.verbosity==2:
            self.progress.close()

        if self.verbosity==1:
            print(f'Train {trainer.current_epoch+1}/{trainer.max_epochs} Done.')


    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

        if not self.enable : return
        
        if self.verbosity==2:
            metrics = {}
            for name,value in trainer.logged_metrics.items():
                metrics[name]=f'{float( trainer.logged_metrics[name] ):3.3f}'
            self.progress.set_postfix(metrics)
            self.progress.update(1)


progress_bar = SmartProgressBar(verbosity=2)
