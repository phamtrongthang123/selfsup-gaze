import torch 
from torch.optim.lr_scheduler import SequentialLR

class WarmupLinearCosineAnnealing(SequentialLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """
    def __init__(self, optimizer, lr_warmup_decay=0.01, lr_warmup_epochs=5, t_total=600, lr_min=0, last_epoch=-1, verbose=False):
        self.warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=lr_warmup_decay, total_iters=lr_warmup_epochs
            )
        self.main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t_total - lr_warmup_epochs, eta_min=lr_min
        )
       
        super(WarmupLinearCosineAnnealing, self).__init__(optimizer,schedulers=[self.warmup_lr_scheduler, self.main_lr_scheduler], milestones=[lr_warmup_epochs], last_epoch=last_epoch, verbose=verbose)



