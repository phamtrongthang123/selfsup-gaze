import os
from datetime import datetime

import torch
import numpy as np
from tqdm import tqdm

from ..utils.random_seed import set_seed
from ..utils.getter import get_instance
from ..utils.meter import AverageValueMeter
from ..utils.device import move_to, detach
from ..utils.exponential_moving_average import ExponentialMovingAverage
from ..loggers import TensorboardLogger, NeptuneLogger

__all__ = ['SelfSupervisedTrainer']


class SelfSupervisedTrainer:
    def __init__(self, config):
        super().__init__()

        self.load_config_dict(config)
        self.config = config

        # Train ID
        self.train_id = self.config.get('id', 'None')
        self.train_id += '-' + datetime.now().strftime('%Y_%m_%d-%H_%M_%S')

        # Get arguments
        self.nepochs = self.config['trainer']['nepochs']
        self.log_step = self.config['trainer']['log_step']
        self.val_step = self.config['trainer']['val_step']
        self.debug = self.config['debug']

        # Instantiate global variables
        self.best_loss = np.inf
        self.val_loss = list()

        # Instantiate loggers
        self.save_dir = os.path.join(self.config['trainer']['log_dir'],
                                     self.train_id)
        self.tsboard = TensorboardLogger(path=self.save_dir)
        # self.tsboard = NeptuneLogger(project_name="thesis-master/torchism", name="test", path=self.save_dir, model_params=config)
        self.amp = False 
        if 'amp' in config:
            self.amp = config['amp']
        self.scaler = torch.cuda.amp.GradScaler() if self.amp else None
        self.model_ema = False
        if 'model_ema' in config:
            self.model_ema = config['model_ema']
            self.best_loss_ema = np.inf
            self.val_loss_ema = list()

    def load_config_dict(self, config):
        # Get device
        dev_id = 'cuda:{}'.format(config['gpus']) \
            if torch.cuda.is_available() and config.get('gpus', None) is not None \
            else 'cpu'
        self.device = torch.device(dev_id)

        # Get pretrained model
        pretrained_path = config["pretrained"]

        pretrained = None
        if (pretrained_path != None):
            pretrained = torch.load(pretrained_path, map_location=dev_id)
            for item in ["model"]:
                config[item] = pretrained["config"][item]

        # 2: Define network
        set_seed(config['seed'])
        self.model = get_instance(config['model']).to(self.device)

        # Train from pretrained if it is not None
        if pretrained is not None:
            self.model.load_state_dict(pretrained['model_state_dict'])

        # # 3: Define loss
        # set_seed(config['seed'])
        # self.criterion = get_instance(config['loss']).to(self.device)

        # 4: Define Optimizer
        set_seed(config['seed'])
        self.optimizer = get_instance(config['optimizer'],
                                      params=self.model.parameters())
        if pretrained is not None:
            self.optimizer.load_state_dict(pretrained['optimizer_state_dict'])

        # 5: Define Scheduler
        set_seed(config['seed'])
        self.scheduler = get_instance(config['scheduler'],
                                      optimizer=self.optimizer, t_total=config['trainer']['nepochs'])


    def save_checkpoint(self, epoch, val_loss,  val_loss_ema=None,):

        data = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }
        torch.save(data, os.path.join(self.save_dir, "last.pth"))
        if self.model_ema:
            data['model_ema_state_dict'] = self.instance_model_ema.state_dict()

        if val_loss < self.best_loss:
            print(
                f'Loss is improved from {self.best_loss: .6f} to {val_loss: .6f}. Saving weights...')
            torch.save(data, os.path.join(self.save_dir, 'best_loss.pth'))
            # Update best_loss
            self.best_loss = val_loss
        else:
            print(f'Loss is not improved from {self.best_loss:.6f}.')


        if val_loss_ema is not None:
            if val_loss_ema < self.best_loss_ema:
                print(
                    f'EMA Loss is improved from {self.best_loss_ema: .6f} to {val_loss_ema: .6f}. Saving weights...')
                torch.save(data, os.path.join(self.save_dir, 'best_loss_ema.pth'))
                # Update best_loss
                self.best_loss_ema = val_loss_ema
            else:
                print(f'EMA Loss is not improved from {self.best_loss_ema:.6f}.')


    def train_epoch(self, epoch, dataloader):
        # 0: Record loss during training process
        running_loss = AverageValueMeter()
        total_loss = AverageValueMeter()
        self.model.train()
        print('Training........')
        progress_bar = tqdm(dataloader)
        for i, (fixation, fix_masks) in enumerate(progress_bar):
            # 1: Load img_inputs and labels
            fixation = move_to(fixation, self.device)
            fix_masks = move_to(fix_masks, self.device)
            fixation_inp, fix_masks_inp = fixation[:,:-1,:], fix_masks[:,:-1,:]
            # for out, we only need x,y. No need for time.
            fixation_out, fix_masks_out = fixation[:,1:,:2], fix_masks[:,1:,:]
            # 2: Clear gradients from previous iteration
            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                # 3: Get network outputs
                outs = self.model(fixation_inp, fix_masks_inp)
                # 4: Calculate the loss
                loss = self.model.build_loss(outs, fixation_out, fix_masks_out)
            
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # 5: Calculate gradients
                loss.backward()
                # 6: Performing backpropagation
                self.optimizer.step()
            with torch.no_grad():
                # 7: Update loss
                running_loss.add(loss.item())
                total_loss.add(loss.item())

                if (i + 1) % self.log_step == 0 or (i + 1) == len(dataloader):
                    self.tsboard.update_loss(
                        'train', running_loss.value()[0], epoch * len(dataloader) + i)
                    running_loss.reset()

            if self.model_ema and i % self.model_ema['model_ema_steps'] == 0:
                self.instance_model_ema.update_parameters(self.model)
                if epoch < self.config['scheduler']['args']['lr_warmup_epochs']:
                    # Reset ema buffer to keep copying weights during warmup period
                    self.instance_model_ema.n_averaged.fill_(0)
        print('+ Training result')
        avg_loss = total_loss.value()[0]
        print('Loss:', avg_loss)
        self.val_loss.append(avg_loss)




    def train(self, train_dataloader, val_dataloader):
        set_seed(self.config['seed'])
        if self.model_ema:
            adjust = self.config['dataset']['train']['loader']['args']['batch_size'] * self.model_ema['model_ema_steps'] / self.config['trainer']['nepochs']
            alpha = 1.0 - self.model_ema['model_ema_decay']
            alpha = min(1.0, alpha * adjust)
            self.instance_model_ema = ExponentialMovingAverage(self.model, device=self.device, decay=1.0 - alpha)

        for epoch in range(self.nepochs):
            print('\nEpoch {:>3d}'.format(epoch))
            print('-----------------------------------')

            # Note learning rate
            for i, group in enumerate(self.optimizer.param_groups):
                self.tsboard.update_lr(i, group['lr'], epoch)

            # 1: Training phase
            self.train_epoch(epoch=epoch, dataloader=train_dataloader)

            print()


            val_loss = self.val_loss[-1]
            self.save_checkpoint(epoch, val_loss)