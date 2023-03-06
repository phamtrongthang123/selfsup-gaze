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

__all__ = ['SupervisedTrainer']


class SupervisedTrainer:
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
        self.best_metric = {k: 0.0 for k in self.metric.keys()}
        self.val_loss = list()
        self.val_metric = {k: list() for k in self.metric.keys()}

        # Instantiate loggers
        self.save_dir = os.path.join(self.config['trainer']['log_dir'],
                                     self.train_id)
        # self.tsboard = TensorboardLogger(path=self.save_dir)
        self.tsboard = NeptuneLogger(project_name="thesis-master/torchism", name="test", path=self.save_dir, model_params=config)
        self.amp = False 
        if 'amp' in config:
            self.amp = config['amp']
        self.scaler = torch.cuda.amp.GradScaler() if self.amp else None
        self.model_ema = False
        if 'model_ema' in config:
            self.model_ema = config['model_ema']
            self.best_loss_ema = np.inf
            self.best_metric_ema = {k: 0.0 for k in self.metric.keys()}
            self.val_loss_ema = list()
            self.val_metric_ema = {k: list() for k in self.metric.keys()}

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

        # 3: Define loss
        set_seed(config['seed'])
        self.criterion = get_instance(config['loss']).to(self.device)

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

        # 6: Define metrics
        set_seed(config['seed'])
        self.metric = {mcfg['name']: get_instance(mcfg)
                       for mcfg in config['metric']}

    def save_checkpoint(self, epoch, val_loss, val_metric, val_loss_ema=None, val_metric_ema=None):

        data = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }
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

        for k in self.metric.keys():
            if val_metric[k] > self.best_metric[k]:
                print(
                    f'{k} is improved from {self.best_metric[k]: .6f} to {val_metric[k]: .6f}. Saving weights...')
                torch.save(data, os.path.join(
                    self.save_dir, f'best_metric_{k}.pth'))
                self.best_metric[k] = val_metric[k]
            else:
                print(
                    f'{k} is not improved from {self.best_metric[k]:.6f}.')

        if val_loss_ema is not None:
            if val_loss_ema < self.best_loss_ema:
                print(
                    f'EMA Loss is improved from {self.best_loss_ema: .6f} to {val_loss_ema: .6f}. Saving weights...')
                torch.save(data, os.path.join(self.save_dir, 'best_loss_ema.pth'))
                # Update best_loss
                self.best_loss_ema = val_loss_ema
            else:
                print(f'EMA Loss is not improved from {self.best_loss_ema:.6f}.')

            for k in self.metric.keys():
                if val_metric_ema[k] > self.best_metric_ema[k]:
                    print(
                        f'EMA {k} is improved from {self.best_metric_ema[k]: .6f} to {val_metric_ema[k]: .6f}. Saving weights...')
                    torch.save(data, os.path.join(
                        self.save_dir, f'best_metric_ema_{k}.pth'))
                    self.best_metric_ema[k] = val_metric_ema[k]
                else:
                    print(
                        f'EMA {k} is not improved from {self.best_metric_ema[k]:.6f}.')
        # print('Saving current model...')
        # torch.save(data, os.path.join(self.save_dir, 'current.pth'))

    def train_epoch(self, epoch, dataloader):
        # 0: Record loss during training process
        running_loss = AverageValueMeter()
        total_loss = AverageValueMeter()
        for m in self.metric.values():
            m.reset()
        self.model.train()
        print('Training........')
        progress_bar = tqdm(dataloader)
        for i, (inp, lbl) in enumerate(progress_bar):
            # 1: Load img_inputs and labels
            inp = move_to(inp, self.device)
            lbl = move_to(lbl, self.device)
            # 2: Clear gradients from previous iteration
            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                # 3: Get network outputs
                outs = self.model(inp)
                # 4: Calculate the loss
                loss = self.criterion(outs, lbl)
            
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

                # 8: Update metric
                outs = detach(outs)
                lbl = detach(lbl)
                for m in self.metric.values():
                    m.update(outs, lbl)

            if self.model_ema and i % self.model_ema['model_ema_steps'] == 0:
                self.instance_model_ema.update_parameters(self.model)
                if epoch < self.config['scheduler']['args']['lr_warmup_epochs']:
                    # Reset ema buffer to keep copying weights during warmup period
                    self.instance_model_ema.n_averaged.fill_(0)
        print('+ Training result')
        avg_loss = total_loss.value()[0]
        print('Loss:', avg_loss)
        for k in self.metric.keys():
            m = self.metric[k].value()
            self.metric[k].summary()
            self.tsboard.update_metric('train', k, m, epoch)

        

    @torch.no_grad()
    def val_epoch(self, epoch, dataloader, ema=False):
        running_loss = AverageValueMeter()
        for m in self.metric.values():
            m.reset()
        if ema:
            self.instance_model_ema.eval()
        else:
            self.model.eval()
        print('Evaluating........')
        progress_bar = tqdm(dataloader)
        for i, (inp, lbl) in enumerate(progress_bar):
            # 1: Load inputs and labels
            inp = move_to(inp, self.device)
            lbl = move_to(lbl, self.device)
            # 2: Get network outputs
            if ema:
                outs = self.instance_model_ema(inp)
            else:   
                outs = self.model(inp)
            # 3: Calculate the loss
            loss = self.criterion(outs, lbl)
            # 4: Update loss
            running_loss.add(loss.item())
            # 5: Update metric
            outs = detach(outs)
            lbl = detach(lbl)
            for m in self.metric.values():
                m.update(outs, lbl)

        print('+ Evaluation result')
        avg_loss = running_loss.value()[0]
        print('Loss:', avg_loss)
        if not ema:
            self.val_loss.append(avg_loss)
            self.tsboard.update_loss('val', avg_loss, epoch)

            for k in self.metric.keys():
                m = self.metric[k].value()
                self.metric[k].summary()
                self.val_metric[k].append(m)
                self.tsboard.update_metric('val', k, m, epoch)
        else:
            self.val_loss_ema.append(avg_loss)
            self.tsboard.update_loss('val_ema', avg_loss, epoch)

            for k in self.metric.keys():
                m = self.metric[k].value()
                self.metric[k].summary()
                self.val_metric_ema[k].append(m)
                self.tsboard.update_metric('val_ema', k, m, epoch)


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

            # 2: Evalutation phase
            if (epoch + 1) % self.val_step == 0:
                # 2: Evaluating model
                self.val_epoch(epoch, dataloader=val_dataloader)
                if self.model_ema:
                    self.val_epoch(epoch, dataloader=val_dataloader, ema=True)
                print('-----------------------------------')

                # 3: Learning rate scheduling
                # unless it is ReduceLROnPlateau, we don't need to pass the metric value
                # self.scheduler.step(self.val_loss[-1])
                self.scheduler.step()

                # 4: Saving checkpoints
                if not self.debug:
                    # Get latest val loss here
                    val_loss = self.val_loss[-1]
                    val_metric = {k: m[-1] for k, m in self.val_metric.items()}
                    if self.model_ema:
                        val_loss_ema = self.val_loss_ema[-1]
                        val_metric_ema = {k: m[-1] for k, m in self.val_metric_ema.items()}
                        self.save_checkpoint(epoch, val_loss, val_metric,val_loss_ema, val_metric_ema)
                    else:
                        self.save_checkpoint(epoch, val_loss, val_metric)