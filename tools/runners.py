import copy
import datetime
import itertools
import os
import gc
import random
import re
import time
from glob import glob
from itertools import chain

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.optim as optim
from tools.datasets import OpenVaccineDataset
from tools.loggers import myLogger
# from tools.metrics import 
from tools.models import EMA, guchioGRU1
from tools.schedulers import pass_scheduler
from tools.splitters import mySplitter
from torch.utils.data import DataLoader
from torch.utils.data.sampler import (RandomSampler, SequentialSampler,
                                      WeightedRandomSampler)
# from transformers import RobertaForMaskedLM
from torch.nn import MSELoss

random.seed(71)
torch.manual_seed(71)


class Runner(object):
    def __init__(self, exp_id, checkpoint, device,
                 debug, config, default_config):
        # set logger
        self.exp_time = datetime\
            .datetime.now()\
            .strftime('%Y-%m-%d-%H-%M-%S')

        self.exp_id = exp_id
        self.checkpoint = checkpoint
        self.device = device
        self.debug = debug
        self.logger = myLogger(f'./logs/{self.exp_id}.log')

        # set default configs
        self._fill_config_by_default_config(config, default_config)

        # log info
        self.logger.info(f'exp_id: {exp_id}')
        self.logger.info(f'checkpoint: {checkpoint}')
        self.logger.info(f'debug: {debug}')
        self.logger.info(f'config: {config}')

        # unpack config info
        self.description = config['description']
        # uppercase means raaaw value
        self.cfg_SINGLE_FOLD = config['SINGLE_FOLD']
        self.cfg_split = config['split']
        self.cfg_loader = config['loader']
        self.cfg_dataset = config['dataset']
        self.cfg_fobj = config['fobj']
        self.cfg_model = config['model']
        self.cfg_optimizer = config['optimizer']
        self.cfg_scheduler = config['scheduler']
        self.cfg_train = config['train']
        self.cfg_predict = config['predict']

        self.histories = {
            'train_loss': [],
            'valid_loss': [],
            'valid_acc': [],
        }

    def _fill_config_by_default_config(self, config, default_config):
        for (d_key, d_value) in default_config.items():
            if d_key not in config:
                message = f' --- fill {d_key} by dafault values, ' \
                          f'{d_value} ! --- '
                self.logger.warning(message)
                config[d_key] = d_value
            elif isinstance(d_value, dict):
                self._fill_config_by_default_config(config[d_key], d_value)

    def train(self):
        trn_start_time = time.time()
        # load and preprocess train.csv
        trn_df = pd.read_json('./inputs/origin/train.json', lines=True)

        # split data
        splitter = mySplitter(**self.cfg_split, logger=self.logger)
        fold = splitter.split(
            trn_df['id'],
            trn_df[trn_df.columns],
            group=trn_df['sentiment']
        )

        # load and apply checkpoint if needed
        if self.checkpoint:
            self.logger.info(f'loading checkpoint from {self.checkpoint} ...')
            checkpoint = torch.load(self.checkpoint)
            checkpoint_fold_num = checkpoint['fold_num']
            self.histories = checkpoint['histories']

        for fold_num, (trn_idx, val_idx) in enumerate(fold):
            if (self.checkpoint and fold_num < checkpoint_fold_num) \
               or (self.checkpoint and fold_num == checkpoint_fold_num
                   and checkpoint_fold_num == self.cfg_train['max_epoch'] - 1):
                self.logger.info(f'pass fold {fold_num}')
                continue

            if fold_num not in self.histories:
                self.histories[fold_num] = {
                    'trn_loss': [],
                    'val_loss': [],
                    'val_jac': [],
                }

            if self.debug:
                trn_idx = trn_idx[:self.cfg_loader['trn_batch_size'] * 3]
                val_idx = val_idx[:self.cfg_loader['tst_batch_size'] * 3]

            # build loader
            fold_trn_df = trn_df.iloc[trn_idx]

            if self.cfg_train['pseudo']:
                fold_trn_df = pd.concat([fold_trn_df,
                                         pd.read_csv(self.cfg_train['pseudo'][fold_num])],
                                        axis=0).reset_index(drop=True)

            trn_loader = self._build_loader(mode='train', df=fold_trn_df,
                                            **self.cfg_loader)
            fold_val_df = trn_df.iloc[val_idx]
            val_loader = self._build_loader(mode='test', df=fold_val_df,
                                            **self.cfg_loader)

            # get fobj
            fobj = self._get_fobj(**self.cfg_fobj)

            # build model and related objects
            # these objects have state
            model = self._get_model(**self.cfg_model)
            module = model if self.device == 'cpu' else model.module
            optimizer = self._get_optimizer(model=model, **self.cfg_optimizer)
            scheduler = self._get_scheduler(optimizer=optimizer,
                                            max_epoch=self.cfg_train['max_epoch'],
                                            **self.cfg_scheduler)
            if self.checkpoint and checkpoint_fold_num == fold_num:
                module.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                checkpoint_epoch = checkpoint['current_epoch']
                iter_epochs = range(checkpoint_epoch,
                                    self.cfg_train['max_epoch'], 1)
            else:
                checkpoint_epoch = -1
                iter_epochs = range(0, self.cfg_train['max_epoch'], 1)

            epoch_start_time = time.time()
            epoch_best_jaccard = -1
            self.logger.info('start trainging !')
            for current_epoch in iter_epochs:
                if self.checkpoint and current_epoch <= checkpoint_epoch:
                    print(f'pass epoch {current_epoch}')
                    continue

                start_time = time.time()
                # send to device
                model = model.to(self.device)
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(self.device)

                self._warmup(current_epoch, self.cfg_train['warmup_epoch'],
                             model)

                warmup_batch = self.cfg_train['warmup_batch'] if current_epoch == 0 else 0

                ema_model = copy.deepcopy(model)
                ema_model.eval()
                ema = EMA(model=ema_model,
                          mu=self.cfg_train['ema_mu'],
                          level=self.cfg_train['ema_level'],
                          n=self.cfg_train['ema_n'])

                if isinstance(self.cfg_train['accum_mod'], int):
                    accum_mod = self.cfg_train['accum_mod']
                elif isinstance(self.cfg_train['accum_mod'], list):
                    accum_mod = self.cfg_train['accum_mod'][current_epoch]
                else:
                    raise NotImplementedError('accum_mod')

                trn_loss = self._train_loop(
                    model, optimizer, fobj, trn_loader, warmup_batch,
                    ema, accum_mod,
                    self.cfg_train['loss_weight_type'],
                    self.cfg_train['use_dist_loss'],
                    self.cfg_train['single_word'])
                ema.on_epoch_end(model)
                if self.cfg_train['ema_n'] > 0:
                    ema.set_weights(ema_model)  # NOTE: model?
                else:
                    ema_model = model
                val_loss, val_ids, val_preds, val_labels = \
                    self._valid_loop(ema_model, fobj, val_loader,
                                     self.cfg_train['loss_weight_type'],
                                     self.cfg_predict['single_word'])
                epoch_best_score = max(epoch_best_score, val_loss)

                self.logger.info(
                    f'epoch: {current_epoch} / '
                    + f'trn loss: {trn_loss:.5f} / '
                    + f'val loss: {val_loss:.5f} / '
                    + f'lr: {optimizer.param_groups[0]["lr"]:.6f} / '
                    + f'accum_mod: {accum_mod} / '
                    + f'time: {int(time.time()-start_time)}sec')

                self.histories[fold_num]['trn_loss'].append(trn_loss)
                self.histories[fold_num]['val_loss'].append(val_loss)

                scheduler.step()

                # send to cpu
                ema_model = ema_model.to('cpu')
                # model = model.to('cpu')
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cpu()

                self._save_checkpoint(fold_num, current_epoch,
                                      ema_model, optimizer, scheduler,
                                      val_preds, val_labels, val_loss)

            best_filename = self._search_best_filename(fold_num)
            if not os.path.exists(f'./checkpoints/{self.exp_id}/best'):
                os.mkdir(f'./checkpoints/{self.exp_id}/best')
            os.rename(
                best_filename,
                f'./checkpoints/{self.exp_id}/best/{best_filename.split("/")[-1]}')
            left_files = glob(f'./checkpoints/{self.exp_id}/{fold_num}/*')
            for left_file in left_files:
                os.remove(left_file)

            fold_time = int(time.time() - epoch_start_time) // 60
            line_message = f'{self.exp_id}: {self.description} \n' \
                f'fini fold {fold_num} in {fold_time} min. \n' \
                f'epoch best jaccard: {epoch_best_jaccard}'
            self.logger.send_line_notification(line_message)

            if self.cfg_SINGLE_FOLD:
                break

        fold_best_jacs = []
        for fold_num in range(self.cfg_split['split_num']):
            fold_best_jacs.append(max(self.histories[fold_num]['val_jac']))
        jac_mean = np.mean(fold_best_jacs)
        jac_std = np.std(fold_best_jacs)

        trn_time = int(time.time() - trn_start_time) // 60
        line_message = \
            f'----------------------- \n' \
            f'{self.exp_id}: {self.description} \n' \
            f'jaccard      : {jac_mean:.5f}+-{jac_std:.5f} \n' \
            f'best_jacs    : {fold_best_jacs} \n' \
            f'time         : {trn_time} min \n' \
            f'-----------------------'
        self.logger.send_line_notification(line_message)

    def _get_fobj(self, fobj_type):
        if fobj_type == 'mse':
            fobj = MSELoss()
        else:
            raise Exception(f'invalid fobj_type: {fobj_type}')
        return fobj

    def _get_model(self, model_type, num_output_units,
                   pretrained_model_name_or_path):
        if model_type == 'guchio_gru_1':
            model = guchioGRU1(
                num_output_units,
                pretrained_model_name_or_path
            )
        else:
            raise Exception(f'invalid model_type: {model_type}')
        if self.device == 'cpu':
            return model
        else:
            return torch.nn.DataParallel(model)

    def _get_optimizer(self, optim_type, lr, model):
        if optim_type == 'sgd':
            optimizer = optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=0.9,
                # weight_decay=1e-4,
                nesterov=True,
            )
        elif optim_type == 'adam':
            optimizer = optim.Adam(
                model.parameters(),
                lr=lr,
            )
        elif optim_type == 'rmsprop':
            optimizer = optim.RMSprop(
                model.parameters(),
                lr=lr,
                momentum=0.9,
            )
        else:
            raise Exception(f'invalid optim_type: {optim_type}')
        return optimizer

    def _get_scheduler(self, scheduler_type, max_epoch,
                       optimizer, every_step_unit, cosine_eta_min,
                       multistep_milestones, multistep_gamma):
        if scheduler_type == 'pass':
            scheduler = pass_scheduler()
        elif scheduler_type == 'every_step':
            scheduler = optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda epoch: every_step_unit**epoch,
            )
        elif scheduler_type == 'multistep':
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=multistep_milestones,
                gamma=multistep_gamma
            )
        elif scheduler_type == 'cosine':
            # scheduler examples:
            #     [http://katsura-jp.hatenablog.com/entry/2019/01/30/183501]
            # if you want to use cosine annealing, use below scheduler.
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max_epoch - 1, eta_min=cosine_eta_min
            )
        else:
            raise Exception(f'invalid scheduler_type: {scheduler_type}')
        return scheduler

    def _build_loader(self, mode, df,
                      trn_sampler_type, trn_batch_size,
                      tst_sampler_type, tst_batch_size,
                      dataset_type, neutral_weight=1.,
                      longer_posneg_rate=1.):
        if mode == 'train':
            sampler_type = trn_sampler_type
            batch_size = trn_batch_size
            drop_last = True
        elif mode == 'test':
            sampler_type = tst_sampler_type
            batch_size = tst_batch_size
            drop_last = False
        else:
            raise NotImplementedError('mode {mode} is not valid for loader')

        if dataset_type == 'open_vaccine_dataset':
            dataset = OpenVaccineDataset(mode=mode, df=df,
                                         logger=self.logger,
                                         debug=self.debug,
                                         **self.cfg_dataset)
        else:
            raise NotImplementedError()

        if sampler_type == 'sequential':
            sampler = SequentialSampler(data_source=dataset)
        elif sampler_type == 'random':
            sampler = RandomSampler(data_source=dataset)
        else:
            raise NotImplementedError(
                f'sampler_type: {sampler_type} is not '
                'implemented for mode: {mode}')
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=os.cpu_count(),
            # num_workers=1,
            worker_init_fn=lambda x: np.random.seed(),
            drop_last=drop_last,
            pin_memory=True,
        )
        return loader

    def _warmup(self, current_epoch, warmup_batch_or_epoch, model):
        module = model if self.device == 'cpu' else model.module
        if current_epoch == 0:
            for name, child in module.named_children():
                if 'classifier' in name:
                    self.logger.info(name + ' is unfrozen')
                    for param in child.parameters():
                        param.requires_grad = True
                else:
                    self.logger.info(name + ' is frozen')
                    for param in child.parameters():
                        param.requires_grad = False
        if current_epoch == warmup_batch_or_epoch:
            self.logger.info("Turn on all the layers")
            # for name, child in model.named_children():
            for name, child in module.named_children():
                for param in child.parameters():
                    param.requires_grad = True

    def _save_checkpoint(self, fold_num, current_epoch,
                         model, optimizer, scheduler,
                         val_ids, val_preds, val_labels, val_loss):
        if not os.path.exists(f'./checkpoints/{self.exp_id}/{fold_num}'):
            os.makedirs(f'./checkpoints/{self.exp_id}/{fold_num}')
        # pth means pytorch
        cp_filename = f'./checkpoints/{self.exp_id}/{fold_num}/' \
            f'fold_{fold_num}_epoch_{current_epoch}_{val_loss:.5f}' \
            f'_checkpoint.pth'
        # f'_{val_metric:.5f}_checkpoint.pth'
        module = model if self.device == 'cpu' else model.module
        cp_dict = {
            'fold_num': fold_num,
            'current_epoch': current_epoch,
            'model_state_dict': module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_ids': val_ids,
            'val_preds': val_preds,
            'val_labels': val_labels,
            'histories': self.histories,
        }
        self.logger.info(f'now saving checkpoint to {cp_filename} ...')
        torch.save(cp_dict, cp_filename)

    def _search_best_filename(self, fold_num):
        best_loss = np.inf
        best_filename = ''
        for filename in glob(f'./checkpoints/{self.exp_id}/{fold_num}/*'):
            split_filename = filename.split('/')[-1].split('_')
            temp_loss = float(split_filename[2])
            if temp_loss < best_loss:
                best_filename = filename
                best_loss = temp_loss
        return best_filename

    def _load_best_checkpoint(self, fold_num):
        best_cp_filename = self._search_best_filename(fold_num)
        self.logger.info(f'the best file is {best_cp_filename} !')
        best_checkpoint = torch.load(best_cp_filename)
        return best_checkpoint
