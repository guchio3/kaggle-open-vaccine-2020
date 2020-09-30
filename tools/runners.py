import copy
import datetime
import gc
import os
import random
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
from tools.metrics import MCRMSE
from tools.models import EMA, guchioGRU1
from tools.schedulers import pass_scheduler
from tools.splitters import mySplitter
# from transformers import RobertaForMaskedLM
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler

random.seed(71)
torch.manual_seed(71)


class r001BaseRunner(object):
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
        # self.cfg_dataset = config['dataset']
        self.cfg_fobj = config['fobj']
        self.cfg_model = config['model']
        self.cfg_optimizer = config['optimizer']
        self.cfg_scheduler = config['scheduler']
        self.cfg_train = config['train']
        if config['features'] is None:
            self.cfg_features = []
        else:
            self.cfg_features = config['features']

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
        trn_df['reactivity_mean'] = trn_df['reactivity'].apply(
            lambda x: np.mean(x))
        if self.cfg_train['sn_filter']:
            trn_df = trn_df[trn_df.SN_filter == 1].reset_index(drop=True)
        trn_df = trn_df[trn_df.signal_to_noise >=
                        self.cfg_train['signal_to_noise_thresh']]\
            .reset_index(drop=True)

        # split data
        splitter = mySplitter(**self.cfg_split, logger=self.logger)
        fold = splitter.split(
            trn_df,
            trn_df['reactivity_mean'],
            group=trn_df['id']
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

            # if self.cfg_train['pseudo']:
            #     fold_trn_df = pd.concat([fold_trn_df,
            #                              pd.read_csv(self.cfg_train['pseudo'][fold_num])],
            #                             axis=0).reset_index(drop=True)

            trn_loader = self._build_loader(mode='train', df=fold_trn_df,
                                            **self.cfg_loader)
            fold_val_df = trn_df.iloc[val_idx]
            val_loader = self._build_loader(mode='test', df=fold_val_df,
                                            **self.cfg_loader)

            # get fobj
            fobj = self._get_fobj(**self.cfg_fobj)

            # build model and related objects
            # these objects have state
            model = self._get_model(**self.cfg_model,
                                    num_features=len(self.cfg_features))
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
            epoch_best_score = np.inf
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

                trn_loss = self._train_loop(model, optimizer, fobj,
                                            trn_loader, warmup_batch,
                                            ema, accum_mod)
                ema.on_epoch_end(model)
                if self.cfg_train['ema_n'] > 0:
                    ema.set_weights(ema_model)  # NOTE: model?
                else:
                    ema_model = model
                val_loss, val_ids, val_preds, val_labels = \
                    self._valid_loop(ema_model, fobj, val_loader)
                epoch_best_score = min(epoch_best_score, val_loss)

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
                                      val_ids, val_preds, val_labels, val_loss)

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
                f'epoch best score: {epoch_best_score:.5f}'
            self.logger.send_line_notification(line_message)

            if self.cfg_SINGLE_FOLD:
                self.logger.info('stop training at the end of the first fold!')
                break

        fold_best_losses = []
        if self.cfg_SINGLE_FOLD:
            loss_mean = min(self.histories[0]['val_loss'])
            loss_std = 0.
            fold_best_losses = [loss_mean, ]
        else:
            for fold_num in range(self.cfg_split['split_num']):
                fold_best_losses.append(
                    min(self.histories[fold_num]['val_loss']))
            loss_mean = np.mean(fold_best_losses)
            loss_std = np.std(fold_best_losses)

        trn_time = int(time.time() - trn_start_time) // 60
        line_message = \
            f'----------------------- \n' \
            f'{self.exp_id}: {self.description} \n' \
            f'loss      : {loss_mean:.5f}+-{loss_std:.5f} \n' \
            f'best_losses    : {fold_best_losses} \n' \
            f'time         : {trn_time} min \n' \
            f'-----------------------'
        self.logger.send_line_notification(line_message)

    def predict(self):
        # load and preprocess train.csv
        tst_df = pd.read_json('./inputs/origin/test.json', lines=True)
        pub_tst_df = tst_df.query('seq_scored == 68')
        pri_tst_df = tst_df.query('seq_scored == 91')

        pub_tst_loader = self._build_loader(mode='test', df=pub_tst_df,
                                            **self.cfg_loader)
        pri_tst_loader = self._build_loader(mode='test', df=pri_tst_df,
                                            **self.cfg_loader)
        # build model and related objects
        # these objects have state
        model = self._get_model(**self.cfg_model,
                                num_features=len(self.cfg_features))

        ckpt_filenames = glob(f'./checkpoints/{self.exp_id}/best/*')

        tst_sub_ids_list = []
        tst_sub_preds_list = []
        best_val_scores = []
        for ckpt_filename in tqdm(ckpt_filenames):
            best_val_scores.append(
                float(ckpt_filename.split('/')[-1].split('_')[4]))
            checkpoint = torch.load(ckpt_filename)
            if self.device == 'cuda':
                model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint['model_state_dict'])

            # send to device
            model = model.to(self.device)

            pub_tst_ids, pub_tst_preds = self._test_loop(
                model, pub_tst_loader, seq_len=107)
            pri_tst_ids, pri_tst_preds = self._test_loop(
                model, pri_tst_loader, seq_len=130)

            pub_tst_sub_ids, pub_tst_sub_preds = self._explode_preds(
                pub_tst_ids, pub_tst_preds)
            pri_tst_sub_ids, pri_tst_sub_preds = self._explode_preds(
                pri_tst_ids, pri_tst_preds)

            tst_sub_ids = np.concatenate(
                [pub_tst_sub_ids, pri_tst_sub_ids], axis=0)
            tst_sub_preds = np.concatenate(
                [pub_tst_sub_preds, pri_tst_sub_preds], axis=0)

            tst_sub_ids_list.append(tst_sub_ids)
            tst_sub_preds_list.append(tst_sub_preds)

        res_tst_sub_ids = tst_sub_ids_list[0]
        res_tst_sub_preds = np.mean(tst_sub_preds_list, axis=0)

        sub_df = pd.DataFrame()
        sub_df['id_seqpos'] = res_tst_sub_ids
        sub_df['reactivity'] = res_tst_sub_preds[:, 0]
        sub_df['deg_Mg_pH10'] = res_tst_sub_preds[:, 1]
        sub_df['deg_Mg_50C'] = res_tst_sub_preds[:, 2]
        sub_df['deg_pH10'] = 0.
        sub_df['deg_50C'] = 0.

        save_filename = f'./submissions/{self.exp_id}' \
                        f'_{float(np.mean(best_val_scores)):.5f}' \
                        f'+-{float(np.std(best_val_scores)):.5f}_sub.csv'
        sub_df.to_csv(save_filename, index=False)

    def _train_loop(self, model, optimizer, fobj,
                    loader, warmup_batch, ema, accum_mod):
        model.train()
        running_loss = 0

        for batch_i, batch in enumerate(tqdm(loader)):
            if warmup_batch > 0:
                self._warmup(batch_i, warmup_batch, model)

            # ids = batch['ids'].to(self.device)
            encoded_sequence = batch['encoded_sequence'].to(self.device)
            encoded_structure = batch['encoded_structure'].to(self.device)
            encoded_predicted_loop_type = batch['encoded_predicted_loop_type']\
                .to(self.device)
            features = []
            for feature_name in self.cfg_features:
                features.append(batch[feature_name].to(self.device))
            # reactivity_error = batch['reactivity_error'].to(self.device)
            # deg_error_Mg_pH10 = batch['deg_error_Mg_pH10'].to(self.device)
            # deg_error_pH10 = batch['deg_error_pH10'].to(self.device)
            # deg_error_Mg_50C = batch['deg_error_Mg_50C'].to(self.device)
            # deg_error_50C = batch['deg_error_50C'].to(self.device)
            reactivity = batch['reactivity'].to(self.device)
            deg_Mg_pH10 = batch['deg_Mg_pH10'].to(self.device)
            # deg_pH10 = batch['deg_pH10'].to(self.device)
            deg_Mg_50C = batch['deg_Mg_50C'].to(self.device)
            # deg_50C = batch['deg_50C'].to(self.device)

            labels = torch.stack([reactivity, deg_Mg_pH10, deg_Mg_50C], dim=-1)
            # labels = torch.transpose(
            #     torch.cat([reactivity, deg_Mg_pH10, deg_Mg_50C]), 0, 1)
            # torch.cat([reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C,
            # deg_50C]), 0, 1)

            logits = model(encoded_sequence,
                           encoded_structure,
                           encoded_predicted_loop_type,
                           features)
            logits = logits[:, :68, :]

            train_loss = fobj(logits, labels)

            train_loss.backward()

            running_loss += train_loss.item()

            if (batch_i + 1) % accum_mod == 0:
                optimizer.step()
                optimizer.zero_grad()

                ema.on_batch_end(model)

        train_loss = running_loss / len(loader)

        return train_loss

    def _valid_loop(self, model, fobj, loader):
        model.eval()
        running_loss = 0

        with torch.no_grad():
            valid_ids = []
            valid_preds = []
            valid_labels = []

            for batch in tqdm(loader):
                id = batch['id']
                encoded_sequence = batch['encoded_sequence'].to(self.device)
                encoded_structure = batch['encoded_structure'].to(self.device)
                encoded_predicted_loop_type = \
                    batch['encoded_predicted_loop_type'].to(self.device)
                features = []
                for feature_name in self.cfg_features:
                    features.append(batch[feature_name].to(self.device))
                # reactivity_error = batch['reactivity_error'].to(self.device)
                # deg_error_Mg_pH10 = batch['deg_error_Mg_pH10'].to(self.device)
                # deg_error_pH10 = batch['deg_error_pH10'].to(self.device)
                # deg_error_Mg_50C = batch['deg_error_Mg_50C'].to(self.device)
                # deg_error_50C = batch['deg_error_50C'].to(self.device)
                reactivity = batch['reactivity'].to(self.device)
                deg_Mg_pH10 = batch['deg_Mg_pH10'].to(self.device)
                # deg_pH10 = batch['deg_pH10'].to(self.device)
                deg_Mg_50C = batch['deg_Mg_50C'].to(self.device)
                # deg_50C = batch['deg_50C'].to(self.device)

                labels = torch.stack(
                    [reactivity, deg_Mg_pH10, deg_Mg_50C], dim=-1)
                # labels = torch.transpose(
                #     torch.cat([reactivity, deg_Mg_pH10, deg_Mg_50C]), 0, 1)
                # torch.cat([reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C,
                # deg_50C]), 0, 1)

                logits = model(encoded_sequence,
                               encoded_structure,
                               encoded_predicted_loop_type,
                               features)
                logits = logits[:, :68, :]

                valid_loss = fobj(logits, labels)

                running_loss += valid_loss.item()

                valid_ids.append(id)
                valid_preds.append(logits)
                valid_labels.append(labels)

            valid_loss = running_loss / len(loader)

            valid_ids = list(chain.from_iterable(valid_ids))
            valid_preds = list(chain.from_iterable(valid_preds))
            valid_labels = list(chain.from_iterable(valid_labels))

        return valid_loss, valid_ids, valid_preds, valid_labels

    def _test_loop(self, model, loader, seq_len):
        model.eval()

        with torch.no_grad():
            test_ids = []
            test_preds = []

            for batch in tqdm(loader):
                id = batch['id']
                encoded_sequence = batch['encoded_sequence'].to(self.device)
                encoded_structure = batch['encoded_structure'].to(self.device)
                encoded_predicted_loop_type = \
                    batch['encoded_predicted_loop_type'].to(self.device)
                features = []
                for feature_name in self.cfg_features:
                    features.append(batch[feature_name].to(self.device))

                logits = model(encoded_sequence,
                               encoded_structure,
                               encoded_predicted_loop_type,
                               features)
                logits = logits[:, :seq_len, :]

                test_ids.append(id)
                test_preds.append(logits.cpu().numpy())

            test_ids = list(chain.from_iterable(test_ids))
            test_preds = list(chain.from_iterable(test_preds))

        return test_ids, test_preds

    def _explode_preds(self, test_ids, test_preds):
        exploded_test_ids, exploded_test_preds = [], []
        for test_id, test_pred in zip(test_ids, test_preds):
            for i, test_pred_i in enumerate(test_pred):
                exploded_test_ids.append(f'{test_id}_{i}')
                exploded_test_preds.append(test_pred_i)
        return exploded_test_ids, exploded_test_preds

    def _get_fobj(self, fobj_type):
        if fobj_type == 'mcrmse':
            fobj = MCRMSE
        else:
            raise Exception(f'invalid fobj_type: {fobj_type}')
        return fobj

    def _get_model(self, model_type, num_layers, embed_dropout, dropout,
                   num_embeddings, embed_dim, out_dim, num_features,
                   num_trans_layers, num_trans_attention_heads):
        if model_type == 'guchio_gru_1':
            model = guchioGRU1(
                num_layers=num_layers,
                embed_dropout=embed_dropout, dropout=dropout,
                num_embeddings=num_embeddings, embed_dim=embed_dim,
                out_dim=out_dim, num_features=num_features,
                num_trans_layers=num_trans_layers,
                num_trans_attention_heads=num_trans_attention_heads,
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
                                         debug=self.debug,)
            #                             **self.cfg_dataset)
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
                if 'WU' in name:
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
        self.logger.info('Now searching the best filename...')
        best_loss = np.inf
        best_filename = ''
        for filename in glob(f'./checkpoints/{self.exp_id}/{fold_num}/*'):
            split_filename = filename.split('/')[-1].split('_')
            temp_loss = float(split_filename[4])
            if temp_loss < best_loss:
                best_filename = filename
                best_loss = temp_loss
        if best_filename == '':
            raise Exception("Could't find the best filename.")
        else:
            self.logger.info(f'Hit! the best filename is {best_filename}')
            self.logger.info(f'     the best loss is {best_loss}')
        return best_filename

    def _load_best_checkpoint(self, fold_num):
        best_cp_filename = self._search_best_filename(fold_num)
        self.logger.info(f'the best file is {best_cp_filename} !')
        best_checkpoint = torch.load(best_cp_filename)
        return best_checkpoint
