import torch
import pickle
import pandas as pd
import numpy as np
import os, os.path as osp
import re

from tqdm import tqdm
from . import metrics
from ..utils import cudaify


class Predictor:

    def __init__(self,
                 loader,
                 cuda=True,
                 debug=False):

        self.loader = loader
        self.cuda = cuda
        self.debug = debug

    def set_local_rank(self, local_rank=0):
        self.local_rank = local_rank

    def predict(self, model, criterion, epoch):
        self.epoch = epoch
        y_pred = []
        y_true = []
        losses = []
        if self.local_rank == 0:
            iterator = tqdm(enumerate(self.loader), total=len(self.loader))
        else:
            iterator = enumerate(self.loader)
        with torch.no_grad():
            for i, data in iterator:
                if self.debug:
                    if i > 10:
                        break
                batch, labels = data
                if self.cuda:
                    batch, labels = cudaify(batch, labels, device=self.local_rank)
                output = model(batch)
                if criterion:
                    losses += [criterion(output, labels).item()]
                output = torch.softmax(output, dim=1)
                y_pred += list(output.cpu().numpy())
                y_true += list(labels.cpu().numpy())

        y_pred = np.asarray(y_pred)
        y_true = np.asarray(y_true)

        return y_true, y_pred, losses


class Evaluator(Predictor):

    def __init__(self,
                 loader,
                 metrics,
                 valid_metric,
                 mode,
                 improve_thresh,
                 prefix,
                 save_checkpoint_dir,
                 save_best,
                 early_stopping=np.inf,
                 thresholds=np.arange(0.05, 1.05, 0.05),
                 cuda=True,
                 debug=False):
        
        super(Evaluator, self).__init__(
            loader=loader, 
            cuda=cuda,
            debug=debug)

        if type(metrics) is not list: metrics = list(metrics)

        # List of strings corresponding to desired metrics
        # These strings should correspond to function names defined
        # in metrics.py
        self.metrics = metrics
        # valid_metric should be included within metrics
        # This specifies which metric we should track for validation improvement
        self.valid_metric = valid_metric
        # Mode should be one of ['min', 'max']
        # This determines whether a lower (min) or higher (max) 
        # valid_metric is considered to be better
        self.mode = mode
        # This determines by how much the valid_metric needs to improve
        # to be considered an improvement
        self.improve_thresh = improve_thresh
        # Specifies part of the model name
        self.prefix = prefix
        self.save_checkpoint_dir = save_checkpoint_dir
        # save_best = True, overwrite checkpoints if score improves
        # If False, save all checkpoints
        self.save_best = save_best
        self.metrics_file = os.path.join(save_checkpoint_dir, 'metrics.csv')
        #if os.path.exists(self.metrics_file): os.system('rm {}'.format(self.metrics_file))
        # How many epochs of no improvement do we wait before stopping training?
        self.early_stopping = early_stopping
        self.stopping = 0
        self.thresholds = thresholds

        self.history = []
        self.epoch = None

        self.reset_best()

    def reset_best(self):
        self.best_model = None
        self.best_score = -np.inf

    def set_logger(self, logger):
        self.logger = logger
        self.print  = self.logger.info

    def validate(self, model, criterion, epoch, save_pickle):
        y_true, y_pred, losses = self.predict(model, criterion, epoch)
        # Save predictions
        if save_pickle:
            with open(osp.join(self.save_checkpoint_dir, f'.tmp_preds_rank{self.local_rank}.pkl'), 'wb') as f:
                pickle.dump({'y_true': y_true, 'y_pred': y_pred, 'losses': losses}, f)
            with open(osp.join(self.save_checkpoint_dir, f'.done_rank{self.local_rank}.txt'), 'w') as f:
                f.write('done')
        else:
            return y_true, y_pred, losses

    def generate_metrics_df(self):
        df = pd.concat([pd.DataFrame(d, index=[0]) for d in self.history])
        df.to_csv(self.metrics_file, index=False)

    # Used by Trainer class
    def check_stopping(self):
        return self.stopping >= self.early_stopping

    def check_improvement(self, score):
        # If mode is 'min', make score negative
        # Then, higher score is better (i.e., -0.01 > -0.02)
        multiplier = -1 if self.mode == 'min' else 1
        score = multiplier * score
        improved = score >= (self.best_score + self.improve_thresh)
        if improved:
            self.stopping = 0
            self.best_score = score
        else:
            self.stopping += 1
        return improved

    def save_checkpoint(self, model, valid_metric, y_true, y_pred):
        save_file = '{}_{}_VM-{:.4f}.pth'.format(self.prefix, str(self.epoch).zfill(3), valid_metric).upper()
        save_file = os.path.join(self.save_checkpoint_dir, save_file)
        if self.save_best:
            if self.check_improvement(valid_metric):
                if self.best_model is not None: 
                    os.system('rm {}'.format(self.best_model))
                self.best_model = save_file
                torch.save(model.state_dict(), save_file)
                # Save predictions
                # with open(os.path.join(self.save_checkpoint_dir, 'valid_predictions.pkl'), 'wb') as f:
                #     pickle.dump({'y_true': y_true, 'y_pred': y_pred}, f)
        else:
            torch.save(model.state_dict(), save_file)
            # Save predictions
            with open(os.path.join(self.save_checkpoint_dir, 'valid_predictions.pkl'), 'wb') as f:
                pickle.dump({'y_true': y_true, 'y_pred': y_pred}, f)
        # Save latest model to latest.pth
        torch.save(model.state_dict(), os.path.join(self.save_checkpoint_dir, 'latest.pth'))

    def calculate_metrics(self, y_true, y_pred, losses):
        metrics_dict = {}
        metrics_dict['loss'] = np.mean(losses)
        for metric in self.metrics:
            if metric == 'loss': continue
            metric = getattr(metrics, metric)
            metrics_dict.update(metric(y_true, y_pred, thresholds=self.thresholds))
        print_results = 'epoch {epoch} // VALIDATION'.format(epoch=self.epoch)
        if type(self.valid_metric) == list:
            valid_metric = np.mean([metrics_dict[vm] for vm in self.valid_metric])
        else:
            valid_metric = metrics_dict[self.valid_metric]
        metrics_dict['vm'] = valid_metric
        max_str_len = np.max([len(k) for k in metrics_dict.keys()])
        for key in metrics_dict.keys():
            self.print('{key} | {value:.5g}'.format(key=key.ljust(max_str_len), value=metrics_dict[key]))
        metrics_dict['epoch'] = int(self.epoch)
        self.history += [metrics_dict]
        self.generate_metrics_df()
        return valid_metric


