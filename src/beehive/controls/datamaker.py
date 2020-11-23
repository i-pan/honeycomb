import logging
import numpy as np
import pandas as pd
import os, os.path as osp

from ..builder import build_dataset, build_dataloader


def get_train_val_test_splits(cfg, df): 
    i, o = cfg.data.inner_fold, cfg.data.outer_fold
    if isinstance(i, (int,float)):
        if cfg.local_rank == 0:
            logger = logging.getLogger('root')
            logger.info(f'<inner fold> : {i}')
            logger.info(f'<outer fold> : {o}')
        test_df = df[df.outer == o]
        df = df[df.outer != o]
        train_df = df[df[f'inner{o}'] != i]
        valid_df = df[df[f'inner{o}'] == i]
        valid_df = valid_df.drop_duplicates().reset_index(drop=True)
        test_df = test_df.drop_duplicates().reset_index(drop=True)
    else:
        if cfg.local_rank == 0:
            logger = logging.getLogger('root')
            logger.info('No inner fold specified ...')
            logger.info(f'<outer fold> : {o}')
        test_df = None
        train_df = df[df.outer != o]
        valid_df = df[df.outer == o]
        valid_df = valid_df.drop_duplicates().reset_index(drop=True)
    return train_df, valid_df, test_df


def prepend_filepath(lst, prefix): 
    return np.asarray([osp.join(prefix, item) for item in lst])


def get_train_val_dataloaders(cfg):
    INPUT_COL = 'filename'
    LABEL_COL = 'Target'

    df = pd.read_csv(cfg.data.annotations)
    
    train_df, valid_df, _ = get_train_val_test_splits(cfg, df)
    data_dir = cfg.data.data_dir
    train_inputs = prepend_filepath(train_df[INPUT_COL], data_dir)
    train_labels = train_df[LABEL_COL].values
    valid_inputs = prepend_filepath(valid_df[INPUT_COL], data_dir)
    valid_labels = valid_df[LABEL_COL].values

    train_dataset = build_dataset(cfg, 
        data_info=dict(inputs=train_inputs, labels=train_labels),
        mode='train')
    valid_dataset = build_dataset(cfg, 
        data_info=dict(inputs=valid_inputs, labels=valid_labels),
        mode='valid')

    if cfg.local_rank == 0:
        logger = logging.getLogger('root')
        logger.info(f'TRAIN : n={len(train_dataset)}')
        logger.info(f'VALID : n={len(valid_dataset)}')

    train_loader = build_dataloader(cfg,
        dataset=train_dataset,
        mode='train')
    valid_loader = build_dataloader(cfg,
        dataset=valid_dataset,
        mode='valid')

    return train_loader, valid_loader


