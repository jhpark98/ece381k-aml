import gc
import ctypes
import torch
import numpy as np
import pandas as pd
import math
from math import pi, sqrt, exp

# Config and paths; Adjust as necessary
class PATHS:
    MAIN_DIR = "/kaggle/input/child-mind-institute-detect-sleep-states/"
    SPLIT_DIR = "/kaggle/input/child-sleep-mind-split-train/"
    
    SUBMISSION = MAIN_DIR + "sample_submission.csv"
    TRAIN_EVENTS = MAIN_DIR + "train_events.csv"
    TRAIN_SERIES = MAIN_DIR + "train_series.parquet"
    TEST_SERIES = MAIN_DIR + "test_series.parquet"
    
    @staticmethod
    def get_series_filename(series_id):
        # Adjust path if needed. This was for a Kaggle splitting scenario
        return PATHS.SPLIT_DIR + f'{series_id}_test_series.parquet'

class CFG:
    DEMO_MODE = False
    VERBOSE = True
    SEED = 42
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Model related configs
    ARCH = [(2, 8,  2, 17),
            (8, 32,  2, 11),
            (32, 64, 2, 7)]
    IN_CHANNELS = 2
    HIDDEN_SIZE = 64
    KERNEL_SIZE = 25
    STRIDE = 2
    N_LAYERS = 5
    DCONV_PADDING = 5
    LEN_MULT = 2**len(ARCH)
    MAX_CHUNK_SIZE = 24*60*12
    SIGMA = 36

def clean_memory():
    gc.collect()
    ctypes.CDLL("libc.so.6").malloc_trim(0)
    torch.cuda.empty_cache()

def get_predictions(res_df, target, SIGMA):
    """
    Find local maxima for event predictions. Returns a list of [step, event, score].
    """
    q = res_df[target].max() * 0.1
    tmp = res_df.loc[res_df[target] > q].copy()
    tmp['gap'] = tmp['step'].diff()
    tmp = tmp[tmp['gap'] > 5*5]
    res = []
    for i in range(len(tmp) + 1):
        start_i = 0 if i == 0 else tmp['step'].iloc[i-1]
        end_i = tmp['step'].iloc[i] if i < len(tmp) else res_df['step'].max()
        v = res_df[(res_df['step'] > start_i) & (res_df['step'] < end_i)]
        if len(v) > 0 and v[target].max() > q:
            idx = v[target].idxmax()
            step = v.loc[idx, 'step']
            span = 3*SIGMA
            score = res_df[(res_df['step'] > step - span) & (res_df['step'] < step + span)][target].sum()
            res.append([step, target, score])
    return res

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
