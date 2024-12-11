import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from math import pi, exp
from .utils import PATHS

class SleepDatasetTrain(Dataset):
    """
    Dataset for training or inference.
    If events and sigma are provided, targets (distributions) are generated.
    Otherwise, only features are returned.
    """
    def __init__(self, series_ids, events=None, len_mult=1, continuous=None, sigma=None):
        self.series_ids = series_ids
        self.events = events
        self.continuous = continuous
        self.len_mult = len_mult
        self.sigma = sigma

    def load_data(self, series_id):
        filename = PATHS.get_series_filename(series_id)
        data = pd.read_parquet(filename)

        if self.events is not None and self.sigma is not None:
            if self.continuous is not None:
                start, end = self.continuous[series_id]
            else:
                start, end = 0, 1000000
            gap = 6*60*12
            tmp = self.events[(self.events.series_id == series_id) & 
                              (self.events.night >= start) & (self.events.night <= end)]
            # Focus on region around events
            data = data[(data.step > (tmp.step.min() - gap)) & (data.step < (tmp.step.max() + gap))]

            data = data.set_index(['series_id', 'step']).join(tmp.set_index(['series_id', 'step'])[['event','night']]).reset_index()
            norm = 1/ np.sqrt(pi / self.sigma)
            for evt in ['wakeup', 'onset']:
                steps = data[data.event == evt]['step'].values
                col = f'{evt}_val'
                data[col] = 0.0
                for i in steps:
                    x = 0.5*((data.step.astype(np.int64) - i)/self.sigma)**2
                    data[col] += np.exp(-x)*norm
                if data[col].sum() > 0:
                    data[col] /= data[col].sum()

        n = int((len(data) // self.len_mult) * self.len_mult)
        return data.iloc[:n]

    def __len__(self):
        return len(self.series_ids)

    def __getitem__(self, index):
        series_id = self.series_ids[index]
        data = self.load_data(series_id)
        X = data[['anglez','enmo']].values.astype(np.float32)
        X = torch.from_numpy(X)

        if self.sigma is not None and self.events is not None:
            Y = data[['wakeup_val', 'onset_val']].values.astype(np.float32)
            Y = torch.from_numpy(Y)
            return X, Y
        else:
            return X
