import torch
from torch import nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from configs.config import CFG
from .utils import PATHS, clean_memory, set_seed
from .dataset import SleepDatasetTrain
from .model import GRUNET  # Ensure model.py is created with GRUNET class inside
# import gc

def train_model():
    set_seed(CFG.SEED)
    
    # Load training events
    train_events = pd.read_csv(PATHS.TRAIN_EVENTS)
    series_ids = train_events.series_id.unique()
    # Filter or define continuous dictionary if needed
    # For simplicity, assume we skip that step or already have `continuous` dict
    continuous = None
    
    # Shuffle and split
    np.random.shuffle(series_ids)
    valid_prop = 0.2
    n_valid = int(len(series_ids)*valid_prop)
    valid_series_ids = series_ids[:n_valid]
    train_series_ids = series_ids[n_valid:]

    # Create datasets
    # Initially pick sigma as some value or schedule during training
    sigma = 90
    train_ds = SleepDatasetTrain(train_series_ids, events=train_events, len_mult=CFG.LEN_MULT, continuous=continuous, sigma=sigma)
    valid_ds = SleepDatasetTrain(valid_series_ids, events=train_events, len_mult=CFG.LEN_MULT, continuous=continuous, sigma=sigma)

    # Model, optimizer, scheduler
    net = GRUNET(arch=CFG.ARCH, out_channels=2, hidden_size=CFG.HIDDEN_SIZE, kernel_size=CFG.KERNEL_SIZE, 
                 stride=CFG.ARCH[-1][0], dconv_padding=CFG.DCONV_PADDING, n_layers=CFG.N_LAYERS, bidir=True, 
                 print_shape=False).to(CFG.DEVICE)
    
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=0.0)
    loss_fct = nn.KLDivLoss(reduction='sum')
    m = nn.LogSoftmax(dim=0)

    EPOCHS = 2  # For demonstration

    for epoch in range(EPOCHS):
        net.train()
        train_loss = 0
        for i in range(len(train_ds)):
            X, Y = train_ds[i]
            Y = Y.to(CFG.DEVICE, non_blocking=True)

            pred = torch.zeros(Y.shape).to(CFG.DEVICE, non_blocking=True)
            optimizer.zero_grad()

            h = None
            seq_len = X.shape[0]
            for j in range(0, seq_len, CFG.MAX_CHUNK_SIZE):
                X_chunk = X[j: j + CFG.MAX_CHUNK_SIZE].float().to(CFG.DEVICE, non_blocking=True)
                y_pred, h = net(X_chunk, h)
                h = [hi.detach() for hi in h]
                pred[j: j+CFG.MAX_CHUNK_SIZE, :] = y_pred
                del X_chunk, y_pred

            pred = m(pred.float())
            loss = loss_fct(pred.float(), Y.float())
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=2.0)
            optimizer.step()
            train_loss += loss.item()

            del pred, loss, Y, X, h
            clean_memory()

        train_loss /= len(train_ds)
        print(f"Epoch {epoch}, Train Loss = {train_loss}")

        # Validation
        net.eval()
        val_loss = 0
        with torch.no_grad():
            for i in tqdm(range(len(valid_ds)), desc=f"Epoch {epoch} [Valid]"):
                X, Y = valid_ds[i]
                Y = Y.to(CFG.DEVICE, non_blocking=True)

                pred = torch.zeros(Y.shape).to(CFG.DEVICE, non_blocking=True)

                h = None
                seq_len = X.shape[0]
                for j in range(0, seq_len, CFG.MAX_CHUNK_SIZE):
                    X_chunk = X[j: j + CFG.MAX_CHUNK_SIZE].float().to(CFG.DEVICE, non_blocking=True)
                    y_pred, h = net(X_chunk, h)
                    h = [hi.detach() for hi in h]
                    pred[j: j+CFG.MAX_CHUNK_SIZE, :] = y_pred
                    del X_chunk, y_pred
                
                pred = m(pred.float())
                loss = loss_fct(pred.float(), Y.float())
                val_loss += loss.item()

                del pred, loss, Y, X, h
                clean_memory()

        val_loss /= len(valid_ds)
        print(f"Epoch {epoch}, Valid Loss = {val_loss}")

        # Save model checkpoint
        torch.save(net.state_dict(), f"model_epoch{epoch}.pth")

if __name__ == "__main__":
    train_model()
