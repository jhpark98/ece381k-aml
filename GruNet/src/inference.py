import torch
import pandas as pd
from tqdm.auto import tqdm

from .utils import PATHS, CFG, clean_memory, get_predictions
from .dataset import SleepDatasetTrain
from .model import GRUNET

def create_predictions(test_ds, i, net):
    net.eval()
    with torch.no_grad():
        series_id = test_ds.series_ids[i]
        data = test_ds.load_data(series_id)
        X = test_ds[i]
        pred = torch.zeros(X.shape).to(CFG.DEVICE, non_blocking=True)

        h = None
        seq_len = X.shape[0]
        for j in range(0, seq_len, CFG.MAX_CHUNK_SIZE):
            X_chunk = X[j: j + CFG.MAX_CHUNK_SIZE].float().to(CFG.DEVICE, non_blocking=True)
            y_pred, h = net(X_chunk, h)
            h = [hi.detach() for hi in h]

            # Optionally trim edges if desired
            v = torch.min(y_pred)
            y_pred[:10, :] = v
            y_pred[-10:, :] = v

            pred[j: j+CFG.MAX_CHUNK_SIZE, :] = y_pred
            del X_chunk, y_pred
        clean_memory()
    res_df = pd.DataFrame(torch.softmax(pred.cpu(), axis=0).numpy(), columns=['wakeup', 'onset'])   
    res_df['step'] = data['step'].values
    return res_df

def run_inference():
    # Suppose you have test series IDs and no events (for submission)
    # In demo mode, we load some subset or train data as test
    test_series_ids = ['series_id_1', 'series_id_2']  # Replace with actual test IDs

    test_ds = SleepDatasetTrain(test_series_ids, events=None, len_mult=CFG.LEN_MULT, sigma=None)

    # Load trained model
    net = GRUNET(arch=CFG.ARCH, out_channels=2, hidden_size=CFG.HIDDEN_SIZE, kernel_size=CFG.KERNEL_SIZE, 
                 stride=CFG.ARCH[-1][0], dconv_padding=CFG.DCONV_PADDING, n_layers=CFG.N_LAYERS, bidir=True, 
                 print_shape=False).to(CFG.DEVICE)
    net.load_state_dict(torch.load("model_best_mAP0.pth", map_location=CFG.DEVICE))
    net.eval()

    all_df = []
    for i in tqdm(range(len(test_ds))):
        series_id = test_ds.series_ids[i]
        res_df = create_predictions(test_ds, i, net)

        onset_pred = get_predictions(res_df, 'onset', CFG.SIGMA)
        wakeup_pred = get_predictions(res_df, 'wakeup', CFG.SIGMA)
        pred_df = pd.DataFrame(wakeup_pred + onset_pred, columns=['step', 'event', 'score'])
        pred_df['series_id'] = series_id
        pred_df['row_id'] = pred_df.index
        pred_df = pred_df.sort_values(by='step').drop_duplicates(subset='step').reset_index(drop=True)

        # Filter out events too close to borders if desired
        min_step, max_step = res_df.step.min(), res_df.step.max()
        pred_df = pred_df[(pred_df.step > min_step + 12*60) & (pred_df.step < max_step - 12*60)]

        all_df.append(pred_df)
        clean_memory()

    final_df = pd.concat(all_df).reset_index(drop=True)
    final_df['row_id'] = final_df.index
    final_df = final_df[['row_id', 'series_id', 'step', 'event', 'score']]
    final_df.to_csv('submission.csv', index=False)
    print("Inference complete. Saved submission.csv.")

if __name__ == "__main__":
    run_inference()
