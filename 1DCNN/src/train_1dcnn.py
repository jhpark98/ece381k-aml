"""
Created on April 2023

@author: kmat
"""

from distutils.log import debug
from operator import ge, is_
import os
import glob
import json
from re import A
from tracemalloc import start
from turtle import title
import warnings
import argparse
import sys
import random
import time

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.callbacks import Callback, LearningRateScheduler, CSVLogger, ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from train_utils.scheduler import lrs_wrapper_cos

from model.model import build_1d_model_multiclass, build_1d_model_multiclass_controledstride
from train_utils.dataloader_1d import load_dataset, get_dataset

D = True
SSLmodel = False
STRIDE = 1
# Ws=[1, 24*4, 24*8, 24*16]
Ws=[1, 4, 24, 24*4, 24*8, 24*16]
MULTIOUT = False
USECUTMIX = True
SEED = 111
class SleepAwake():
    def __init__(self, 
                 input_shape=(7168, 4), 
                 output_shape=(7168,1), 
                 num_cls=11,
                 weight_file=None, 
                 is_train_model=False,
                 model_type=None,
                 ):
        
        print("\rLoading Models...", end="")
        
        self.input_shape = tuple(input_shape)
        self.output_shape = tuple(output_shape)
        self.num_cls = num_cls
        self.is_train_model = is_train_model
        self.weight_file = weight_file
        self.model_type = MODELTYPE if model_type is None else model_type
        self.load_model(weight_file, is_train_model)
        print("Loading Models......Finish")
        
            
    def load_model(self, weight_file=None, is_train_model=False):
        """build model and load weights"""
        if SSLmodel:
            self.model, self.losses, self.loss_weights, self.metrics = build_1d_model_multiclass(input_shape=(None,self.input_shape[-1]), num_cls = self.num_cls, D=D, Ws=Ws, SSLmodel=SSLmodel, outstride=STRIDE)
        else:
            if self.model_type == "normal":
                self.model, self.model_infer, self.losses, self.loss_weights, self.metrics = build_1d_model_multiclass(input_shape=(None,self.input_shape[-1]), num_cls = self.num_cls, D=D, Ws=Ws, SSLmodel=SSLmodel, outstride=STRIDE)
            
            elif "contstride" in self.model_type:
                self.model, self.model_infer, self.losses, self.loss_weights, self.metrics = build_1d_model_multiclass_controledstride(input_shape=(None,self.input_shape[-1]), num_cls = self.num_cls, D=D, Ws=Ws, SSLmodel=SSLmodel, outstride=STRIDE, model_type="v3" if "v3" in self.model_type else "v2")

            # self.model, self.model_infer, self.losses, self.loss_weights, self.metrics = build_1d_model_multiclass(input_shape=(None,self.input_shape[-1]), num_cls = self.num_cls, D=D, Ws=Ws, SSLmodel=SSLmodel, outstride=STRIDE)
        if not weight_file is None:
            #if is_train_model:
            #    self.model.load_weights(weight_file, by_name=True, skip_mismatch=True)
            #else:
            if is_train_model:
                print("load SSL weights?")
                self.model.load_weights(weight_file, by_name=True, skip_mismatch=True)
            else:
                self.model.load_weights(weight_file)
        if not is_train_model:
            self.model.trainable = False
            # self.names_of_model_inputs = [inp.name for inp in self.sub_model.input]
            self.tf_model = tf.function(lambda x: self.model_infer(x))
            
            self.model = self.model_infer


    def train(self, train_dataset, val_dataset, save_dir, num_data, 
              learning_rate=0.002, n_epoch=150, batch_size=32, 
              ):
        if not self.is_train_model:
            raise ValueError("Model must be loaded as is_train_model=True")
        
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        
        lr_schedule = LearningRateScheduler(lrs_wrapper_cos(learning_rate, n_epoch, epoch_st=5))
        # lr_schedule = LearningRateScheduler(lrs_wrapper_cos_annealing(learning_rate, [int(n_epoch//3), n_epoch], epoch_wu=3))

        
        logger = CSVLogger(save_dir + 'log.csv')
        weight_file = "{epoch:02d}.hdf5"
        cp_callback = ModelCheckpoint(save_dir + weight_file, 
                                      monitor = 'val_loss', 
                                      save_weights_only = True,
                                      save_best_only = True,
                                      period = 50,
                                      verbose = 1)
                
        optim = Adam(lr=learning_rate, clipnorm=0.001)
        self.model.compile(loss = self.losses,
                           loss_weights = self.loss_weights, 
                           metrics = self.metrics,
                           optimizer = optim,
                           )
        
        print("step per epoch", num_data[0]//batch_size, num_data[1]//batch_size)
        

        self.hist = self.model.fit(get_dataset(train_dataset, 
                                               batch_size=batch_size, 
                                               input_shape=self.input_shape,
                                               is_train=True,
                                               SSLmodel=SSLmodel,
                                                outstride=STRIDE,
                                                use_cutmix=USECUTMIX,
                                                debug_mode=Cfg["IS_DEBUG"]=="True",), 
                    steps_per_epoch=num_data[0]//batch_size, 
                    epochs=n_epoch, 
                    validation_data=get_dataset(val_dataset, 
                                               batch_size=batch_size, 
                                               input_shape=self.input_shape,
                                               is_train=False,
                                               SSLmodel=SSLmodel,
                                                outstride=STRIDE,
                                                use_cutmix=USECUTMIX,
                                                debug_mode=Cfg["IS_DEBUG"]=="True",),
                    validation_steps=num_data[1]//batch_size,
                    callbacks=[lr_schedule, logger], 
                    )
        
        print("Saving weights and results...")
        self.model.save_weights(save_dir + "final_weights.h5")
        csv_hist = save_dir + "hist.csv"
        pd.DataFrame(self.hist.history).to_csv(csv_hist, index=False)
        print("Done")
    
    def split_inputs(self, inputs):
        """
        inputs: (h, w, NUM_CH)
        returns:
            outputs: (batch, input_shape[0], input_shape[1], NUM_CH)
            tlbrs: top left bottom right (batch, 4)

        """
        h, w, _ = inputs.shape
        outputs = []
        tlbrs = []
        for i in range(0, h-self.input_shape[0]+1, self.input_shape[0]):
            for j in range(0, w-self.input_shape[1]+1, self.input_shape[1]):
                outputs.append(inputs[i:i+self.input_shape[0], j:j+self.input_shape[1], :])
                tlbrs.append([i, j, i+self.input_shape[0], j+self.input_shape[1]])
        outputs = np.stack(outputs, axis=0)
        tlbrs = np.stack(tlbrs, axis=0)
        return outputs, tlbrs

    def preprocess(self, input_img):
        """
        input_img:already normalized to [0, 1]
        """
        input_img = tf.cast(input_img, tf.float32)
        return input_img

    def predict(self, inputs, targets=None,):
        """
        inputs: (data_length, num_ch)
        """
        # input_img = self.preprocess(input_img)
        data_length = inputs.shape[0]
        crop_length = (data_length//self.input_shape[0]) * self.input_shape[0]
        inputs = tf.reshape(tf.cast(inputs[:crop_length], tf.float32), [data_length//self.input_shape[0], self.input_shape[0], self.input_shape[-1]])
        # preds, preds_switch, _ = self.tf_model(inputs)
        preds, preds_switch, _ = self.tf_model(inputs)

        # 2ch(nan)使ってみる
        # preds_switch = preds_switch * (1-inputs[...,2:3])

        preds = tf.reshape(preds, [-1, self.output_shape[-1]])
        preds_switch = tf.reshape(preds_switch, [-1, 5])
        if targets is not None:
            targets = tf.cast(targets[:crop_length], tf.float32)
            targets = tf.reshape(targets, [-1, self.output_shape[-1]])
            return preds, preds_switch, targets
        else:
            return preds, preds_switch
    
    def overlap_predict(self, inputs, overlap=0.8, drop_step=240, tta=False):
        """
        inputs: (data_length, num_ch)
        edge prediction is a little unaccurate.
        overlap prediction is more accurate.
        """
        # input_img = self.preprocess(input_img)
        data_length = inputs.shape[0]
        max_step = int(self.input_shape[0] * (1-overlap)) 
        num_step = int(np.ceil((data_length - self.input_shape[0]) / max_step)) + 1
        starts = np.linspace(0, data_length-self.input_shape[0], num_step, dtype=np.int32, endpoint=True)
        starts = STRIDE * (starts // STRIDE)
        batch_inputs = []
        for start in starts:
            batch_inputs.append(inputs[start:start+self.input_shape[0]])
        batch_inputs = tf.cast(np.stack(batch_inputs, axis=0), tf.float32)
        preds, preds_switch, pred_nan = self.model.predict(batch_inputs)
        if tta:
            batch_tta_inputs = batch_inputs * tf.constant([[[1,-1,1,1,1,1,1,1,1,1]]], dtype=tf.float32)
            tta_preds, tta_preds_switch, _ = self.model.predict(batch_tta_inputs)
            preds = (preds + tta_preds) * 0.5
            preds_switch = (preds_switch + tta_preds_switch) * 0.5
        
        preds_flat = np.zeros((data_length//STRIDE))
        preds_switch_flat = np.zeros((data_length//STRIDE, preds_switch.shape[-1]))
        counts = np.zeros((data_length//STRIDE))
        starts = starts // STRIDE
        drop_step = drop_step // STRIDE
        out_length = self.input_shape[0] // STRIDE
        for i, start in enumerate(starts):
            preds_flat[(start+drop_step):(start+out_length-drop_step)] += preds[i,drop_step:-drop_step,0]
            preds_switch_flat[(start+drop_step):(start+out_length-drop_step)] += preds_switch[i,drop_step:-drop_step]
            counts[(start+drop_step):(start+out_length-drop_step)] += 1
            if i==0:
                counts[:drop_step] = 1
                preds_flat[:drop_step] = preds_flat[drop_step:(2*drop_step)].mean()
            if i==(len(starts)-1):
                counts[-drop_step:] = 1
                preds_flat[-drop_step:] = preds_flat[-(2*drop_step):-drop_step].mean()

        preds_flat /= counts
        preds_switch_flat /= counts.reshape(-1,1)
        return preds_flat, preds_switch_flat

    def overlap_predict_no_oom(self, inputs, overlap=0.8, drop_step=240, tta=False, max_batch = 64):
        """
        to avoid OOM on kaggle notebook
        """
        data_length = inputs.shape[0]
        max_step = int(self.input_shape[0] * (1-overlap)) 
        num_step = int(np.ceil((data_length - self.input_shape[0]) / max_step)) + 1
        starts = np.linspace(0, data_length-self.input_shape[0], num_step, dtype=np.int32, endpoint=True)
        starts_all = starts

        preds_flat = np.zeros((data_length))
        preds_switch_flat = np.zeros((data_length, 6))
        counts = np.zeros((data_length))
        final_batch_start = (len(starts) // max_batch) * max_batch
        for j in range(0, len(starts), max_batch):
            starts = starts_all[j:j+max_batch]

            batch_inputs = []
            for start in starts:
                batch_inputs.append(inputs[start:start+self.input_shape[0]])
            batch_inputs = tf.cast(np.stack(batch_inputs, axis=0), tf.float32)
            preds, preds_switch, _ = self.model.predict(batch_inputs)
        
            drop_step = drop_step
            out_length = self.input_shape[0]
            for i, start in enumerate(starts):
                preds_flat[(start+drop_step):(start+out_length-drop_step)] += preds[i,drop_step:-drop_step,0]
                preds_switch_flat[(start+drop_step):(start+out_length-drop_step)] += preds_switch[i,drop_step:-drop_step]
                counts[(start+drop_step):(start+out_length-drop_step)] += 1
                if i==0 and j==0:
                    counts[:drop_step] = 1
                    preds_flat[:drop_step] = preds_flat[drop_step:(2*drop_step)].mean()
                if i==(len(starts)-1) and j==final_batch_start:
                    counts[-drop_step:] = 1
                    preds_flat[-drop_step:] = preds_flat[-(2*drop_step):-drop_step].mean()
                    print("last batch")

        preds_flat /= counts
        preds_switch_flat /= counts.reshape(-1,1)
        return preds_flat, preds_switch_flat
        
def set_seeds(num=111):
    tf.random.set_seed(num)
    np.random.seed(num)
    random.seed(num)
    os.environ["PYTHONHASHSEED"] = str(num)
    

def run_training_main(epochs=20, 
                      batch_size=64,
                      input_shape=(7168, 4), 
                      output_shape=(7168, 1),
                      learning_rate=0.0005,
                      data_dir="data/data_processed", 
                      data_dir_gen=None, # "data/data_generated_060_120" 
                      gen_pretrain=False,
                      save_path="", 
                      load_path=None,
                      train_fold_no=0,
                      num_fold=2,
                      train_all=False):
    
    K.clear_session()
    set_seeds(111)
    
    files = load_dataset(data_dir, only_feature=True)    
    
    if False:
        files = files[::10]
    if SSLmodel or train_all: # すべてtrain
        print("Train all data")
        files_train = files
        files_val = files[::10]
    else:
        series_fold = pd.read_csv(os.path.join(Cfg["preprocess_dir"], f"series_id_{num_fold}fold_seed{SEED}.csv"))
        id2fold = {row["series_id"]: row["fold"] for i, row in series_fold.iterrows()}
        file_series_ids = [os.path.basename(f["file"]).split(".")[0].split("_")[1] for f in files]
        print(len(id2fold))
        print(len(file_series_ids))
        file_fold = [id2fold[series_id] for series_id in file_series_ids]
        files_train = [f for f, fold in zip(files, file_fold) if fold != train_fold_no]
        files_val = [f for f, fold in zip(files, file_fold) if fold == train_fold_no]

        
        # if num_fold == 2:
        #     series_fold = pd.read_csv("data/data_processed/series_id_2fold.csv")
        #     id2fold = {row["series_id"]: row["fold"] for i, row in series_fold.iterrows()}
        #     file_series_ids = [os.path.basename(f["file"]).split(".")[0].split("_")[1] for f in files]
        #     print(len(id2fold))
        #     print(len(file_series_ids))
        #     file_fold = [id2fold[series_id] for series_id in file_series_ids]
        #     files_train = [f for f, fold in zip(files, file_fold) if fold != train_fold_no]
        #     files_val = [f for f, fold in zip(files, file_fold) if fold == train_fold_no]

        #     """

        #     if train_fold_no == 0:
        #         # files_train = files[:int(len(files)*0.5)]
        #         # files_val = files[int(len(files)*0.5):]

        #         files_train = [f for f, fold in zip(files, file_fold) if fold != train_fold_no]
        #         files_val = [f for f, fold in zip(files, file_fold) if fold == train_fold_no]


        #         if gen_files is not None:
        #             print(len(files), len(gen_files))
        #             if gen_pretrain:
        #                 files_train = gen_files[:int(len(gen_files)*0.5)]
        #             else:
        #                 files_train += gen_files[:int(len(gen_files)*0.5)] # [:int(len(gen_files)*0.5)]
        #             # files_val += gen_files[int(len(gen_files)*0.5):] # valは元データのみ

        #     elif train_fold_no == 1:
        #         # files_train = files[int(len(files)*0.5):]
        #         # files_val = files[:int(len(files)*0.5)]
                
        #         files_train = [f for f, fold in zip(files, file_fold) if fold != train_fold_no]
        #         files_val = [f for f, fold in zip(files, file_fold) if fold == train_fold_no]
        #         if gen_files is not None:
        #             if gen_pretrain:
        #                 files_train = gen_files[int(len(gen_files)*0.5):]
        #             else:
        #                 files_train += gen_files[int(len(gen_files)*0.5):]
        #             # files_val += gen_files[:int(len(gen_files)*0.5)]
        #     else:
        #         raise Exception("train_fold_no must be 0 or 1")
        #     """
        # elif num_fold == 5:
        #     start = int(len(files) * train_fold_no / num_fold)
        #     end = int(len(files) * (train_fold_no+1) / num_fold)
        #     files_train = files[:start] + files[end:]
        #     files_val = files[start:end]
        #     if gen_files is not None:
        #         files_train += gen_files[:start] + gen_files[end:]
        #         # files_val += gen_files[start:end]
        # else:
        #     raise Exception("num_fold must be 2 or 5")
    print(len(files_train), len(files_val))
    print([f["length"] // input_shape[0] for f in files_train])
    np.random.shuffle(files_train)
    np.random.shuffle(files_val)
    num_cls = 11
    train_data_steps = sum([f["length"] // input_shape[0] for f in files_train])
    val_data_steps = sum([f["length"] // input_shape[0] for f in files_val])
    # train_data_steps = sum([f["end_step_for_train"] // input_shape[0] for f in files_train])
    # val_data_steps = sum([f["end_step_for_train"] // input_shape[0] for f in files_val])

    num_data = [train_data_steps, val_data_steps]
    print(num_data)
    model_params = {"input_shape": input_shape,
                    "output_shape": output_shape,  
                    "num_cls": num_cls,
                    "weight_file": load_path,
                    "is_train_model": True,
                    }
    train_params = {#"train_inputs": train_inputs,
                    #"train_targets": train_targets,
                    #"val_inputs": val_inputs,
                    #"val_targets": val_targets,
                    "train_dataset": files_train,
                    "val_dataset": files_val,
                    
                    "num_data": num_data,
                    "save_dir": save_path,
                    "learning_rate": learning_rate, 
                    "n_epoch": epochs, 
                    "batch_size": batch_size,
                    }  
    
    #with tf.device('/device:GPU:0'):
    scrl = SleepAwake(**model_params)#(**model_params)
    scrl.train(**train_params)



def run_prediction(input_shape=(7168, 4), 
                      output_shape=(7168, 1),
                      data_dir="data/data_processed", 
                      load_path="",
                      plot_length = None,
                      train_fold_no=0,
                      num_fold=2,
                      train_all=False,
                      overlap_ratio = 0.80):
    K.clear_session()
    set_seeds(111)
    
    files = load_dataset(data_dir)

    # series_fold = pd.read_csv(f"data/data_processed/series_id_{num_fold}fold.csv")
    series_fold = pd.read_csv(os.path.join(Cfg["preprocess_dir"],f"series_id_{num_fold}fold_seed{SEED}.csv"))

    id2fold = {row["series_id"]: row["fold"] for i, row in series_fold.iterrows()}
    file_series_ids = [os.path.basename(f["file"]).split(".")[0].split("_")[1] for f in files]
    print(len(id2fold))
    print(len(file_series_ids))
    file_fold = [id2fold[series_id] for series_id in file_series_ids]

    if train_all:
        print("Train all data validation all data")
        files_train = files
        files_val = files
    else:
        files_train = [f for f, fold in zip(files, file_fold) if fold != train_fold_no]
        files_val = [f for f, fold in zip(files, file_fold) if fold == train_fold_no]


    val_ids = [os.path.basename(f["file"]) for f in files_val]
    # val_idsを保存する
    with open(os.path.dirname(load_path) + "/val_ids.json", "w") as f:
        json.dump(val_ids, f)

    model_params = {"input_shape": input_shape,
                    "output_shape": output_shape,  
                    "weight_file": load_path,
                    "is_train_model": False,
                    }
    
    
    sa_model = SleepAwake(**model_params)#(**model_params)

    
    # wight_fileのディレクトリにplot保存のためのフォルダを作る
    if plot_length is not None:
        if not os.path.exists(os.path.dirname(load_path) + "/plot_1/"):
            os.makedirs(os.path.dirname(load_path) + "/plot_1/")
        
        for i, f in enumerate(files_val):
            offset = 0
            features = np.load(f["file"])[offset: (offset + plot_length*2)]
            steps = np.load(f["file_step"])[offset: (offset + plot_length*2)]
            inputs = features[:,:-2]
            targets = features[:,-1:]

            inp_nan = features[:, 2]


            # preds, preds_switch, targets = sa_model.predict(inputs, targets)
            preds, preds_switch, pred_nan = sa_model.overlap_predict(inputs, overlap=overlap_ratio)
            preds_switch =  (preds_switch[:,0] + preds_switch[:,1] + preds_switch[:,2]) * 0.33
            # 描画サイズを大きく
            plt.figure(figsize=(18,7))
            plt.subplot(3,1,1)
            plt.plot(inputs[:plot_length,1]) # angle
            plt.plot(inputs[:plot_length,2]) # error(同じ数値が他の日にある)
            plt.plot(pred_nan[:plot_length], linestyle="dashed", alpha=0.5)
            plt.grid()

            plt.subplot(3,1,2)
            plt.plot(preds.flatten()[:plot_length])
            plt.plot(targets.flatten()[:plot_length])
            plt.grid()

            plt.subplot(3,1,3)
            plt.plot(preds_switch.flatten()[:plot_length])
            # targetは破線で描画する
            plt.plot(0.2 * targets.flatten()[:plot_length], linestyle="dashed", alpha=0.5)
            plt.grid()
            plt.savefig(os.path.dirname(load_path) + "/plot_1/" + os.path.basename(f["file"]).split(".")[0] + ".png")
            # plt.show()
            # close
            plt.close()

            



    else:
        
        if not os.path.exists(os.path.dirname(load_path) + f"/pred{int(overlap_ratio*100):03}/"):
            os.makedirs(os.path.dirname(load_path) + f"/pred{int(overlap_ratio*100):03}/")
        for i, f in enumerate(files_val):
            stt = time.time()
            end_step = f["end_step_for_train"]
            features = np.load(f["file"])# [:end_step]
            steps = np.load(f["file_step"])# [:end_step]
            pred_nan = features[:, 2]
            nan_span = features[:, 3]
            nan_counter = features[:, -4]

            # inputs = features[:,:-1]
            inputs = features[:,:-2]
            daily_step = features[:, 0]
            targets = features[:,-1:]
            # print("load time", time.time()-stt)
            stt = time.time()

            preds, preds_switch = sa_model.overlap_predict(inputs, overlap=overlap_ratio)
            # plt.hist(preds.flatten(), bins=100)
            # plt.show()
            # preds_noom, preds_switch_noom = sa_model.overlap_predict_no_oom(inputs, overlap=overlap_ratio)

            # plt.hist((preds_noom - preds).flatten(), bins=100)
            # plt.show()

            # # predsのhistgram
            # plt.hist(preds.flatten(), bins=100)
            # plt.show()
            # raise Exception()


            print("pred time", time.time()-stt)
            stt = time.time()

            # save prediction
            # csvで保存する場合
            # save_path = os.path.dirname(load_path) + "/pred/" + os.path.basename(f["file"]).replace(".npy", ".csv")
            # df = pd.DataFrame({"step": steps, "pred_awake": preds.flatten(), "pred_switch": preds_switch.flatten(), "pred_nan": inputs[:,2]})
            # df.to_csv(save_path, index=False)

            # parquetで保存。csvよりも軽いはやい
            save_path = os.path.dirname(load_path) + f"/pred{int(overlap_ratio*100):03}/" + os.path.basename(f["file"]).replace(".npy", ".parquet")
            # df = pd.DataFrame({"step": steps, "daily_step": daily_step, "pred_awake": preds.flatten(), "pred_switch": preds_switch.flatten(), "pred_nan": pred_nan})
            if preds_switch.shape[-1] > 1:
                df = pd.DataFrame({"step": steps[:len(preds_switch)*STRIDE][::STRIDE],
                                "daily_step": daily_step[:len(preds_switch)*STRIDE][::STRIDE], 
                                "pred_awake": preds.flatten().astype(np.float32), 
                                "pred_switch10p": preds_switch[:,0].astype(np.float32), 
                                "pred_switch10": preds_switch[:,1].astype(np.float32), 
                                "pred_switch8": preds_switch[:,2].astype(np.float32),
                                "pred_switch6": preds_switch[:,3].astype(np.float32),
                                "pred_switch4": preds_switch[:,4].astype(np.float32),
                                "pred_switch2": preds_switch[:,5].astype(np.float32),
                                "pred_nan": pred_nan[:len(preds_switch)*STRIDE][::STRIDE].astype(np.float32),
                                "pred_nan_span": nan_span[:len(preds_switch)*STRIDE][::STRIDE].astype(np.float32),
                                "pred_nan_counter": nan_counter[:len(preds_switch)*STRIDE][::STRIDE].astype(np.float32),
                                })
            else:
                df = pd.DataFrame({"step": steps[:len(preds_switch)*STRIDE][::STRIDE],
                                "daily_step": daily_step[:len(preds_switch)*STRIDE][::STRIDE], 
                                "pred_awake": preds.flatten().astype(np.float32), 
                                "pred_switch": preds_switch[:,0].astype(np.float32), 
                                "pred_nan": pred_nan[:len(preds_switch)*STRIDE][::STRIDE].astype(np.float32),
                                })
            df.to_parquet(save_path, index=False)
            # print("save time", time.time()-stt)

            view = False
            if view:
                # 描画サイズを大きく
                plt.figure(figsize=(15,7))
                plt.subplot(3,1,1)
                plt.plot(inputs[20000:40000,1])
                plt.grid()

                plt.subplot(3,1,2)
                plt.plot(preds.flatten()[20000:40000])
                plt.plot(targets.flatten()[20000:40000])
                plt.grid()

                plt.subplot(3,1,3)
                plt.plot(preds_switch.flatten()[20000:40000])
                # targetは破線で描画する
                plt.plot(0.2 * targets.flatten()[20000:40000], linestyle="dashed", alpha=0.5)
                plt.grid()
                # plt.savefig(os.path.dirname(load_path) + "/plot_1/" + os.path.basename(f["file"]).split(".")[0] + ".png")
                plt.show()
                # plt.close()



if __name__ == "__main__":
    Cfg = json.load(open("SETTINGS.json"))


    if Cfg["IS_DEBUG"]=="True":
        print("--DEBUG MODE -> run only 2 epochs, 2fold split --")
        folds = [0,1]
        num_fold = 2
    else:
        folds = [0,1,2,3,4]
        num_fold = 5


    train_all = False
    D = True
    SSLmodel = False
    STRIDE = 1
    USECUTMIX = True
    Ws=[1, 4, 24, 24*4, 24*8, 24*16] #, 12, 24*32]
    for run_no, MODELTYPE in enumerate(["contstride", "normal"]):
        for SEED in [42, 111]:
            description = f"SplitStem{num_fold}foldSEED{SEED}controledStride" if "contstride" in MODELTYPE else f"SplitStem{num_fold}foldSEED{SEED}normal"
            length = 2880 * 5 if "contstride" in MODELTYPE else 1024 * 14
            for fold_no in folds:
                run_training_main(epochs = 2 if Cfg["IS_DEBUG"]=="True" else 35,
                                    batch_size=int(2) if Cfg["IS_DEBUG"]=="True" else int(96),
                                    input_shape=(length, 10), 
                                    output_shape=(length, 1),
                                    learning_rate=0.0012,
                                    data_dir=Cfg["preprocess_dir"],
                                    save_path=os.path.join(Cfg["weight_dir_1dcnn"], f"exp00_run_{run_no:02d}_{description}_fold{fold_no}/"),
                                    data_dir_gen=None,
                                    gen_pretrain=False, 
                                    train_fold_no=fold_no,
                                    load_path = None,
                                    num_fold=num_fold,
                                    train_all=train_all) 


                run_prediction(input_shape=(length, 10), 
                                    output_shape=(length, 1),
                                    data_dir=Cfg["preprocess_dir"], 
                                    load_path=os.path.join(Cfg["weight_dir_1dcnn"], f"exp00_run_{run_no:02d}_{description}_fold{fold_no}/final_weights.h5"),
                                    plot_length = None,
                                    train_fold_no=fold_no,
                                    num_fold=num_fold,
                                    train_all=train_all,
                                    overlap_ratio=0.90)
                continue
