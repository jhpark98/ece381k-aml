import glob
import json
import os

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

AUTO = tf.data.experimental.AUTOTUNE

def load_dataset(path, only_feature=False):
    """
    pathにあるnpyファイル名を読み込み、ファイル名とデータ長のdictのリストを返す。
    """
    files = sorted(glob.glob(os.path.join(path, "*feature.npy")))
    files_step = sorted(glob.glob(os.path.join(path, "*step.npy")))
    files_meta = sorted(glob.glob(os.path.join(path, "*meta.json")))
    dataset = []
    for i, (file, file_step, file_meta) in enumerate(zip(files, files_step, files_meta)):
        data = np.load(file)
        meta = json.load(open(file_meta))
        #for i in range(5):
        #    plt.plot(data[:7200, i])
        #    plt.show()
        #raise Exception
        
        #if only_feature:
        #    dataset.append({"file": file, "length": data.shape[0], "channel": data.shape[1]})
        #else:
        
        dataset.append({"file": file, "file_step": file_step, "length": data.shape[0], "channel": data.shape[1], "end_step_for_train": min(int(meta["end_step_for_train"]), data.shape[0]), "id_no": i}) #id_for classificaiton
    return dataset

def decode_npy(file, length, num_ch):
    header_offset = 128 # npy_header_offset(filename)
    dtype = tf.float32
    def npy_header_offset(npy_path):
        with open(str(npy_path), 'rb') as f:
            if f.read(6) != b'\x93NUMPY':
                raise ValueError('Invalid NPY file.')
            version_major, version_minor = f.read(2)
            if version_major == 1:
                header_len_size = 2
            elif version_major == 2:
                header_len_size = 4
            else:
                raise ValueError('Unknown NPY file version {}.{}.'.format(version_major, version_minor))
            header_len = sum(b << (8 * i) for i, b in enumerate(f.read(header_len_size)))
            header = f.read(header_len)
            if not header.endswith(b'\n'):
                raise ValueError('Invalid NPY file.')
            return f.tell()
    size = length * num_ch
    raw = tf.io.read_file(file)
    raw = tf.strings.substr(raw, pos=header_offset, len=size * dtype.size)
    output = tf.io.decode_raw(raw, dtype)#, fixed_length= size * dtype.size)
    return output

def read_npy_tf_function(file, length, num_ch):
    """
    tf py functionでnp.loadを実行する
    """
    def npy_load(file, length, num_ch):
        data = np.load(file.numpy().decode())
        return data # [:length, :num_ch]
    data = tf.py_function(npy_load, [file, length, num_ch], tf.float32)
    return data

def decode_files(dataset, is_train=False):
    file = dataset["file"]
    length = dataset["length"]
    channel = dataset["channel"]
    end_step_for_train = dataset["end_step_for_train"]
    npy_data = read_npy_tf_function(file, length, channel)
    dataset["data_array"] = tf.reshape(npy_data, [length, channel])
    # if is_train:
    #     dataset["data_array"] = dataset["data_array"][:end_step_for_train, :] # 悪化したので不使用中
    #     dataset["length"] = end_step_for_train

    return dataset

def build_tf_dataset(original_dataset, is_train=False):
    def gen_wrapper(dataset, data_keys=None):
        def generator():
            for data in dataset:
                yield data
        return generator
    output_types = {"file": tf.string,
                    "file_step": tf.string,
                    "length": tf.int32,
                    "channel": tf.int32,
                    "end_step_for_train": tf.int32,
                    "id_no": tf.int32,
                    }
    dataset = tf.data.Dataset.from_generator(gen_wrapper(original_dataset), output_types=output_types)
    if is_train:
        dataset = dataset.shuffle(64)
    dataset = dataset.map(lambda x: decode_files(x, is_train=is_train), num_parallel_calls=AUTO)
    return dataset

def change_resolution(data_array, scale=1.0):
    """
    data_array: [data_length, num_ch]
    """
    data_length, ch = tf.unstack(tf.shape(data_array))
    target_resolution = tf.cast(tf.cast(data_length, tf.float32) * scale, tf.int32)
    data_array = data_array[tf.newaxis, :, tf.newaxis, :] # tf.reshape(data_array, [1, data_length, 1, ch])
    data_array = tf.image.resize(data_array, [target_resolution, 1])
    data_array = tf.reshape(data_array, [target_resolution, -1])
    return data_array, target_resolution


def shift_data(data_array, max_shift=720):
    # shift = tf.random.uniform(shape=(), minval=-max_shift, maxval=max_shift, dtype=tf.int32)
    # 15minの倍数にしたいので、15*12=180で割って余りを出す
    # shift = tf.random.uniform(shape=(), minval=-max_shift//180, maxval=max_shift//180, dtype=tf.int32) * 180 -> めっちゃオーバーフィットする
    # 1minの倍数の場合
    # shift = tf.random.uniform(shape=(), minval=-max_shift//12, maxval=max_shift//12, dtype=tf.int32) * 12
    # 30s
    # shift = tf.random.uniform(shape=(), minval=-max_shift//6, maxval=max_shift//6, dtype=tf.int32) * 6
    shift = tf.random.uniform(shape=(), minval=-max_shift, maxval=max_shift, dtype=tf.int32)

    # if tf.random.uniform(shape=(), minval=0, maxval=1) < 0.35:
    #     shift = (shift//6) * 6
    #     if tf.random.uniform(shape=(), minval=0, maxval=1) < 0.5:
    #         shift = (shift // 12) * 12


    data_array = tf.roll(data_array, shift=shift, axis=0)# 最初と最後がおかしくなるけど、どうするか？
    return data_array

def shift_data_not_include_edge(data_array, max_shift=720):
    shift = tf.random.uniform(shape=(), minval=1, maxval=max_shift, dtype=tf.int32)
    if tf.random.uniform(shape=(), minval=0, maxval=1) < 0.5:
        data_array = data_array[shift:]
    else:
        data_array = data_array[:-shift]
    return data_array, shift

def patch_data(data_array, data_length, patch_size=720):
    """
    [data_length, num_ch] -> [num_split, patch_size, num_ch]
    """
    num_split = data_length // patch_size
    data_array = tf.reshape(data_array[:num_split*patch_size, :], [num_split, patch_size, -1])
    return data_array, num_split

def add_target_as_feature(data):
    target = data["data_array"][..., -1:]
    # target = data["data_array"][..., :-1]
    # TODO
    # 17280ずらすと一日ずれる。逆方向にずらしたやつもみてみたい。というかそれでマージしてやらないと、端っこおかしくなる。
    # 理想はシャッフルとか、前後日とかがいいけど、どうかな。あと入力はtargetではなく特徴量
    target_1 = tf.roll(target, shift=17280, axis=0)
    target_2 = tf.roll(target, shift=-17280, axis=0)
    data["data_array"] = tf.concat([data["data_array"][...,:-1], target_1, target_2, data["data_array"][...,-1:]], axis=-1)
    return data

def add_other_day_as_feature(data, num_days=3):
    feat = data["data_array"]
    # target = data["data_array"][..., :-1]
    # TODO
    # 17280ずらすと一日ずれる。逆方向にずらしたやつもみてみたい。というかそれでマージしてやらないと、端っこおかしくなる。
    # 理想はシャッフルとか、前後日とかがいいけど、どうかな。あと入力はtargetではなく特徴量
    feat_1 = tf.roll(feat, shift=17280, axis=0)# [..., :-2]
    feat_2 = tf.roll(feat, shift=-17280, axis=0)# [..., :-2]
    data["data_array"] = tf.concat([data["data_array"], feat_1, feat_2], axis=-1)
    return data

def split_data_length(data, split_length, is_train=False):
    data_array = data["data_array"]
    data_length = data["length"]
    # data_length = data["end_step_for_train"]

    # if is_train:
    #     if tf.random.uniform(shape=(), minval=0, maxval=1) < 0.5: # resize augmentation
    #         resolution_ratio = tf.random.uniform(shape=(), minval=0.8, maxval=1.2)
    #         data_array, target_resolution = change_resolution(data_array, resolution_ratio)
    #         data_length = target_resolution
    if is_train:
        data_array = shift_data(data_array, split_length//2)
        # data_array, shift = shift_data_not_include_edge(data_array, split_length//2)
        # data_length = data_length - shift
    data_array, num_split = patch_data(data_array, data_length, split_length)
    id_no_tiled = tf.tile(tf.expand_dims(data["id_no"], axis=0), [num_split])
    data = {"data_array" : data_array, "id_no": id_no_tiled}
    return data

def unstack_day_features(data, num_days=3):
    data_array = data["data_array"]
    data_length = data["length"]
    data_array = tf.split(data_array, num_days, axis=-1)
    data_array = tf.stack(data_array, axis=0)
    data = {"data_array" : data_array, "length": data_length}
    return data

def augmentations(data):
    anglez = data["data_array"][..., 1:2]
    if tf.random.uniform(shape=(), minval=0, maxval=1) > 0.5:
        anglez = anglez * -1
    # if tf.random.uniform(shape=(), minval=0, maxval=1) > 0.1:
    #     anglez = anglez + tf.random.uniform(shape=(), minval=-0.1, maxval=0.1)
    #     anglez = anglez * tf.random.uniform(shape=(), minval=0.8, maxval=1.2)

    
    data["data_array"] = tf.concat([data["data_array"][..., :1], anglez, data["data_array"][..., 2:]], axis=-1)
    # if tf.random.uniform(shape=(), minval=0, maxval=1) > 0.5:
    #     data["data_array"] = data["data_array"][::-1]
    return data

def assign_inputs_targets_multiclass(data, feat_ch=10, outstride=1, shift_target=False):
    # data["data_array"][:, :, -2:-1]は11クラス分類の評価指標ベースのピラミッド。onehotに変換する
    # switch_onehot = tf.one_hot(tf.cast(data["data_array"][:, :, -2], tf.int32), num_cls)

    # switch_onehot = tf.stack([tf.cast(data["data_array"][:, :, -2]>(0.5+i), tf.float32) for i in range(10)], axis=-1)# 一時的

    inputs = {"inputs": data["data_array"][:, :, :(feat_ch)]}
    # inputs["inputs"] = tf.concat(tf.split(inputs["inputs"], 3, axis=-1), axis=0)
    manyout=True
    if not manyout:
        targets = {"out_state": data["data_array"][:, :, (feat_ch+1):(feat_ch+2)],
                "out_state_nan": data["data_array"][:, :, 2:3] -1 + data["data_array"][:, :, 4:5], # is nan. nan existしない部分は無効
                "out_event_s10": tf.cast(data["data_array"][:, :, feat_ch:(feat_ch+1)]>9.5, tf.float32),
                "out_event_s8": tf.cast(data["data_array"][:, :, feat_ch:(feat_ch+1)]>7.5, tf.float32),
                # "out_event_s6": tf.cast(data["data_array"][:, :, -2:-1]>5.5, tf.float32),
                    }
    else:
        sleepawake = data["data_array"][:, :, (feat_ch+1):(feat_ch+2)]
        sleepawake_weight = tf.cast(data["data_array"][:, :, feat_ch:(feat_ch+1)]>5.5, tf.float32) + tf.cast(data["data_array"][:, :, feat_ch:(feat_ch+1)]>3.5, tf.float32)  + tf.cast(data["data_array"][:, :, feat_ch:(feat_ch+1)]>1.5, tf.float32) + 1.
        if shift_target:
            targets = {"out_state": sleepawake, # tf.concat([sleepawake, sleepawake_weight], axis=-1),
                "out_state_nan": data["data_array"][:, :, 2:3] -1 + data["data_array"][:, :, 4:5], # is nan. nan existしない部分は無効
                "out_event_s10p": tf.cast(data["data_array"][:, :, feat_ch:(feat_ch+1)]>10.5, tf.float32),
                "out_event_s10": tf.cast(data["data_array"][:, :, feat_ch:(feat_ch+1)]>9.5, tf.float32),
                "out_event_s8": tf.cast(data["data_array"][:, :, feat_ch:(feat_ch+1)]>8.5, tf.float32),
                "out_event_s6": tf.cast(data["data_array"][:, :, feat_ch:(feat_ch+1)]>7.5, tf.float32),
                "out_event_s4": tf.cast(data["data_array"][:, :, feat_ch:(feat_ch+1)]>6.5, tf.float32),
                "out_event_s2": tf.cast(data["data_array"][:, :, feat_ch:(feat_ch+1)]>5.5, tf.float32),
                }
        else:
            targets = {"out_state": sleepawake, # tf.concat([sleepawake, sleepawake_weight], axis=-1),
                "out_state_nan": data["data_array"][:, :, 2:3] -1 + data["data_array"][:, :, 4:5], # is nan. nan existしない部分は無効
                "out_event_s10p": tf.cast(data["data_array"][:, :, feat_ch:(feat_ch+1)]>10.5, tf.float32),
                "out_event_s10": tf.cast(data["data_array"][:, :, feat_ch:(feat_ch+1)]>9.5, tf.float32),
                "out_event_s8": tf.cast(data["data_array"][:, :, feat_ch:(feat_ch+1)]>7.5, tf.float32),
                "out_event_s6": tf.cast(data["data_array"][:, :, feat_ch:(feat_ch+1)]>5.5, tf.float32),
                "out_event_s4": tf.cast(data["data_array"][:, :, feat_ch:(feat_ch+1)]>3.5, tf.float32),
                "out_event_s2": tf.cast(data["data_array"][:, :, feat_ch:(feat_ch+1)]>1.5, tf.float32),
                }
    if outstride != 1:
        targets["out_event_s10p"] = targets["out_event_s10p"][:, ::outstride, :]
        targets["out_event_s10"] = targets["out_event_s10"][:, ::outstride, :]
        targets["out_event_s8"] = targets["out_event_s8"][:, ::outstride, :]
        targets["out_event_s6"] = targets["out_event_s6"][:, ::outstride, :]
        targets["out_event_s4"] = targets["out_event_s4"][:, ::outstride, :]
        targets["out_event_s2"] = targets["out_event_s2"][:, ::outstride, :]
        targets["out_state"] = targets["out_state"][:, ::outstride, :]
        targets["out_state_nan"] = targets["out_state_nan"][:, ::outstride, :]
                
    # inputs = {"inputs": tf.concat([data["data_array"][:, :, :(feat_ch)], data["data_array"][:, :, (feat_ch+2):(feat_ch*2+2)], data["data_array"][:, :, (2*feat_ch+4):(feat_ch*3+4)]], axis=0)}
    # targets = {"out_state": tf.concat([data["data_array"][:, :, (feat_ch+1):(feat_ch+2)], data["data_array"][:, :, (2*feat_ch+3):(2*feat_ch+4)], data["data_array"][:, :, (3*feat_ch+5):(3*feat_ch+6)]], axis=0),
    #            "out_state_nan": tf.concat([data["data_array"][:, :, 2:3] -1 + data["data_array"][:, :, 4:5], data["data_array"][:, :, (feat_ch+4):(feat_ch+5)] -1 + data["data_array"][:, :, (feat_ch+6):(feat_ch+7)], data["data_array"][:, :, (2*feat_ch+6):(2*feat_ch+7)] -1 + data["data_array"][:, :, (2*feat_ch+8):(2*feat_ch+9)]], axis=0), # is nan. nan existしない部分は無効
    #            "out_event_s10": tf.concat([tf.cast(data["data_array"][:, :, feat_ch:(feat_ch+1)]>9.5, tf.float32), tf.cast(data["data_array"][:, :, (2*feat_ch+2):(2*feat_ch+3)]>9.5, tf.float32), tf.cast(data["data_array"][:, :, (3*feat_ch+4):(3*feat_ch+5)]>9.5, tf.float32)], axis=0),
    #            "out_event_s8": tf.concat([tf.cast(data["data_array"][:, :, feat_ch:(feat_ch+1)]>7.5, tf.float32), tf.cast(data["data_array"][:, :, (2*feat_ch+2):(2*feat_ch+3)]>7.5, tf.float32), tf.cast(data["data_array"][:, :, (3*feat_ch+4):(3*feat_ch+5)]>7.5, tf.float32)], axis=0),
    #            # "out_event_s6": tf.cast(data["data_array"][:, :, -2:-1]>5.5, tf.float32),
    #             }


    return inputs, targets

def cutout_augmentation(inputs, targets, p=0.5, min_duration=0.01, max_duration=0.05, fill_value=0.0):
    array = inputs["inputs"]
    batch, data_length, ch = tf.unstack(tf.shape(array))
    data_length = tf.cast(data_length, tf.float32)
    if tf.random.uniform(shape=(), minval=0, maxval=1) < p:
        duration = tf.random.uniform(shape=(), minval=min_duration, maxval=max_duration)
        start = tf.random.uniform(shape=(), minval=0, maxval=1-duration)
        start = tf.cast(start * data_length, tf.int32)
        duration = tf.cast(duration * data_length, tf.int32)
        array = tf.concat([array[:, :start, :], fill_value * tf.ones((batch, duration, ch), tf.float32), array[:, (start+duration):, :]], axis=1)
    inputs["inputs"] = array
    return inputs, targets


def assign_inputs_targets_ssl(data, feat_ch=10):
    inputs = {"inputs": data["data_array"][:, :, :(feat_ch)]}
    targets = {"embedding_outputs": tf.ones((10,10)),} # dummy
    return inputs, targets



def cutmix_batch(inputs, targets, p=0.5, outstride=1):
    inputs_copy = inputs
    targets_copy = targets

    batch, data_length = tf.unstack(tf.shape(inputs["inputs"]))[:2]
    cut_indices = tf.random.uniform(shape=(2,), minval=0, maxval=data_length, dtype=tf.int32)
    

    if tf.random.uniform(shape=(), minval=0, maxval=1) < p:
        cut_index = tf.reduce_min(cut_indices)
        shuffle_indices = tf.random.shuffle(tf.range(batch))

        cut_index_tar = cut_index // outstride

        cond = targets["out_state"][:, cut_index_tar:(cut_index_tar+1), 0:1] == tf.gather(targets["out_state"][:, cut_index_tar:(cut_index_tar+1), 0:1], shuffle_indices)
        cond = tf.cast(cond, tf.float32)

        base_inp = inputs["inputs"]
        shuffled_inp = tf.gather(inputs_copy["inputs"], shuffle_indices)
        combined = tf.concat([base_inp[:, :cut_index, :], shuffled_inp[:, cut_index:, :]], axis=1)
        mixed = combined * cond + base_inp * (1-cond)
        inputs["inputs"] = mixed

        #for col in ["out_state","out_state_nan","out_event_s10","out_event_s8"]:
        base = targets["out_state"]
        shuffled = tf.gather(targets_copy["out_state"], shuffle_indices)
        combined = tf.concat([base[:, :cut_index_tar, :], shuffled[:, cut_index_tar:, :]], axis=1)
        mixed = combined * cond + base * (1-cond)
        targets["out_state"] = mixed

        base = targets["out_state_nan"]
        shuffled = tf.gather(targets_copy["out_state_nan"], shuffle_indices)
        combined = tf.concat([base[:, :cut_index_tar, :], shuffled[:, cut_index_tar:, :]], axis=1)
        mixed = combined * cond + base * (1-cond)
        targets["out_state_nan"] = mixed

        base = targets["out_event_s10p"]
        shuffled = tf.gather(targets_copy["out_event_s10p"], shuffle_indices)
        combined = tf.concat([base[:, :cut_index_tar, :], shuffled[:, cut_index_tar:, :]], axis=1)
        mixed = combined * cond + base * (1-cond)
        targets["out_event_s10p"] = mixed

        base = targets["out_event_s10"]
        shuffled = tf.gather(targets_copy["out_event_s10"], shuffle_indices)
        combined = tf.concat([base[:, :cut_index_tar, :], shuffled[:, cut_index_tar:, :]], axis=1)
        mixed = combined * cond + base * (1-cond)
        targets["out_event_s10"] = mixed

        base = targets["out_event_s8"]
        shuffled = tf.gather(targets_copy["out_event_s8"], shuffle_indices)
        combined = tf.concat([base[:, :cut_index_tar, :], shuffled[:, cut_index_tar:, :]], axis=1)
        mixed = combined * cond + base * (1-cond)
        targets["out_event_s8"] = mixed

        base = targets["out_event_s6"]
        shuffled = tf.gather(targets_copy["out_event_s6"], shuffle_indices)
        combined = tf.concat([base[:, :cut_index_tar, :], shuffled[:, cut_index_tar:, :]], axis=1)
        mixed = combined * cond + base * (1-cond)
        targets["out_event_s6"] = mixed

        base = targets["out_event_s4"]
        shuffled = tf.gather(targets_copy["out_event_s4"], shuffle_indices)
        combined = tf.concat([base[:, :cut_index_tar, :], shuffled[:, cut_index_tar:, :]], axis=1)
        mixed = combined * cond + base * (1-cond)
        targets["out_event_s4"] = mixed

        base = targets["out_event_s2"]
        shuffled = tf.gather(targets_copy["out_event_s2"], shuffle_indices)
        combined = tf.concat([base[:, :cut_index_tar, :], shuffled[:, cut_index_tar:, :]], axis=1)
        mixed = combined * cond + base * (1-cond)
        targets["out_event_s2"] = mixed



    if tf.random.uniform(shape=(), minval=0, maxval=1) < p:
        cut_index = tf.reduce_max(cut_indices)
        shuffle_indices = tf.random.shuffle(tf.range(batch))

        cut_index_tar = cut_index // outstride


        cond = targets["out_state"][:, cut_index_tar:(cut_index_tar+1), 0:1] == tf.gather(targets["out_state"][:, cut_index_tar:(cut_index_tar+1), 0:1], shuffle_indices)
        cond = tf.cast(cond, tf.float32)

        base_inp = inputs["inputs"]
        shuffled_inp = tf.gather(inputs_copy["inputs"], shuffle_indices)
        combined = tf.concat([base_inp[:, :cut_index, :], shuffled_inp[:, cut_index:, :]], axis=1)
        mixed = combined * cond + base_inp * (1-cond)
        inputs["inputs"] = mixed

        #for col in ["out_state","out_state_nan","out_event_s10","out_event_s8"]:
        base = targets["out_state"]
        shuffled = tf.gather(targets_copy["out_state"], shuffle_indices)
        combined = tf.concat([base[:, :cut_index_tar, :], shuffled[:, cut_index_tar:, :]], axis=1)
        mixed = combined * cond + base * (1-cond)
        targets["out_state"] = mixed

        base = targets["out_state_nan"]
        shuffled = tf.gather(targets_copy["out_state_nan"], shuffle_indices)
        combined = tf.concat([base[:, :cut_index_tar, :], shuffled[:, cut_index_tar:, :]], axis=1)
        mixed = combined * cond + base * (1-cond)
        targets["out_state_nan"] = mixed

        base = targets["out_event_s10p"]
        shuffled = tf.gather(targets_copy["out_event_s10p"], shuffle_indices)
        combined = tf.concat([base[:, :cut_index_tar, :], shuffled[:, cut_index_tar:, :]], axis=1)
        mixed = combined * cond + base * (1-cond)
        mixed = combined

        base = targets["out_event_s10"]
        shuffled = tf.gather(targets_copy["out_event_s10"], shuffle_indices)
        combined = tf.concat([base[:, :cut_index_tar, :], shuffled[:, cut_index_tar:, :]], axis=1)
        mixed = combined * cond + base * (1-cond)
        targets["out_event_s10"] = mixed

        base = targets["out_event_s8"]
        shuffled = tf.gather(targets_copy["out_event_s8"], shuffle_indices)
        combined = tf.concat([base[:, :cut_index_tar, :], shuffled[:, cut_index_tar:, :]], axis=1)
        mixed = combined * cond + base * (1-cond)
        targets["out_event_s8"] = mixed

        base = targets["out_event_s6"]
        shuffled = tf.gather(targets_copy["out_event_s6"], shuffle_indices)
        combined = tf.concat([base[:, :cut_index_tar, :], shuffled[:, cut_index_tar:, :]], axis=1)
        mixed = combined

        base = targets["out_event_s4"]
        shuffled = tf.gather(targets_copy["out_event_s4"], shuffle_indices)
        combined = tf.concat([base[:, :cut_index_tar, :], shuffled[:, cut_index_tar:, :]], axis=1)
        mixed = combined

        base = targets["out_event_s2"]
        shuffled = tf.gather(targets_copy["out_event_s2"], shuffle_indices)
        combined = tf.concat([base[:, :cut_index_tar, :], shuffled[:, cut_index_tar:, :]], axis=1)
        mixed = combined

    return inputs, targets


def get_dataset(files, batch_size, input_shape, is_train=True, SSLmodel=False, outstride=1, use_cutmix=True, shift_target=False, debug_mode=False):
    num_days = 3
    dataset = build_tf_dataset(files, is_train=is_train)
    dataset = dataset.map(lambda x: split_data_length(x, input_shape[0], is_train), num_parallel_calls=AUTO)
    # return dataset
    dataset = dataset.unbatch()
    if is_train:
        dataset = dataset.map(augmentations, num_parallel_calls=AUTO)
        if debug_mode:
            dataset = dataset.shuffle(64)
        else:
            dataset = dataset.shuffle(2048)
    else:
        if debug_mode:
            dataset = dataset.shuffle(64)
        else:
            dataset = dataset.shuffle(256)
    if is_train:
        dataset = dataset.repeat()

    dataset = dataset.batch(batch_size)
    if SSLmodel:
        dataset = dataset.map(lambda x: assign_inputs_targets_ssl(x), num_parallel_calls=AUTO)

    else:
        dataset = dataset.map(lambda x: assign_inputs_targets_multiclass(x, feat_ch=input_shape[-1], outstride=outstride, shift_target=shift_target), num_parallel_calls=AUTO)
        if is_train and use_cutmix:
            dataset = dataset.map(lambda x, y: cutmix_batch(x, y, outstride=outstride, p=0.5), num_parallel_calls=AUTO)
        
    dataset = dataset.prefetch(AUTO)
    return dataset


if __name__ == "__main__":
    files = load_dataset("data/data_processed")[:10]
    print(files)
    dataset = get_dataset(files, 32, [7200, 4], True)
    for inputs, targets in dataset:
    #for inputs in dataset:
        print(inputs["inputs"].shape)
        print(targets["targets"].shape)
        # 各バッチ毎に描画
        for i in range(inputs["inputs"].shape[0]):
            for j in range(4):
                plt.plot(inputs["inputs"][i, :, j])
                plt.title("input_{}".format(j))
                plt.show()
                print(j, inputs["inputs"][i, :, j].numpy().mean(), inputs["inputs"][i, :, j].numpy().min())

            plt.plot(inputs["inputs"][i, :, 0])
            plt.plot(inputs["inputs"][i, :, 1])
            plt.plot(inputs["inputs"][i, :, 2])
            plt.plot(inputs["inputs"][i, :, 3])
            plt.title("input")
            plt.show()
            plt.plot(targets["targets"][i, :, 0])
            plt.title("target")
            plt.show()




        break




