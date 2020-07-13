import os
from pathlib import Path

import numpy as np
import scipy
import scipy.stats
import torch
import time
from config import (
    FEATURE_DIM,
    device,
    SAMPLE_NUM_PER_CLASS,
    BATCH_NUM_PER_CLASS,
    CLASS_NUM,
    FEATURE_H,
    FEATURE_W,
)


def find_T(s):
    return s.find("T!")


def return_T(s):
    T_idx = find_T(s)
    return s[T_idx + 2 : T_idx + 13]


def get_highest_weight_name(path="weights/User"):

    weights_root = Path(path)
    name_list = [x for x in weights_root.iterdir() if x.is_dir()]
    name_list.sort(reverse=True)
    last_time_weight_folder = name_list[0]
    s = sorted(
        [str(x) for x in last_time_weight_folder.iterdir()], key=lambda x: return_T(x),
    )[-1]
    T_idx = find_T(s)
    s = s[T_idx:]
    return (
        last_time_weight_folder / ("feature_encoder_" + s),
        last_time_weight_folder / ("relation_network_" + s),
    )


def test_net(
    feature_net,
    relation_net,
    test_episode,
    dataset,
    class_range=(0, 0),
    train=False,
):
    acc = []
    with torch.no_grad():
        for i in range(test_episode):
            # print(i)
            start=time.time()
            if train:
                (
                    sample_images,
                    sample_labels,
                    test_images,
                    test_labels,
                ) = dataset.get_train_datas(
                    CLASS_NUM, SAMPLE_NUM_PER_CLASS, BATCH_NUM_PER_CLASS, class_range
                )
               
            else:
                (
                    sample_images,
                    sample_labels,
                    test_images,
                    test_labels,
                ) = dataset.get_test_datas(
                    CLASS_NUM, SAMPLE_NUM_PER_CLASS, BATCH_NUM_PER_CLASS, class_range
                )
                
            test_labels = test_labels.to(device)
            # calculate features
            sample_features = feature_net(sample_images.to(device))  # 5x64
            sample_features = sample_features.view(
                CLASS_NUM, SAMPLE_NUM_PER_CLASS, FEATURE_DIM, FEATURE_H, FEATURE_W,
            )
            sample_features = torch.sum(sample_features, 1).squeeze(1)
            sample_features_ext = sample_features.unsqueeze(0).repeat(
                BATCH_NUM_PER_CLASS * CLASS_NUM, 1, 1, 1, 1
            )
            
            test_features = feature_net(test_images.to(device))  # 20x64
            test_features_ext = test_features.unsqueeze(0).repeat(CLASS_NUM, 1, 1, 1, 1)
            test_features_ext = torch.transpose(test_features_ext, 0, 1)

            relation_pairs = torch.cat(
                (sample_features_ext, test_features_ext), 2
            ).view(-1, FEATURE_DIM * 2, FEATURE_H, FEATURE_W)
            relations = relation_net(relation_pairs).view(-1, CLASS_NUM)

            _, predict_labels = relations.max(1)
            #print('run')
            end=time.time()
            dur=end-start
            #print(dur)
            
            correct = predict_labels.eq(test_labels).sum()

            acc.append(correct.float().item() / predict_labels.shape[0])
    test_accuracy = np.mean(acc)
    confidence_interval = scipy.stats.t.interval(
        0.95, len(acc) - 1, loc=test_accuracy, scale=scipy.stats.sem(acc)
    )
    return test_accuracy, confidence_interval


if __name__ == "__main__":
    ret = get_highest_weight_name()
