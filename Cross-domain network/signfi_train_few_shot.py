import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import StepLR
import datasets
import serialization
from config import (
    logprint,
    logger,
    FEATURE_DIM,
    device,
    LR,
    EPISODE,
    TEST_T,
    TEST_EPISODE,
    BATCH_NUM_PER_CLASS,
    SAMPLE_NUM_PER_CLASS,
    CLASS_NUM,
    FEATURE_H,
    FEATURE_W,
    Net_factor,
    TRAIN_DATA,
    TEST_DATA,
    #VAL_DATA,
)
from models import CNNEncoder, RelationNetwork

# from apex import amp, optimizers
from test import test_net


def main():

    logprint("init data")
    train_dataset = datasets.Datasets(TRAIN_DATA)
    test_dataset = datasets.Datasets(TEST_DATA)
    #val_dataset = datasets.Datasets(VAL_DATA)
    if not os.path.exists("weights/User_ours"):
        os.makedirs("weights/User_ours")
    weight_name = Path(time.strftime("%m_%d_%H_%M", time.localtime()))
    weight_path = "weights/User_ours" / weight_name
    os.makedirs(weight_path)
    logprint("init neural networks")

    feature_encoder = CNNEncoder(Net_factor)
    relation_network = RelationNetwork(Net_factor)

    feature_encoder.to(device)
    relation_network.to(device)

    feature_encoder_optim = torch.optim.AdamW(feature_encoder.parameters(), lr=LR)
    feature_encoder_post_scheduler = lr_scheduler.CosineAnnealingLR(
        feature_encoder_optim, T_max=20000, eta_min=5e-5
    )
    feature_encoder_scheduler = GradualWarmupScheduler(
        feature_encoder_optim,
        multiplier=3,
        total_epoch=100,
        after_scheduler=feature_encoder_post_scheduler,
    )
    relation_network_optim = torch.optim.Adam(relation_network.parameters(), lr=LR)
    # relation_network_optim = optim.AdaBound(relation_network.parameters(), lr=LR)
    relation_network_post_scheduler = lr_scheduler.CosineAnnealingLR(
        relation_network_optim, T_max=20000, eta_min=5e-5
    )
    relation_network_scheduler = GradualWarmupScheduler(
        relation_network_optim,
        multiplier=3,
        total_epoch=100,
        after_scheduler=relation_network_post_scheduler,
    )
    loss_f = nn.MSELoss().to(device)
    feature_encoder.zero_grad()
    relation_network.zero_grad()
    feature_encoder_optim.step()
    relation_network_optim.step()

    # Step 3: build graph
    logprint("Training...")
    max_acc = 0.0
    max_train_acc = 0.0
    for episode in range(EPISODE):

        # init dataset
        # sample_dataloader is to obtain previous samples for compare
        # batch_dataloader is to batch samples for training

        # sample datas
        samples, sample_labels, batches, batch_labels = train_dataset.get_train_datas(
            CLASS_NUM, SAMPLE_NUM_PER_CLASS, BATCH_NUM_PER_CLASS, (0, 10)
        )
        
        # calculate features
        
        sample_features = feature_encoder(samples.to(device))
        sample_features = sample_features.view(
            CLASS_NUM, SAMPLE_NUM_PER_CLASS, FEATURE_DIM, FEATURE_H, FEATURE_W
        )
        sample_features = torch.sum(sample_features, 1).squeeze(1)

        batch_features = feature_encoder(batches.to(device))

        # calculate relations
        # each batch sample link to every samples to calculate relations
        # to form a 100x128 matrix for relation network
        sample_features_ext = sample_features.unsqueeze(0).repeat(
            BATCH_NUM_PER_CLASS * CLASS_NUM, 1, 1, 1, 1
        )
        batch_features_ext = batch_features.unsqueeze(0).repeat(CLASS_NUM, 1, 1, 1, 1)
        batch_features_ext = torch.transpose(batch_features_ext, 0, 1)

        relation_pairs = torch.cat((sample_features_ext, batch_features_ext), 2).view(
            -1, FEATURE_DIM * 2, FEATURE_H, FEATURE_W
        )
        relations = relation_network(relation_pairs).view(-1, CLASS_NUM)

        one_hot_labels = (
            torch.zeros(BATCH_NUM_PER_CLASS * CLASS_NUM, CLASS_NUM)
            .scatter_(1, batch_labels.view(-1, 1).long(), 1)
            .to(device)
        )

        loss = loss_f(relations, one_hot_labels)

        # training

        feature_encoder.zero_grad()
        relation_network.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_value_(feature_encoder.parameters(), 0.5)
        torch.nn.utils.clip_grad_value_(relation_network.parameters(), 0.5)

        feature_encoder_optim.step()
        relation_network_optim.step()

        feature_encoder_scheduler.step()
        relation_network_scheduler.step()

        if (episode + 1) % 20 == 0:
            logprint(
                f"episode:{episode + 1} loss {loss.item():.5f} lr {feature_encoder_optim.param_groups[0]['lr']:.5f}"
            )
        #
        if (episode + 1) % TEST_T == 0:

            # test
            logprint("Testing...")
            feature_encoder.eval()
            relation_network.eval()
            with torch.no_grad():
                test_accuracy, confidence_interval = test_net(
                    feature_encoder,
                    relation_network,
                    TEST_EPISODE,
                    test_dataset,
                    class_range=(100, 200),
                    train=False,
                )
                test_train_accuracy, confidence_interval_train = test_net(
                    feature_encoder,
                    relation_network,
                    TEST_EPISODE,
                    train_dataset,
                    class_range=(0, 10),
                    train=True,
                )
              
            logprint(
                f"test train accuracy: {test_train_accuracy:.4f} h:{confidence_interval_train} best:{max_train_acc:.4f}"
            )
            logprint(
                f"test accuracy: {test_accuracy:.4f} h:{confidence_interval} best:{max_acc:.4f}"
            )
            
            if test_train_accuracy > max_train_acc:
                max_train_acc = test_train_accuracy
            if test_accuracy > max_acc:
                
                max_acc = test_accuracy
                time_str = time.strftime("%m_%d_%H_%M_%S", time.localtime())

                log_str = f"T!{time_str}_D!{train_dataset.get_datas_name()}_{CLASS_NUM}W{SAMPLE_NUM_PER_CLASS}S_E!{episode}_A!{max_acc:.4f}"
                serialization.save_net(
                    feature_encoder, weight_path / f"feature_encoder_{log_str}.pkl",
                )
                serialization.save_net(
                    relation_network, weight_path / f"relation_network_{log_str}.pkl",
                )

            feature_encoder.train()
            relation_network.train()


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        logger.error(error)
        raise error
