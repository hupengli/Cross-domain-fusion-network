from pathlib import Path
import datasets
import serialization
from config import (
    logprint,
    logger,
    device,
    Net_factor,
)
from models import CNNEncoder, RelationNetwork

from test import get_highest_weight_name, test_net


def main():
    dataset = datasets.Datasets(datasets.Datasets_list.signfi_150_user4)
    feature_encoder = CNNEncoder(Net_factor)
    relation_network = RelationNetwork(Net_factor)
    feature_encoder.to(device)
    relation_network.to(device)
    path=Path("weights/User_ours")
    relation_network_best_path=path/'06_15_12_09/relation_network_T!06_15_12_11_23_D!signfi_150_user123_5W1S_E!299_A!0.8830.pkl'
    feature_encoder_best_path=path/'06_15_12_09/feature_encoder_T!06_15_12_11_23_D!signfi_150_user123_5W1S_E!299_A!0.8830.pkl'

    serialization.load_net(feature_encoder, feature_encoder_best_path)
    serialization.load_net(
        relation_network, relation_network_best_path,
    )

    # test
    logprint("DIY Testing...")
    feature_encoder.eval()
    relation_network.eval()
    test_accuracy, confidence_interval = test_net(
        feature_encoder,
        relation_network,
        5,
        dataset,
        class_range=(0, 150),
        train=False,
    )
    logprint(f"test accuracy: {test_accuracy:.4f} h:{confidence_interval} ")


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        logger.error(error)
        raise error
