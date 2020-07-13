import logging
import datasets

Net_factor = 1 / 6
FEATURE_DIM = int(96 * Net_factor)
FEATURE_DIM_ORI = 32
CLASS_NUM = 5
SAMPLE_NUM_PER_CLASS = 1
BATCH_NUM_PER_CLASS =9
EPISODE = 40000
TEST_EPISODE = 80
TEST_T = 100
LR = 8e-4
TRAIN_DATA = datasets.Datasets_list.signfi_276_lab
TEST_DATA = datasets.Datasets_list.signfi_276_home
VAL_DATA = datasets.Datasets_list.signfi_150_user124
# device = "cpu"
device = "cuda"

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler("log")
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def logprint(x):
    print(x)
    logger.info(x)


FEATURE_H =8
FEATURE_W =50

