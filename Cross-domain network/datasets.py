import numpy as np
import torch
import os, platform


def shuffle_union(*union):
    len_l = [len(i) for i in union]
    assert np.unique(len_l).size == 1  # x in union should have same length
    shuffle_idx = np.random.permutation(len_l[0])
    return [x[shuffle_idx] for x in union]


def base_sampler(
    datas, labels, classes_list, classes_num, meta_train_num, meta_test_num, inst_per_c
):
    random_true_label = np.random.choice(classes_list, classes_num, replace=False)
    random_fake_label = np.arange(0, classes_num)

    meta_train_data = []
    meta_test_data = []
    for i in random_true_label:
        # instance_shuffle_idx = np.random.permutation(inst_per_c)
        temp = datas[np.where(labels == i)]
        # temp[:] = temp[instance_shuffle_idx]
        meta_train_data.append(temp[:meta_train_num])
        meta_test_data.append(temp[meta_train_num : meta_train_num + meta_test_num])
    meta_train_data = np.concatenate(meta_train_data)
    meta_test_data = np.concatenate(meta_test_data)

    meta_train_idx = random_fake_label.repeat(meta_train_num)
    meta_test_idx = random_fake_label.repeat(meta_test_num)
    meta_test_data, meta_test_idx = shuffle_union(meta_test_data, meta_test_idx)
    return (
        torch.tensor(meta_train_data).float(),
        torch.tensor(meta_train_idx).long(),
        torch.tensor(meta_test_data).float(),
        torch.tensor(meta_test_idx).long(),
    )

testmean = 0.5


def pre_process_amp(x):
    return x / 200 - testmean
    # return x / 300


def load_signfi_datas(x, pre_process_func):
    return pre_process_func(
        np.transpose(
            np.load(os.path.join(signfi_root, x)).astype(np.float32), (3, 2, 1, 0)
        )
    )


def load_signfi_labels(x):
    return np.load(os.path.join(signfi_root, x)).astype(np.int64)


if platform.system() == "Linux":
    signfi_root = "/home/yk/signfi_matlab2numpy"
else:
    signfi_root = "G:\JetBrainProject\csi_signfi"

print("Loading Data")
signfi_150_12345_amp = load_signfi_datas("150_amp_datas.npy", pre_process_amp)
signfi_150_12345_labels = load_signfi_labels("150_labels.npy")

signfi_276_home_amp = load_signfi_datas("amp_dataset_home_276_csid_home.npy", pre_process_amp)
signfi_276_home_labels = load_signfi_labels("label_dataset_home_276_label_home.npy")

signfi_276_lab_amp = load_signfi_datas("amp_dataset_lab_276_dl_csid_lab.npy", pre_process_amp)
signfi_276_lab_labels = load_signfi_labels("label_dataset_lab_276_dl_csid_lab.npy")
print("Finish")

total_classes = np.arange(0, 276)
# np.random.shuffle(total_classes)

class Datasets_list:
    signfi_150_user1 = (
        signfi_150_12345_amp[0:1500],
        signfi_150_12345_labels[0:1500],
        "signfi_150_user1",
        150,
        10,
    )
    signfi_150_user234 = (
        signfi_150_12345_amp[1500:6000],
        signfi_150_12345_labels[1500:6000],
        "signfi_150_user234",
        150,
        30,
    )
    signfi_150_user2 = (
        signfi_150_12345_amp[1500:3000],
        signfi_150_12345_labels[1500:3000],
        "signfi_150_user2",
        150,
        10,
    )
    signfi_150_user134 = (
        signfi_150_12345_amp[np.r_[0:1500,3000:6000]],
        signfi_150_12345_labels[np.r_[0:1500,3000:6000]],
        "signfi_150_user134",
        150,
        30,
    )
    signfi_150_user3 = (
        signfi_150_12345_amp[3000:4500],
        signfi_150_12345_labels[3000:4500],
        "signfi_150_user3",
        150,
        10,
    )
    signfi_150_user124 = (
        signfi_150_12345_amp[np.r_[0:3000,4500:6000]],
        signfi_150_12345_labels[np.r_[0:3000,4500:6000]],
        "signfi_150_user124",
        150,
        30,
    )
    signfi_150_user4 = (
        signfi_150_12345_amp[4500:6000],
        signfi_150_12345_labels[4500:6000],
        "signfi_150_user4",
        150,
        10,
    )
    signfi_150_user123 = (
        signfi_150_12345_amp[0:4500],
        signfi_150_12345_labels[0:4500],
        "signfi_150_user123",
        150,
        30,
    )
    signfi_150_user5 = (
        signfi_150_12345_amp[6000:],
        signfi_150_12345_labels[6000:],
        "signfi_150_user5",
        150,
        10,
    )
    signfi_276_lab = (
        signfi_276_lab_amp,
        signfi_276_lab_labels,
        "signfi_276_lab",
        276,
        20,
    )
    signfi_276_home = (
        signfi_276_home_amp,
        signfi_276_home_labels,
        "signfi_276_home",
        276,
        10,
    )


class Datasets:
    def __init__(self, datasets_union):

        self.x, self.y, self.name, self.class_num, self.instance_num = datasets_union
        self.instance_num = 10  ####
        self.x_split_by_class = self.__gen_basic_classifier_datas_by_class()

    def get_train_datas(self, nway, kshot, batch_kshot, class_range=(0, 10)):
        return base_sampler(
            self.x,
            self.y,
            total_classes[np.arange(*class_range)],
            nway,
            kshot,
            batch_kshot,
            self.instance_num,
        )

    def get_test_datas(self, nway, kshot, batch_kshot, class_range=(100, 250)):
        return base_sampler(
            self.x,
            self.y,
            total_classes[np.arange(*class_range)],
            nway,
            kshot,
            batch_kshot,
            self.instance_num,
        )

    def get_basic_classifier_datas_by_classrange(
        self, class_range=(0, 276), trans=None
    ):
        class_l = total_classes[np.arange(*class_range)]
        _datas, _labels = self.x, self.y
        datas = []
        for i in class_l:
            temp = _datas[np.where(_labels == i)]
            datas.append(temp[: self.instance_num])
            #datas.append(temp)
        datas = np.asarray(datas)
        if trans:
            return trans(datas)
        return datas

    def get_basic_classifier_datas_all(self, trans=None):
        datas, labels = shuffle_union(self.x, self.y,)
        if trans:
            return trans(datas), trans(labels)
        return datas, labels

    def __gen_basic_classifier_datas_by_class(self, total_class=276):
        l = []
        for c in range(total_class):
            l.append(self.x[np.where(self.y == c)])
        l = np.asarray(l)
        return l
    def __gen_idx_split_by_class(self, total_class=276):
        l = []
        for c in range(total_class):
            l.append(np.argwhere(self.y == c).reshape(-1))
        l = np.asarray(l)
        return l
    def get_datas_name(self):
        return self.name


