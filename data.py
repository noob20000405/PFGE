import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os

c10_classes = np.array([[0, 1, 2, 8, 9], [3, 4, 5, 6, 7]], dtype=np.int32)

def loaders(
    dataset,
    path,
    batch_size,
    num_workers,
    transform_train,
    transform_test,
    use_validation=True,
    val_size=5000,
    split_classes=None,
    shuffle_train=True,
    **kwargs
):
    """
    返回:
      - 当 use_validation=True: loaders = {'train','val','test'}
      - 否则:                  loaders = {'train','test'}
    其中:
      train 使用 transform_train；
      val/test 使用 transform_test。
    """
    path = os.path.join(path, dataset.lower())
    ds = getattr(torchvision.datasets, dataset)

    # 1) 原始 train/test 数据集
    train_set_full = ds(root=path, train=True, download=True, transform=transform_train)
    test_set = ds(root=path, train=False, download=True, transform=transform_test)

    # 基类数（可能在 split_classes 后更新）
    num_classes = int(np.max(train_set_full.targets)) + 1

    loaders_dict = {}

    if use_validation:
        assert val_size > 0 and val_size < len(train_set_full.data), \
            f"val_size must be in (0, {len(train_set_full.data)}), got {val_size}"

        print(
            "Using train ("
            + str(len(train_set_full.data) - val_size)
            + ") + validation ("
            + str(val_size)
            + ")"
        )

        # 2) 训练集（保留前 N - val_size）
        train_set = ds(root=path, train=True, download=False, transform=transform_train)
        train_set.data = train_set.data[:-val_size]
        train_set.targets = train_set.targets[:-val_size]

        # 3) 验证集（用无增广 transform_test；从尾部切 val_size）
        val_set = ds(root=path, train=True, download=False, transform=transform_test)
        val_set.train = False  # 仅作标识，避免歧义
        val_set.data = val_set.data[-val_size:]
        val_set.targets = val_set.targets[-val_size:]

        loaders_dict["train"] = torch.utils.data.DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True and shuffle_train,
            num_workers=num_workers,
            pin_memory=True,
        )
        loaders_dict["val"] = torch.utils.data.DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        loaders_dict["test"] = torch.utils.data.DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        # 后续 split_classes 需要同时作用到 train/val/test
        targets_sets = {"train": train_set, "val": val_set, "test": test_set}
    else:
        print("You are going to run models on the test set. Are you sure?")
        train_set = train_set_full  # 直接用全训练集（带增强）
        loaders_dict["train"] = torch.utils.data.DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True and shuffle_train,
            num_workers=num_workers,
            pin_memory=True,
        )
        loaders_dict["test"] = torch.utils.data.DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        targets_sets = {"train": train_set, "test": test_set}

    # 4) （可选）切分 CIFAR-10 的子类集合
    if split_classes is not None:
        assert dataset == "CIFAR10"
        assert split_classes in {0, 1}
        print("Using classes:", c10_classes[split_classes])

        def _apply_split(ds_obj):
            mask = np.isin(ds_obj.targets, c10_classes[split_classes])
            # CIFAR 的 targets 是 list，需要转 array 再还原 list
            ds_obj.data = ds_obj.data[mask, :]
            t = np.array(ds_obj.targets)[mask]
            # 重新映射为 0..4
            ds_obj.targets = np.where(
                t[:, None] == c10_classes[split_classes][None, :]
            )[1].tolist()
            return len(mask), mask.sum()

        # 对 train/val/test 各自应用（如果存在）
        for name, ds_obj in targets_sets.items():
            total, kept = _apply_split(ds_obj)
            print(f"{name.capitalize()}: {kept}/{total}")

        # 更新类别数为 5
        num_classes = 5

    return loaders_dict, num_classes


def loader(path, batch_size, num_workers, shuffle_train=True):
    train_dir = os.path.join(path, "train")
    test_dir = os.path.join(path, "adv_data")

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    transform_train = transforms.Compose(
        [
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )

    train_set = torchvision.datasets.ImageFolder(train_dir, transform=transform_train)
    test_set = torchvision.datasets.ImageFolder(test_dir, transform=transform_test)

    num_classes = 10

    return (
        {
            "train": torch.utils.data.DataLoader(
                train_set,
                batch_size=batch_size,
                shuffle=shuffle_train,
                num_workers=num_workers,
                pin_memory=True,
            ),
            "test": torch.utils.data.DataLoader(
                test_set,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            ),
        },
        num_classes,
    )
