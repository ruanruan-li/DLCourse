import os
import random
import shutil
from collections import defaultdict

# 设置随机种子以保证结果可复现
random.seed(42)

# 数据集路径和大小
datasets = {
    'dataset1': 76,
    'dataset2': 1250,
    'dataset3': 4121,
    'dataset4': 10000
}

# 总样本数量
total_samples = sum(datasets.values())

# 抽样比例
train_ratio = 10813 / total_samples
val_ratio = 2317 / total_samples
test_ratio = 2317 / total_samples

# 存储路径
output_dirs = {
    'train': 'dataset/train',
    'val': 'dataset/val',
    'test': 'dataset/test'
}

# 创建存储文件夹
for dir_path in output_dirs.values():
    os.makedirs(dir_path, exist_ok=True)


# 分层抽样函数
def stratified_sampling(dataset_path, dataset_size, train_ratio, val_ratio, test_ratio):
    images = os.listdir(dataset_path)
    random.shuffle(images)

    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = int(test_ratio * dataset_size)

    train_samples = images[:train_size]
    val_samples = images[train_size:train_size + val_size]
    test_samples = images[train_size + val_size:train_size + val_size + test_size]

    return train_samples, val_samples, test_samples


# 用于存储所有数据集的样本
all_samples = defaultdict(list)

# 进行分层抽样
for dataset_name, dataset_size in datasets.items():
    dataset_path = os.path.join('datasets', dataset_name)

    train_samples, val_samples, test_samples = stratified_sampling(dataset_path, dataset_size, train_ratio, val_ratio,
                                                                   test_ratio)

    all_samples['train'].extend([(dataset_path, img) for img in train_samples])
    all_samples['val'].extend([(dataset_path, img) for img in val_samples])
    all_samples['test'].extend([(dataset_path, img) for img in test_samples])

# 确保样本数量一致
all_samples['train'] = all_samples['train'][:10813]
all_samples['val'] = all_samples['val'][:2317]
all_samples['test'] = all_samples['test'][:2317]

# 将样本复制到对应的文件夹
for split, samples in all_samples.items():
    for dataset_path, img in samples:
        src_path = os.path.join(dataset_path, img)
        dst_path = os.path.join(output_dirs[split], img)
        shutil.copy(src_path, dst_path)

print("数据集分层抽样和存储完成。")