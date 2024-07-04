# 导入必要的库
import os
import torch
import argparse
import numpy as np
from scipy import misc
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from utils.dataset import test_dataset as EvalDataset
from lib.GDLM import GDLM as Network


# 定义评估器函数
def evaluator(model, val_root, map_save_path, trainsize=352):
    """
    评估函数，用于生成模型的预测结果并保存
    """
    val_loader = EvalDataset(image_root=val_root + 'Imgs/', gt_root=val_root + 'GT/', testsize=trainsize)

    model.eval()
    with torch.no_grad():
        for i in range(val_loader.size):
            image, gt, name, _ = val_loader.load_data()
            gt = np.asarray(gt, np.float32)

            image = image.cuda()

            output = model(image)
            output = F.upsample(output[0], size=gt.shape, mode='bilinear', align_corners=False)
            output = output.sigmoid().data.cpu().numpy().squeeze()
            output = (output - output.min()) / (output.max() - output.min() + 1e-8)

            misc.imsave(map_save_path + name, output)
            print('>>> saving prediction at: {}'.format(map_save_path + name))


# 主函数
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='GDLM',
                        choices=['GDLM', 'GDLM-S', 'GDLM-PVTv2-B0', 'GDLM-PVTv2-B1', 'GDLM-PVTv2-B2',
                                 'GDLM-PVTv2-B3', 'GDLM-PVTv2-B4'])
    parser.add_argument('--snap_path', type=str, default='./snapshot/GDLM/Net_epoch_best.pth',
                        help='模型快照路径')
    parser.add_argument('--gpu_id', type=str, default='1', help='使用的GPU编号')
    opt = parser.parse_args()

    txt_save_path = './result/{}/'.format(opt.snap_path.split('/')[-2])
    os.makedirs(txt_save_path, exist_ok=True)

    print('>>> configs:', opt)

    # 设置训练设备
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
    print(f'USE GPU {opt.gpu_id}')
    cudnn.benchmark = True

    # 构建模型
    if opt.model == 'GDLM':
        model = Network(channel=64, arc='EfficientNet-B4', M=[8, 8, 8], N=[4, 8, 16]).cuda()
    elif opt.model == 'GDLM-S':
        model = Network(channel=32, arc='EfficientNet-B1', M=[8, 8, 8], N=[8, 16, 32]).cuda()
    elif opt.model == 'GDLM-PVTv2-B0':
        model = Network(channel=32, arc='PVTv2-B0', M=[8, 8, 8], N=[8, 16, 32]).cuda()
    elif opt.model == 'GDLM-PVTv2-B1':
        model = Network(channel=64, arc='PVTv2-B1', M=[8, 8, 8], N=[4, 8, 16]).cuda()
    elif opt.model == 'GDLM-PVTv2-B2':
        model = Network(channel=64, arc='PVTv2-B2', M=[8, 8, 8], N=[4, 8, 16]).cuda()
    elif opt.model == 'GDLM-PVTv2-B3':
        model = Network(channel=64, arc='PVTv2-B3', M=[8, 8, 8], N=[4, 8, 16]).cuda()
    elif opt.model == 'GDLM-PVTv2-B4':
        model = Network(channel=64, arc='PVTv2-B4', M=[8, 8, 8], N=[4, 8, 16]).cuda()
    else:
        raise Exception("Invalid Model Symbol: {}".format(opt.model))

    # 加载模型权重
    model.load_state_dict(torch.load(opt.snap_path), strict=False)
    model.eval()

    # 评估测试数据集
    for data_name in ['CAMO', 'COD10K', 'NC4K']:
        map_save_path = txt_save_path + "{}/".format(data_name)
        os.makedirs(map_save_path, exist_ok=True)
        evaluator(
            model=model,
            val_root='../dataset/TestDataset/' + data_name + '/',
            map_save_path=map_save_path,
            trainsize=352)