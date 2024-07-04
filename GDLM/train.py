# 导入必要的库
import os
import logging
import numpy as np
from datetime import datetime
from tensorboardX import SummaryWriter
from lib.GDLM import GDLM as Network
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch import optim
from torchvision.utils import make_grid
import eval.python.metrics as Measure
from utils.utils import clip_gradient
from utils.dataset import get_loader, test_dataset

# 定义结构化损失函数
def structure_loss(pred, mask):
    """
    结构化损失函数 (参考: F3Net-AAAI-2020)
    """
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

# 定义训练函数
def train(train_loader, model, optimizer, epoch, save_path, writer):
    """
    训练函数
    """
    global step
    model.train()
    loss_all = 0
    epoch_step = 0
    try:
        for i, (images, gts, grads) in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            images, gts, grads = images.cuda(), gts.cuda(), grads.cuda()

            preds = model(images)
            loss_pred = structure_loss(preds[0], gts)
            loss_grad = grad_loss_func(preds[1], grads)
            loss = loss_pred + loss_grad

            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()

            step += 1
            epoch_step += 1
            loss_all += loss.data

            if i % 20 == 0 or i == total_step or i == 1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f} loss_pred: {:.4f} loss_grad: {:0.4f}'.format(
                      datetime.now(), epoch, opt.epoch, i, total_step, loss.data, loss_pred.data, loss_grad.data))
                logging.info('[Train Info]:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f} loss_pred: {:.4f} loss_grad: {:0.4f}'.format(
                             epoch, opt.epoch, i, total_step, loss.data, loss_pred.data, loss_grad.data))
                writer.add_scalars('Loss_Statistics',
                                   {'loss_pred': loss_pred.data, 'loss_grad': loss_grad.data, 'Loss_total': loss.data},
                                   global_step=step)
                grid_image = make_grid(images[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('RGB', grid_image, step)
                grid_image = make_grid(gts[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('GT', grid_image, step)

                res = preds[0][0].clone().sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_image('Pred_final', torch.tensor(res), step, dataformats='HW')
                res = preds[1][0].clone().sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_image('Pred_grad', torch.tensor(res), step, dataformats='HW')

        loss_all /= epoch_step
        logging.info('[Train Info]: Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
        if epoch % 50 == 0:
            torch.save(model.state_dict(), save_path + 'Net_epoch_{}.pth'.format(epoch))
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path + 'Net_epoch_{}.pth'.format(epoch + 1))
        print('Save checkpoints successfully!')
        raise

# 定义验证函数
def val(test_loader, model, epoch, save_path, writer):
    """
    验证函数
    """
    global best_metric_dict, best_score, best_epoch
    FM, SM, EM = Measure.Fmeasure(), Measure.Smeasure(), Measure.Emeasure()
    metrics_dict = dict()

    model.eval()
    with torch.no_grad():
        for i in range(test_loader.size):
            image, gt, _, _ = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            image = image.cuda()

            res = model(image)
            res = F.upsample(res[0], size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)

            FM.step(pred=res, gt=gt)
            SM.step(pred=res, gt=gt)
            EM.step(pred=res, gt=gt)

        metrics_dict.update(Sm=SM.get_results()['sm'])
        metrics_dict.update(mxFm=FM.get_results()['fm']['curve'].max().round(3))
        metrics_dict.update(mxEm=EM.get_results()['em']['curve'].max().round(3))

        cur_score = metrics_dict['Sm'] + metrics_dict['mxFm'] + metrics_dict['mxEm']
        if epoch == 1:
            best_score = cur_score
            print('[Cur Epoch: {}] Metrics (mxFm={}, Sm={}, mxEm={})'.format(
                epoch, metrics_dict['mxFm'], metrics_dict['Sm'], metrics_dict['mxEm']))
            logging.info('[Cur Epoch: {}] Metrics (mxFm={}, Sm={}, mxEm={})'.format(
                epoch, metrics_dict['mxFm'], metrics_dict['Sm'], metrics_dict['mxEm']))
        else:
            if cur_score > best_score:
                best_metric_dict = metrics_dict
                best_score = cur_score
                best_epoch = epoch
                torch.save(model.state_dict(), save_path + 'Net_epoch_best.pth')
                print('>>> save state_dict successfully! best epoch is {}.'.format(epoch))
            else:
                print('>>> not find the best epoch -> continue training ...')
            print('[Cur Epoch: {}] Metrics (mxFm={}, Sm={}, mxEm={})\n[Best Epoch: {}] Metrics (mxFm={}, Sm={}, mxEm={})'.format(
                epoch, metrics_dict['mxFm'], metrics_dict['Sm'], metrics_dict['mxEm'],
                best_epoch, best_metric_dict['mxFm'], best_metric_dict['Sm'], best_metric_dict['mxEm']))
            logging.info('[Cur Epoch: {}] Metrics (mxFm={}, Sm={}, mxEm={})\n[Best Epoch:{}] Metrics (mxFm={}, Sm={}, mxEm={})'.format(
                epoch, metrics_dict['mxFm'], metrics_dict['Sm'], metrics_dict['mxEm'],
                best_epoch, best_metric_dict['mxFm'], best_metric_dict['Sm'], best_metric_dict['mxEm']))

# 主函数
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=100, help='epoch数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--batchsize', type=int, default=12, help='训练批大小')
    parser.add_argument('--trainsize', type=int, default=352, help='训练集大小')
    parser.add_argument('--clip', type=float, default=0.5, help='梯度裁剪边界')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='学习率衰减率')
    parser.add_argument('--decay_epoch', type=int, default=50, help='每n个epoch衰减学习率')
    parser.add_argument('--model', type=str, default='GDLM', choices=['GDLM', 'GDLM-S', 'GDLM-PVTv2-B0', 'GDLM-PVTv2-B1', 'GDLM-PVTv2-B2', 'GDLM-PVTv2-B3', 'GDLM-PVTv2-B4'])
    parser.add_argument('--load', type=str, default=None, help='从检查点加载模型')
    parser.add_argument('--train_root', type=str, default='../dataset/TrainDataset/', help='训练RGB图像根路径')
    parser.add_argument('--val_root', type=str, default='../dataset/TestDataset/CAMO/', help='测试RGB图像根路径')
    parser.add_argument('--gpu_id', type=str, default='1', help='使用的GPU编号')
    parser.add_argument('--save_path', type=str, default='./lib_pytorch/snapshot/Exp02/', help='模型和日志的保存路径')
    opt = parser.parse_args()

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

    grad_loss_func = torch.nn.MSELoss()

    if opt.load is not None:
        model.load_state_dict(torch.load(opt.load))
        print('load model from ', opt.load)

    optimizer = torch.optim.Adam(model.parameters(), opt.lr)

    save_path = opt.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 加载数据
    print('load data...')
    train_loader = get_loader(image_root=opt.train_root + 'Imgs/', gt_root=opt.train_root + 'GT/', grad_root=opt.train_root + 'Gradient-Foreground/', batchsize=opt.batchsize, trainsize=opt.trainsize, num_workers=4)
    val_loader = test_dataset(image_root=opt.val_root + 'Imgs/', gt_root=opt.val_root + 'GT/', testsize=opt.trainsize)
    total_step = len(train_loader)

    # 日志记录
    logging.basicConfig(filename=save_path + 'log.log', format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]', level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info(">>> current mode: network-train/val")
    logging.info('>>> config: {}'.format(opt))
    print('>>> config: : {}'.format(opt))

    step = 0
    writer = SummaryWriter(save_path + 'summary')

    best_score = 0
    best_epoch = 0

    cosine_schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=20, eta_min=1e-5)
    print(">>> start train...")
    for epoch in range(1, opt.epoch):
        # 调度学习率
        cosine_schedule.step()
        writer.add_scalar('learning_rate', cosine_schedule.get_lr()[0], global_step=epoch)
        logging.info('>>> current lr: {}'.format(cosine_schedule.get_lr()[0]))
        # 训练
        train(train_loader, model, optimizer, epoch, save_path, writer)
        if epoch > opt.epoch // 2:
            # 验证
            val(val_loader, model, epoch, save_path, writer)