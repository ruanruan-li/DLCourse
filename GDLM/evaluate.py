# 导入必要的库
import os
import torch
import argparse
import cv2
from tqdm import tqdm
import prettytable as pt
import eval.python.metrics as Measure


# 获取竞争者模型名称
def get_competitors(root):
    for model_name in os.listdir(root):
        print('\'{}\''.format(model_name), end=', ')


# 定义评估器函数
def evaluator(gt_pth_lst, pred_pth_lst):
    """
    评估函数，用于计算多个评价指标
    """
    # 定义测量指标
    FM = Measure.Fmeasure()
    WFM = Measure.WeightedFmeasure()
    SM = Measure.Smeasure()
    EM = Measure.Emeasure()
    MAE = Measure.MAE()

    assert len(gt_pth_lst) == len(pred_pth_lst)

    with torch.no_grad():
        for idx in tqdm(range(len(gt_pth_lst))):
            gt_pth = gt_pth_lst[idx]
            pred_pth = pred_pth_lst[idx]

            assert os.path.isfile(gt_pth) and os.path.isfile(pred_pth)
            pred_ary = cv2.imread(pred_pth, cv2.IMREAD_GRAYSCALE)
            gt_ary = cv2.imread(gt_pth, cv2.IMREAD_GRAYSCALE)

            assert len(pred_ary.shape) == 2 and len(gt_ary.shape) == 2
            if pred_ary.shape != gt_ary.shape:
                pred_ary = cv2.resize(pred_ary, (gt_ary.shape[1], gt_ary.shape[0]), cv2.INTER_NEAREST)

            FM.step(pred=pred_ary, gt=gt_ary)
            WFM.step(pred=pred_ary, gt=gt_ary)
            SM.step(pred=pred_ary, gt=gt_ary)
            EM.step(pred=pred_ary, gt=gt_ary)
            MAE.step(pred=pred_ary, gt=gt_ary)

        fm = FM.get_results()['fm']
        wfm = WFM.get_results()['wfm']
        sm = SM.get_results()['sm']
        em = EM.get_results()['em']
        mae = MAE.get_results()['mae']

    return fm, wfm, sm, em, mae


# 评估所有数据集
def eval_all(opt, txt_save_path):
    """
    对整个数据集进行评估
    """
    for _data_name in opt.data_lst:
        print('#' * 20, _data_name, '#' * 20)
        filename = os.path.join(txt_save_path, '{}_eval.txt'.format(_data_name))
        with open(filename, 'w+') as file_to_write:
            tb = pt.PrettyTable()
            tb.field_names = ["Dataset", "Method", "Smeasure", "wFmeasure", "MAE", "adpEm", "meanEm", "maxEm", "adpFm",
                              "meanFm", "maxFm"]
            for _model_name in opt.model_lst:
                print('#' * 10, _model_name, '#' * 10)
                gt_src = os.path.join(opt.gt_root, _data_name, 'GT')
                pred_src = os.path.join(opt.pred_root, _model_name, _data_name)

                # 获取有效的文件名列表
                img_name_lst = os.listdir(gt_src)

                fm, wfm, sm, em, mae = evaluator(
                    gt_pth_lst=[os.path.join(gt_src, i) for i in img_name_lst],
                    pred_pth_lst=[os.path.join(pred_src, i) for i in img_name_lst]
                )

                tb.add_row([_data_name, _model_name, sm.round(3), wfm.round(3), mae.round(3), em['adp'].round(3),
                            em['curve'].mean().round(3), em['curve'].max().round(3), fm['adp'].round(3),
                            fm['curve'].mean().round(3), fm['curve'].max().round(3)])
            print(tb)
            file_to_write.write(str(tb))


# 评估COD10K超类
def eval_super_class(opt):
    """
    对COD10K中的超类进行评估
    """
    _super_cls_lst = ['Aquatic', 'Flying', 'Amphibian', 'Terrestrial', 'Other']
    _data_name = 'COD10K'
    print('#' * 20, _data_name, '#' * 20)
    tb = pt.PrettyTable()
    tb.field_names = ["Dataset", "Method", "Smeasure", "wFmeasure", "MAE", "adpEm", "meanEm", "maxEm", "adpFm",
                      "meanFm", "maxFm"]

    for _model_name in opt.model_lst:
        for _super_cls in _super_cls_lst:
            fm, wfm, sm, em, mae = evaluator(
                gt_pth_lst=[i for i in os.listdir(os.path.join(opt.gt_root, _data_name, 'GT')) if _super_cls in i],
                pred_pth_lst=[i for i in os.listdir(os.path.join(opt.pred_root, _model_name, _data_name)) if
                              _super_cls in i]
            )
            tb.add_row([_super_cls, _model_name, sm.round(3), wfm.round(3), mae.round(3), em['adp'].round(3),
                        em['curve'].mean().round(3), em['curve'].max().round(3), fm['adp'].round(3),
                        fm['curve'].mean().round(3), fm['curve'].max().round(3)])
        print(tb)


# 评估COD10K子类
def eval_sub_class(opt):
    """
    对COD10K中的子类进行评估
    """
    _sub_cls_lst = []
    _data_name = 'COD10K'
    print('#' * 20, _data_name, '#' * 20)
    tb = pt.PrettyTable()
    tb.field_names = ["Dataset", "Method", "Smeasure", "wFmeasure", "MAE", "adpEm", "meanEm", "maxEm", "adpFm",
                      "meanFm", "maxFm"]

    for _model_name in opt.model_lst:
        for _sub_cls in _sub_cls_lst:
            fm, wfm, sm, em, mae = evaluator(
                gt_pth_lst=[i for i in os.listdir(os.path.join(opt.gt_root, _data_name, 'GT')) if _sub_cls in i],
                pred_pth_lst=[i for i in os.listdir(os.path.join(opt.pred_root, _model_name, _data_name)) if
                              _sub_cls in i]
            )
            tb.add_row([_sub_cls, _model_name, sm.round(3), wfm.round(3), mae.round(3), em['adp'].round(3),
                        em['curve'].mean().round(3), em['curve'].max().round(3), fm['adp'].round(3),
                        fm['curve'].mean().round(3), fm['curve'].max().round(3)])
        print(tb)


# 主函数
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_root', type=str, help='ground-truth根路径', default='../dataset/TestDataset')
    parser.add_argument('--pred_root', type=str, help='预测结果根路径', default='./result')
    parser.add_argument('--gpu_id', type=str, help='GPU设备编号', default='0')
    parser.add_argument('--data_lst', type=list, help='测试数据集', default=['CAMO', 'COD10K', 'NC4K'],
                        choices=['CHAMELEON', 'CAMO', 'COD10K', 'NC4K'])
    parser.add_argument('--model_lst', type=list, help='候选竞争者',
                        default=['Exp-GDLM-S', 'Exp-GDLM', 'Exp-GDLM-PVTv2-B0', 'Exp-GDLM-PVTv2-B1',
                                 'Exp-GDLM-PVTv2-B2', 'Exp-GDLM-PVTv2-B3'])
    parser.add_argument('--txt_name', type=str, help='保存结果的文件名', default='20221103_GDLM_benchmark')
    parser.add_argument('--check_integrity', type=bool, help='是否检查文件完整性', default=True)
    parser.add_argument('--eval_type', type=str, help='评估类型', default='eval_all',
                        choices=['eval_all', 'eval_super', 'eval_sub'])
    opt = parser.parse_args()

    txt_save_path = './eval_txt/{}/'.format(opt.txt_name)
    os.makedirs(txt_save_path, exist_ok=True)

    # 检查每个候选者的完整性
    if opt.check_integrity:
        for _data_name in opt.data_lst:
            for _model_name in opt.model_lst:
                gt_pth = os.path.join(opt.gt_root, _data_name, 'GT')
                pred_pth = os.path.join(opt.pred_root, _model_name, _data_name)
                if not sorted(os.listdir(gt_pth)) == sorted(os.listdir(pred_pth)):
                    print(len(sorted(os.listdir(gt_pth))), len(sorted(os.listdir(pred_pth))))
                    print('The {} Dataset of {} Model is not matching to the ground-truth'.format(_data_name,
                                                                                                  _model_name))
    else:
        print('>>> skip check the integrity of each candidates')

    # 根据评估类型进行评估
    if opt.eval_type == 'eval_all':
        eval_all(opt, txt_save_path)
    elif opt.eval_type == 'eval_super':
        eval_super_class(opt)
    elif opt.eval_type == 'eval_sub':
        eval_sub_class(opt)
    else:
        raise Exception("Invalid Evaluation Type: {}".format(opt.eval_type))