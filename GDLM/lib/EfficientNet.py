"""model.py - EfficientNet 的模型和模块类。
   它们的构建与官方的 TensorFlow 实现相似。
"""

# 作者: lukemelas (github 用户名)
# Github 仓库: https://github.com/lukemelas/EfficientNet-PyTorch

import torch
from torch import nn
from torch.nn import functional as F
from .utils import (
    round_filters,
    round_repeats,
    drop_connect,
    get_same_padding_conv2d,
    get_model_params,
    efficientnet_params,
    load_pretrained_weights,
    Swish,
    MemoryEfficientSwish,
    calculate_output_image_size
)

# 合法的模型名称
VALID_MODELS = (
    'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3',
    'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7',
    'efficientnet-b8',
    'efficientnet-l2'  # 支持构建 'efficientnet-l2'，但没有预训练权重
)

class MBConvBlock(nn.Module):
    """移动倒置残差瓶颈块。

    参数:
        block_args (namedtuple): BlockArgs，定义在 utils.py 中。
        global_params (namedtuple): GlobalParam，定义在 utils.py 中。
        image_size (tuple 或 list): [image_height, image_width]。

    参考文献:
        [1] https://arxiv.org/abs/1704.04861 (MobileNet v1)
        [2] https://arxiv.org/abs/1801.04381 (MobileNet v2)
        [3] https://arxiv.org/abs/1905.02244 (MobileNet v3)
    """

    def __init__(self, block_args, global_params, image_size=None):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum  # pytorch 与 tensorflow 的不同之处
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # 是否使用跳跃连接和 drop connect

        # 扩展阶段（倒置瓶颈）
        inp = self._block_args.input_filters  # 输入通道数
        oup = self._block_args.input_filters * self._block_args.expand_ratio  # 输出通道数
        if self._block_args.expand_ratio != 1:
            Conv2d = get_same_padding_conv2d(image_size=image_size)
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # 深度卷积阶段
        k = self._block_args.kernel_size
        s = self._block_args.stride
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._depthwise_conv = Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups 使其成为深度卷积
            kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
        image_size = calculate_output_image_size(image_size, s)

        # 挤压与扩展层，若需要
        if self.has_se:
            Conv2d = get_same_padding_conv2d(image_size=(1, 1))
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # 逐点卷积阶段
        final_oup = self._block_args.output_filters
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        self._swish = MemoryEfficientSwish()

    def forward(self, inputs, drop_connect_rate=None):
        """MBConvBlock 的前向函数。

        参数:
            inputs (tensor): 输入张量。
            drop_connect_rate (bool): Drop connect 率（浮点数，介于 0 和 1 之间）。

        返回:
            经过处理后的块的输出。
        """
        # 扩展和深度卷积
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._expand_conv(inputs)
            x = self._bn0(x)
            x = self._swish(x)

        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = self._swish(x)

        # 挤压与扩展
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_reduce(x_squeezed)
            x_squeezed = self._swish(x_squeezed)
            x_squeezed = self._se_expand(x_squeezed)
            x = torch.sigmoid(x_squeezed) * x

        # 逐点卷积
        x = self._project_conv(x)
        x = self._bn2(x)

        # 跳跃连接和 drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # 跳跃连接
        return x

    def set_swish(self, memory_efficient=True):
        """设置 swish 函数为内存高效（用于训练）或标准（用于导出）。

        参数:
            memory_efficient (bool): 是否使用内存高效版本的 swish。
        """
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()


class EfficientNet(nn.Module):
    """EfficientNet 模型。
       最容易通过 .from_name 或 .from_pretrained 方法加载。

    参数:
        blocks_args (list[namedtuple]): 构建块的 BlockArgs 列表。
        global_params (namedtuple): 在块之间共享的一组 GlobalParams。

    参考文献:
        [1] https://arxiv.org/abs/1905.11946 (EfficientNet)

    示例:
        >>> import torch
        >>> from efficientnet.model1 import EfficientNet
        >>> inputs = torch.rand(1, 3, 224, 224)
        >>> model = EfficientNet.from_pretrained('efficientnet-b0')
        >>> model.eval()
        >>> outputs = model(inputs)
    """

    def __init__(self, blocks_args=None, global_params=None):
        super().__init__()
        assert isinstance(blocks_args, list), 'blocks_args 应该是一个列表'
        assert len(blocks_args) > 0, 'block args 必须大于 0'
        self._global_params = global_params
        self._blocks_args = blocks_args

        # Batch norm 参数
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # 获取根据图像大小的静态或动态卷积
        image_size = global_params.image_size
        Conv2d = get_same_padding_conv2d(image_size=image_size)

        # Stem
        in_channels = 3  # RGB
        out_channels = round_filters(32, self._global_params)  # 输出通道数
        self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
        image_size = calculate_output_image_size(image_size, 2)

        # 构建块
        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:
            # 更新块输入和输出滤波器基于深度乘数。
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )

            # 第一个块需要处理步幅和滤波器大小的增加。
            self._blocks.append(MBConvBlock(block_args, self._global_params, image_size=image_size))
            image_size = calculate_output_image_size(image_size, block_args.stride)
            if block_args.num_repeat > 1:  # 修改 block_args 以保持相同的输出大小
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self._global_params, image_size=image_size))

        # Head
        in_channels = block_args.output_filters  # 最终块的输出
        out_channels = round_filters(1280, self._global_params)
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # 最后的线性层
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)

        # 默认情况下将激活函数设置为内存高效的 swish
        self._swish = MemoryEfficientSwish()

    def set_swish(self, memory_efficient=True):
        """设置 swish 函数为内存高效（用于训练）或标准（用于导出）。

        参数:
            memory_efficient (bool): 是否使用内存高效版本的 swish。
        """
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()
        for block in self._blocks:
            block.set_swish(memory_efficient)

    def extract_endpoints(self, inputs):
        """使用卷积层从降低的不同级别提取特征。

        参数:
            inputs (tensor): 输入张量。

        返回:
            包含降低的不同级别特征的字典。
        """
        endpoints = dict()

        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)  # 缩放 drop connect 率
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints) + 1)] = prev_x
            elif idx == len(self._blocks) - 1:
                endpoints['reduction_{}'.format(len(endpoints) + 1)] = x
            prev_x = x

        # Head
        x = self._swish(self._bn1(self._conv_head(x)))
        endpoints['reduction_{}'.format(len(endpoints) + 1)] = x

        return endpoints

    def extract_endpoints_dual(self, inputs, grad_feats):
        """使用卷积层从降低的不同级别提取特征（带有额外的梯度特征）。

        参数:
            inputs (tensor): 输入张量。

        返回:
            包含降低的不同级别特征和梯度特征的字典。
        """
        endpoints = dict()

        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)  # 缩放 drop connect 率
            if idx < 2:
                x = x + grad_feats[idx]
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints) + 1)] = prev_x
            elif idx == len(self._blocks) - 1:
                endpoints['reduction_{}'.format(len(endpoints) + 1)] = x
            prev_x = x

        # Head
        x = self._swish(self._bn1(self._conv_head(x)))
        endpoints['reduction_{}'.format(len(endpoints) + 1)] = x

        return endpoints

    def extract_features(self, inputs):
        """使用卷积层提取特征。

        参数:
            inputs (tensor): 输入张量。

        返回:
            efficientnet 模型的最终卷积层的输出。
        """
        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)  # 缩放 drop connect 率
            x = block(x, drop_connect_rate=drop_connect_rate)

        # Head
        x = self._swish(self._bn1(self._conv_head(x)))

        return x

    def forward(self, inputs):
        """EfficientNet 的前向函数。
           调用 extract_features 提取特征，应用最终线性层，并返回 logits。

        参数:
            inputs (tensor): 输入张量。

        返回:
            经过处理后的模型输出。
        """
        # 卷积层
        x = self.extract_features(inputs)
        # 池化和最终线性层
        x = self._avg_pooling(x)
        return x

    @classmethod
    def from_name(cls, model_name, in_channels=3, **override_params):
        """根据名称创建 efficientnet 模型。

        参数:
            model_name (str): efficientnet 的名称。
            in_channels (int): 输入数据的通道数。
            override_params (其他关键字参数):
                覆盖模型的 global_params 的参数。
                可选键:
                    'width_coefficient', 'depth_coefficient',
                    'image_size', 'dropout_rate',
                    'num_classes', 'batch_norm_momentum',
                    'batch_norm_epsilon', 'drop_connect_rate',
                    'depth_divisor', 'min_depth'

        返回:
            一个 efficientnet 模型。
        """
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, override_params)
        model = cls(blocks_args, global_params)
        model._change_in_channels(in_channels)
        return model

    @classmethod
    def from_pretrained(cls, model_name, weights_path=None, advprop=False,
                        in_channels=3, num_classes=1000, **override_params):
        """根据名称创建一个预训练的 efficientnet 模型。

        参数:
            model_name (str): efficientnet 的名称。
            weights_path (None 或 str):
                str: 本地磁盘上的预训练权重文件路径。
                None: 使用从互联网下载的预训练权重。
            advprop (bool):
                是否加载 advprop 训练的预训练权重（当 weights_path 为 None 时有效）。
            in_channels (int): 输入数据的通道数。
            num_classes (int):
                分类的类别数。
                它控制最终线性层的输出大小。
            override_params (其他关键字参数):
                覆盖模型的 global_params 的参数。
                可选键:
                    'width_coefficient', 'depth_coefficient',
                    'image_size', 'dropout_rate',
                    'batch_norm_momentum',
                    'batch_norm_epsilon', 'drop_connect_rate',
                    'depth_divisor', 'min_depth'

        返回:
            一个预训练的 efficientnet 模型。
        """
        model = cls.from_name(model_name, num_classes=num_classes, **override_params)
        load_pretrained_weights(model, model_name, weights_path=weights_path,
                                load_fc=False, advprop=advprop)
        model._change_in_channels(in_channels)
        return model

    @classmethod
    def get_image_size(cls, model_name):
        """获取给定 efficientnet 模型的输入图像大小。

        参数:
            model_name (str): efficientnet 的名称。

        返回:
            输入图像大小（分辨率）。
        """
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = efficientnet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name):
        """验证模型名称。

        参数:
            model_name (str): efficientnet 的名称。

        返回:
            bool: 是否为合法名称。
        """
        if model_name not in VALID_MODELS:
            raise ValueError('model_name 应该是以下之一: ' + ', '.join(VALID_MODELS))

    def _change_in_channels(self, in_channels):
        """调整模型的第一个卷积层以适应 in_channels，如果 in_channels 不等于 3。

        参数:
            in_channels (int): 输入数据的通道数。
        """
        if in_channels != 3:
            Conv2d = get_same_padding_conv2d(image_size=self._global_params.image_size)
            out_channels = round_filters(32, self._global_params)
            self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)