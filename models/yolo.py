# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
YOLO-specific modules
yolov5的模型搭建模块，非常重要
Usage:
    $ python path/to/models/yolo.py --cfg yolov5s.yaml
"""

import argparse             # 解析命令行参数模块
import contextlib           # 日志模块
import os                   # # sys系统模块 包含了与Python解释器和它的环境有关的函数
import platform
import sys
from copy import deepcopy   # 数据拷贝模块 深拷贝
from pathlib import Path    # Path将str转换为Path对象 使字符串路径易于操作的模块

from torch import nn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import *
from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import LOGGER, check_version, check_yaml, make_divisible, print_args
from utils.plots import feature_visualization
from utils.torch_utils import (fuse_conv_and_bn, initialize_weights, model_info, profile, scale_img, select_device,
                               time_sync)

# 导入thop包 用于计算FLOPs
try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None


# Detect模块是用来构建Detect层的，将输入feature map 通过一个卷积操作和公式计算到我们想要的shape，为后面的计算损失或者NMS作准备。
class Detect(nn.Module):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter
    export = False  # export mode

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers = 3
        self.na = len(anchors[0]) // 2  # number of anchors = 3
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        # 模型中需要保存的参数一般有两种：一种是反向传播需要被optimizer更新的，称为parameter; 另一种不要被更新称为buffer
        # buffer的参数更新是在forward中，而optim.step只能更新nn.parameter类型的参数
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)   反向传播不需要被optimizer更新，称之为buffer
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv 1*1卷积 得到预测后的值（坐标、分类概率。。）
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)

    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()  # 把no调到最后（25个值）

            if not self.training:  # inference
                # 构造网格
                # 因为推理返回的不是归一化后的网格偏移量 需要再加上网格的位置 得到最终的推理坐标 再送入nms
                # 所以这里构建网格就是为了纪律每个grid的网格坐标 方面后面使用
                if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)  # anchor_grid构造网格

                y = x[i].sigmoid()
                if self.inplace:    # 最后一个维度是85，其前四个元素[tx,ty,tw,th]经线性回归计算，得到[bx,by,bw,bh]
                    y[..., 0:2] = (y[..., 0:2] * 2 + self.grid[i]) * self.stride[i]  # xy坐标信息
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh坐标信息
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy, wh, conf = y.split((2, 2, self.nc + 1), 4)  # y.tensor_split((2, 4, 5), 4)  # torch 1.8.0
                    xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, -1, self.no))   # 预测框坐标信息

        return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)  # 预测框坐标 obj,cls

    # 划分单元网格的函数
    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        # 划分单元网格
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        if check_version(torch.__version__, '1.10.0'):  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
            yv, xv = torch.meshgrid(y, x, indexing='ij')
        else:
            yv, xv = torch.meshgrid(y, x)
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid


# 这个模块是整个模型的搭建模块
class Model(nn.Module):
    # YOLOv5 model
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        """
                :params cfg:模型配置文件
                :params ch: input img channels 一般是3 RGB文件
                :params nc: number of classes 数据集的类别个数
                :anchors: 一般是None
                """
        super().__init__()
        if isinstance(cfg, dict):   # 判断cfg是不是字典
            self.yaml = cfg  # model dict
        else:  # is *.yaml 一般执行这里
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name # 获取文件名
            with open(cfg, encoding='ascii', errors='ignore') as f:
                self.yaml = yaml.safe_load(f)  # model dict 以字典形式存放

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels 找文件中的ch，若找不到以init中的ch代替，即3
        # 设置类别数 一般不执行, 因为nc=self.yaml['nc']恒成立
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        # 重写anchor，一般不执行, 因为传进来的anchors一般都是None
        if anchors:
            LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        # 创建网络模型
        # self.model: 初始化的整个网络模型(包括Detect层结构)
        # self.save: 所有层结构中from不等于-1的序号，并排好序  [4, 6, 10, 14, 17, 20, 23]
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist 搭建网络的每一层
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names 初始化name参数，给每一个类附一个类名
        self.inplace = self.yaml.get('inplace', True)   # 加载inplace关键字，若没有则返回true

        # Build strides, anchors
        # 获取Detect模块的stride(相对输入图像的下采样率)和anchors在当前Detect输出的feature map的尺度
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            # 计算三个feature map下采样的倍率  [8, 16, 32]
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward [8,16,32]
            check_anchor_order(m)  # must be in pixel-space (not grid-space) 检查anchor的顺序对不对 低层用低层anchor
            # anchor大小计算，例如[10,13]->[1.25,1.625]
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self._initialize_biases()  # only run once 初始化偏置

        # Init weights, biases
        initialize_weights(self)    # 初始化权重
        self.info()
        LOGGER.info('')

    def forward(self, x, augment=False, profile=False, visualize=False):
        if augment:
            return self._forward_augment(x)  # augmented inference, None
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_augment(self, x):
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, 1), None  # augmented inference, train

    def _forward_once(self, x, profile=False, visualize=False):
        """
                :params x: 输入图像
                :params profile: True 可以做一些性能评估
                :params feature_vis: True 可以做一些特征可视化
                :return train: 一个tensor list 存放三个元素   [bs, anchor_num, grid_w, grid_h, xywh+c+20classes]
                               分别是 [1, 3, 80, 80, 25] [1, 3, 40, 40, 25] [1, 3, 20, 20, 25]
                        inference: 0 [1, 19200+4800+1200, 25] = [bs, anchor_num*grid_w*grid_h, xywh+c+20classes]
                                   1 一个tensor list 存放三个元素 [bs, anchor_num, grid_w, grid_h, xywh+c+20classes]
                                     [1, 3, 80, 80, 25] [1, 3, 40, 40, 25] [1, 3, 20, 20, 25]
        """
        y, dt = [], []  # outputs
        # 前向推理每一层结构   m.i=index   m.f=from   m.type=类名   m.np=number of params
        # if not from previous layer   m.f=当前层的输入来自哪一层的输出  s的m.f都是-1
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                # 这里需要做4个concat操作和1个Detect操作
                # concat操作如m.f=[-1, 6] x就有两个元素,一个是上一层的输出,另一个是index=6的层的输出 再送到x=m(x)做concat操作
                # Detect操作m.f=[17, 20, 23] x有三个元素,分别存放第17层第20层第23层的输出 再送到x=m(x)做Detect的forward
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            if profile: # 性能评估
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x

    def _descale_pred(self, p, flips, scale, img_size):
        """用在上面的__init__函数上
                将推理结果恢复到原图图片尺寸  Test Time Augmentation(TTA)中用到
                de-scale predictions following augmented inference (inverse operation)
                :params p: 推理结果
                :params flips:
                :params scale:
                :params img_size:
        """
        # de-scale predictions following augmented inference (inverse operation)
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        # Clip YOLOv5 augmented inference tails
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4 ** x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y

    def _profile_one_layer(self, m, x, dt):
        c = isinstance(m, Detect)  # is final layer, copy input as inplace fix
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency 初始化detect偏置
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1).detach()  # conv.bias(255) to (3,85)
            b[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b[:, 5:] += math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    # 打印模型中最后Detect层的偏置bias信息(也可以任选哪些层bias信息)
    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            LOGGER.info(
                ('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # 打印模型中Bottleneck层的权重参数weights信息(也可以任选哪些层weights信息)
    def _print_weights(self):
        for m in self.model.modules():
            if type(m) is Bottleneck:
                LOGGER.info('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    """用在detect.py、val.py
    fuse model Conv2d() + BatchNorm2d() layers
    调用torch_utils.py中的fuse_conv_and_bn函数和common.py中Conv模块的fuseforward函数"""
    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers   Conv2d+BN 融合
        LOGGER.info('Fusing layers... ')
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):  # hasattr判断对象是否包含对应的属性
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm    删除对象的属性
                m.forward = m.forward_fuse  # update forward
        self.info()
        return self

    def info(self, verbose=False, img_size=640):  # print model information
        """用在上面的__init__函数上
                调用torch_utils.py下model_info函数打印模型信息
        """
        model_info(self, verbose, img_size)

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self


# parse_model模块用来解析模型文件(从Model中传来的字典形式)，并搭建网络结构。
"""用在上面Model模块中
    解析模型文件(字典形式)，并搭建网络结构
    这个函数其实主要做的就是: 更新当前层的args（参数）,计算c2（当前层的输出channel） =>
                          使用当前层的参数搭建当前层 =>
                          生成 layers + save
    :params d: model_dict 模型文件 字典形式 {dict:7}  yolov5s.yaml中的6个元素 + ch
    :params ch: 记录模型每一层的输出channel 初始ch=[3] 后面会删除
    :return nn.Sequential(*layers): 网络的每一层的层结构
    :return sorted(save): 把所有层结构中from不是-1的值记下 并排序 [4, 6, 10, 14, 17, 20, 23]
"""
def parse_model(d, ch):  # model_dict, input_channels(3)列表中只有3
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")    # 打印信息
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors chahor数量
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5) 输出通道数（80+5）*3=255

    # 开始搭建网络
    layers, save, c2 = [], [], ch[-1]  # layers下面的网络, savelist哪些层需要保存, ch out输出通道数
    # from(当前层输入来自哪些层), number(当前层次数 初定), module(当前层类别), args(当前层类参数 初定)
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        # eval(string) 得到当前层的真实类名 例如: m= Focus -> <class 'models.common.Focus'>
        m = eval(m) if isinstance(m, str) else m  # eval strings    eval推断是什么类型
        for j, a in enumerate(args):
            with contextlib.suppress(NameError):
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings 推断成int

        # ------------------- 更新当前层的args（参数）,计算c2（当前层的输出channel） -------------------
        # depth gain 控制深度  如v5s: n*0.33   n: 当前模块的次数(间接控制深度)
        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in (Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
                 BottleneckCSP, C3, C3TR, C3SPP, C3Ghost, nn.ConvTranspose2d, DWConvTranspose2d, C3x):
            c1, c2 = ch[f], args[0] # c1输入，c2输出
            if c2 != no:  # if not output 判断c2是否是最终输出255
                c2 = make_divisible(c2 * gw, 8) # 调整为8的倍数（对gpu计算更友好）

            # 在初始arg的基础上更新 加入当前层的输入channel并更新当前层
            args = [c1, c2, *args[1:]]  # c1 c2 args的后3个参数 拼接起来
            # 如果当前层是BottleneckCSP/C3/C3TR, 则需要在args中加入bottleneck的个数
            if m in [BottleneckCSP, C3, C3TR, C3Ghost, C3x]:    # 如果为C3层
                args.insert(2, n)  # number of repeats  额外将n加入
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]  # BN层只需要返回上一层的输出channel
        elif m is Concat:
            c2 = sum(ch[x] for x in f)   # Concat层则将f中所有的输出累加得到这层的输出channel
        elif m is Detect:
            args.append([ch[x] for x in f]) # 在args中加入三个Detect层的输出channel
            if isinstance(args[1], int):  # number of anchors    几乎不执行
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Contract: # 不怎么用
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:   # 不怎么用
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]      # args不变

        # *args表示接收任意个数量的参数，调用时会将实际参数打包成一个元组传入实参
        # m_: 得到当前层module  如果n>1就创建多个m(当前层结构), 如果n=1就创建一个m
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type 用''替换掉__main__
        np = sum(x.numel() for x in m_.parameters())  # number params 统计第0层参数量
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        # 把所有层结构中from不是-1的值记下  [6, 4, 14, 10, 17, 20, 23]
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist 统计哪些层需要保存（需要concat的层）
        layers.append(m_)
        if i == 0:
            ch = [] # 去除输入channel [3]
        ch.append(c2)   # [32] [32,64] [32,64,64] 按照-1取出上一层的输出通道
    return nn.Sequential(*layers), sorted(save) # [6,4,14,10,17,20,23]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    parser.add_argument('--batch-size', type=int, default=1, help='total batch size for all GPUs')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--profile', action='store_true', help='profile model speed')
    parser.add_argument('--line-profile', action='store_true', help='profile model speed layer by layer')
    parser.add_argument('--test', action='store_true', help='test all yolo*.yaml')
    opt = parser.parse_args()   # 解析参数
    opt.cfg = check_yaml(opt.cfg)  # check YAML
    print_args(vars(opt))
    device = select_device(opt.device)

    # Create model
    im = torch.rand(opt.batch_size, 3, 640, 640).to(device)
    model = Model(opt.cfg).to(device)

    # Options
    if opt.line_profile:  # profile layer by layer
        _ = model(im, profile=True)

    elif opt.profile:  # profile forward-backward
        results = profile(input=im, ops=[model], n=3)

    elif opt.test:  # test all models
        for cfg in Path(ROOT / 'models').rglob('yolo*.yaml'):
            try:
                _ = Model(cfg)
            except Exception as e:
                print(f'Error in {cfg}: {e}')

    else:  # report fused model summary
        model.fuse()
