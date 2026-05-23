import os
import math
import time
import shutil
import datetime
import numpy as np
import cv2
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
from PIL import Image

from contextlib import contextmanager
from collections import defaultdict
from glob import glob

class Timer:
    """
    计时器类，用于记录和统计代码块的执行时间
    
    使用方法:
    1. 作为上下文管理器:
        with Timer("name"):
            # code block
            
    2. 作为装饰器:
        @Timer.timer("name")
        def function():
            # code block
            
    3. 手动开始/结束:
        Timer.start("name")
        # code block
        Timer.end("name")
    """
    
    _records = defaultdict(list)  # 存储所有计时记录
    _starts = {}  # 存储开始时间
    
    @classmethod
    @contextmanager
    def timer(cls, name):
        """上下文管理器方式使用"""
        try:
            cls.start(name)
            yield
        finally:
            cls.end(name)
    
    @classmethod
    def start(cls, name):
        """开始计时"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        cls._starts[name] = time.perf_counter()
    
    @classmethod
    def end(cls, name):
        """结束计时"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        duration = time.perf_counter() - cls._starts[name]
        cls._records[name].append(duration)
    
    @classmethod
    def get_avg_time(cls, name):
        """获取平均时间"""
        if name in cls._records:
            return np.mean(cls._records[name])
        return 0
    
    @classmethod
    def get_total_time(cls, name):
        """获取总时间"""
        if name in cls._records:
            return np.sum(cls._records[name])
        return 0
    
    @classmethod
    def get_count(cls, name):
        """获取调用次数"""
        if name in cls._records:
            return len(cls._records[name])
        return 0
    
    @classmethod
    def summary(cls):
        """打印所有计时统计信息"""
        print("\n---------------------- Timer Summary ----------------------")
        print(f"{'Name':<30} {'Count':>8} {'Avg(ms)':>10} {'Total(s)':>10}")
        print("-" * 60)
        
        for name in sorted(cls._records.keys()):
            count = cls.get_count(name)
            avg_time = cls.get_avg_time(name) * 1000  # 转换为毫秒
            total_time = cls.get_total_time(name)
            print(f"{name:<30} {count:>8d} {avg_time:>10.2f} {total_time:>10.2f}")
        print("-" * 60)
    
    @classmethod
    def reset(cls):
        """重置所有计时记录"""
        cls._records.clear()
        cls._starts.clear()



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self):
        return time.time() - self.t0

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0

class checkpoint_FiveKNew():
    def __init__(self, args):
        self.args = args
        self.results_table = np.zeros((11, 3))
        date = datetime.datetime.now().strftime("%m-%d") + '-' + args.task

        if args.folder is not None:
            self.test_dir = args.folder

            overview_file = os.path.join(self.test_dir, 'result_overview.txt')
            if os.path.exists(overview_file):
                with open(overview_file, 'r') as f:
                    lines = f.readlines()
                if len(lines) > 0:
                    last_line = lines[-1]
                    self.id_start = last_line.split(',')[-1].split(':')[1].strip()

            for file in sorted(glob(os.path.join(self.test_dir, 'result_*.txt'))):
                label = file.split('_')[-1].split('.')[0]
                if len(label) == 1:
                    label = int(label)
                else:
                    continue

                with open(file, 'r') as f:
                    lines = f.readlines()

                for line in lines:
                    values = line.split(',')
                    psnr = float(values[0].split(':')[1])
                    ssim = float(values[1].split(':')[1])
                    rmse = float(values[2].split(':')[1])
                    im_id = values[3].split(':')[1].strip()
                    self.write_result(psnr, ssim, rmse, im_id, label, write_files=False)
        else:
            self.test_dir = './results/' + date

        self._make_dir(self.test_dir)
        self._make_dir(os.path.join(self.test_dir, 'error_map'))
        self._make_dir(os.path.join(self.test_dir, 'pred'))

        yaml_dir = os.path.join(self.test_dir, args.yaml.split('/')[-1])
        shutil.copy(args.yaml, yaml_dir)

    def _make_dir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)


    def write_result(self, psnr, ssim, im_id, label, write_files=True):
        label = label + 1

        self.results_table[0][0] += psnr
        self.results_table[0][1] += ssim
        self.results_table[0][2] += 1

        self.results_table[label][0] += psnr
        self.results_table[label][1] += ssim
        self.results_table[label][2] += 1

        info = "PSNR:{:.4f}, SSIM:{:.4f}, image_id:{}\n".format(
            psnr, ssim, im_id
        )

        if write_files:
            file = os.path.join(self.test_dir, 'result_{}.txt'.format(label))
            open_type = 'a' if os.path.exists(file) else 'w'
            with open(file, open_type) as f:
                f.writelines(info)
            print(info)

            file = os.path.join(self.test_dir, 'result_overview.txt')
            open_type = 'a' if os.path.exists(file) else 'w'
            with open(file, open_type) as f:
                f.writelines(info)

    def cal_final_result(self):
        file = os.path.join(self.test_dir, 'result_all.txt')
        open_type = 'a' if os.path.exists(file) else 'w'

        with open(file, open_type) as f:
            for i, results in enumerate(self.results_table):
                count = max(results[2], 1)
                psnr_avg = results[0] / count
                ssim_avg = results[1] / count

                f.writelines(
                    "PSNR_avg:{:.2f}, SSIM_avg:{:.4f}, camera_id:{}\n".format(
                        psnr_avg, ssim_avg, i
                    )
                )

    def save_image(self, im, id, mode=None):
        im = im.squeeze(0).clip_(0, 1)
        c, _, _ = im.shape

        if c == 4:
            im = torch.stack([im[0], (im[1] + im[2]) / 2, im[-1]], dim=0)

        save_dir = os.path.join(self.test_dir, mode)
        self._make_dir(save_dir)

        save_path = os.path.join(save_dir, id + '.jpg')
        self._make_dir(os.path.dirname(save_path))

        Image.fromarray((im * 255).byte().permute(1, 2, 0).cpu().numpy()).save(save_path)

def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)


def calc_psnr(sr, hr, scale, rgb_range, benchmark=False):
    diff = (sr - hr).data.div(rgb_range)
    # print(diff.shape)
    if benchmark:
        shave = scale
        if diff.size(1) > 1:
            convert = diff.new(1, 3, 1, 1)
            convert[0, 0, 0, 0] = 65.738
            convert[0, 1, 0, 0] = 129.057
            convert[0, 2, 0, 0] = 25.064
            diff.mul_(convert).div_(256)
            diff = diff.sum(dim=1, keepdim=True)
    else:
        shave = scale + 6
    import math
    shave = math.ceil(shave)
    valid = diff[:, :, shave:-shave, shave:-shave]
    mse = valid.pow(2).mean()

    return -10 * math.log10(mse)


def calc_ssim(img1, img2, scale=2, benchmark=False):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if benchmark:
        border = math.ceil(scale)
    else:
        border = math.ceil(scale) + 6

    img1 = img1.data.squeeze().float().clamp(0, 255).round().cpu().numpy()
    img1 = np.transpose(img1, (1, 2, 0))
    img2 = img2.data.squeeze().cpu().numpy()
    img2 = np.transpose(img2, (1, 2, 0))

    img1_y = np.dot(img1, [65.738, 129.057, 25.064]) / 255.0 + 16.0
    img2_y = np.dot(img2, [65.738, 129.057, 25.064]) / 255.0 + 16.0
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1_y = img1_y[border:h - border, border:w - border]
    img2_y = img2_y[border:h - border, border:w - border]

    if img1_y.ndim == 2:
        # print('aaaaaaaaaaaaaaa')
        return ssim(img1_y, img2_y)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def make_optimizer(args, my_model):

    """lambda x: x.requires_grad是个函数， my_model.parameters()是函数输入的参数
    requires_grad为false时候是为了固定网络的底层，这样在反向过程中就���会计算这些参数对应的梯度
    """


    trainable = filter(lambda x: x.requires_grad, my_model.parameters())

    if args.optimizer == 'SGD':
        optimizer_function = optim.SGD
        kwargs = {'momentum': args.momentum}
    elif args.optimizer == 'ADAM':
        optimizer_function = optim.Adam
        kwargs = {
            'betas': (args.beta1, args.beta2),
            'eps': args.epsilon
        }
    elif args.optimizer == 'RMSprop':
        optimizer_function = optim.RMSprop
        kwargs = {'eps': args.epsilon}

    kwargs['weight_decay'] = args.weight_decay

    return optimizer_function(trainable, **kwargs)


def make_scheduler(args, my_optimizer):
    """学习速率衰减类型"""
    if args.decay_type == 'step':
        scheduler = lrs.StepLR(
            my_optimizer,
            step_size=args.lr_decay_sr,
            gamma=args.gamma_sr,
        )
    elif args.decay_type.find('step') >= 0:
        milestones = args.decay_type.split('_')
        milestones.pop(0)
        milestones = list(map(lambda x: int(x), milestones))
        scheduler = lrs.MultiStepLR(
            my_optimizer,
            milestones=milestones,
            gamma=args.gamma
        )

    scheduler.step(args.start_epoch - 1)

    return scheduler

def my_optimizer(args, model):
    # if args.n_GPUs > 1:
    #     cmod_params = model.model.module.CMod.parameters()
    # else:
    #     cmod_params = model.model.CMod.parameters()

    cmod_params = list(model.get_model().CMod.parameters())

    base_params = list(map(id, cmod_params))
    logits_params = filter(lambda p: (id(p) not in base_params) and (p.requires_grad), model.parameters())
    params = [
        {"params": cmod_params, 'betas': (args.beta1, args.beta2), 'eps': args.epsilon},
        {"params": logits_params,'betas': (args.beta1, args.beta2), 'eps': args.epsilon}
    ]
    return optim.Adam(params)

def my_scheduler(args, my_optimizer, decay_type):
    """学习速率衰减类型"""
    if decay_type == 'step':
        scheduler = lrs.StepLR(
            my_optimizer,
            step_size=args.lr_decay_sr,
            gamma=args.gamma_sr,
        )
    elif decay_type.find('step') >= 0:
        milestones = decay_type.split('_')
        milestones.pop(0)
        milestones = list(map(lambda x: int(x), milestones))
        scheduler = lrs.MultiStepLR(
            my_optimizer,
            milestones=milestones,
        )

    scheduler.step(args.start_epoch - 1)

    return scheduler
