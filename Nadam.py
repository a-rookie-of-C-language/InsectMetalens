import os
# 随机数种子
# np.random.seed(2024)
from collections import namedtuple
from datetime import datetime

import cupy as np
import pandas as pd
from tqdm import tqdm

from FresnelDiffraction import FresnelDiffraction

# 定义一个命名元组来存储损失权重，提高代码可读性
LossWeights = namedtuple('LossWeights',
                         ['fwhm', 'sidelobe_ratio', 'peak_intensity', 'focal_offset', 'DOF', 'intensity_sum'])


class NAG:
    def __init__(self, learning_rate=0.05, momentum=0.9, max_iter=1000, tol=1e-6, verbose=False, metalens=None):
        self.learning_rate = learning_rate  # 学习率
        self.momentum = momentum  # 动量参数
        self.max_iter = max_iter  # 最大迭代次数
        self.tol = tol  # 收敛阈值
        self.verbose = verbose  # 是否打印迭代信息
        # 获取透镜参数
        self.metalens = metalens
        self.phases = self.metalens.phases
        self.asm = FresnelDiffraction(self.metalens)
        self.target_params = {
            "TargetFWHM": 1.5,
            "TargetSideLobeRatio": 0.05,
            "TargetPeakIntensity": 2000,
            "TargetFocalOffset": 0.00001
        }
        self.total_loss = 0
        self.file_name = datetime.now().strftime("result/data_%Y%m%d_%H%M")
        os.makedirs(self.file_name, exist_ok=True)
        self.file_exists = os.path.exists(self.file_name)
        # 初始化动量
        self.v = np.zeros_like(self.phases)

    def fit(self):
        iteration = 0
        gradient_norm = np.inf
        params = self.phases
        prev_loss = np.inf
        tol = self.tol
        p_bar = tqdm(total=self.max_iter)

        while iteration < self.max_iter:
            # 预测性更新参数
            params_pred = params - self.momentum * self.v
            print("momentum ", self.momentum, end="\n")
            # 随机选择样本
            indices = np.random.choice(len(params), size=len(params), replace=False)
            # 计算梯度
            gradient = self.compute_gradient(params_pred, indices)
            if gradient.all() == 0:
                gradient = 1
            print("gradient", gradient, end="\n")
            # 更新动量
            self.v = self.momentum * self.v + self.learning_rate * gradient
            print("v", self.v, end="\n")
            # 更新参数
            params -= self.v

            print("phases:", params)
            # 计算损失函数
            cur_params, loss = self.loss_function(params)
            self.save_results(cur_params)
            print('cue_params：', cur_params)
            # 判断收敛性
            # if abs(loss - prev_loss) < tol:
            # break

            #            prev_loss = loss

            p_bar.set_description(f"第 {iteration + 1} 次迭代, loss: {self.total_loss}")
            p_bar.update(1)
            iteration += 1
            # metalens = MetalensOptimization()
            # metalens.init_phases()
            # self.phases = metalens.phases
            # 更新学习率s
            self.learning_rate, _, _ = self.update_learning_rate(loss, self.learning_rate)

    # def compute_gradient(self, params, indices):
    #     """
    #     计算梯度
    #     :param params:
    #     :param indices:
    #     :return:
    #     """
    #     # 生成[0,pi]的随机数
    #     epsilon = 1e-3 * np.random.uniform(0, np.pi)  # 微小扰动的大小
    #     gradient = np.zeros_like(params)  # 初始化梯度为零向量
    #     # 计算每个相位的梯度
    #     for i in indices:
    #         # 对第i个参数进行微小扰动
    #         params_plus = params.copy()
    #         params_plus[i] += epsilon * 1e2
    #         # 计算扰动后的损失函数值
    #         _, loss_plus = self.loss_function(params_plus)
    #
    #         # 对第i个参数进行微小扰动
    #         params_minus = params.copy()
    #         params_minus[i] -= epsilon
    #         # 计算扰动后的损失函数值
    #         _, loss_minus = self.loss_function(params_minus)
    #         # 计算第i个参数的梯度
    #         gradient[i] = (loss_plus - loss_minus) / (2 * epsilon)
    #     return gradient
    def compute_gradient(self, params, indices):
        # 生成[0,pi]的随机数，注意这里的范围应该是(-pi, pi)，因为扰动是加减epsilon
        epsilon = 1e-3 * np.random.uniform(-np.pi, np.pi)

        # 计算正扰动和负扰动的参数
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[indices] += epsilon
        params_minus[indices] -= epsilon

        # 计算正扰动和负扰动的损失函数
        _, loss_plus = self.loss_function(params_plus)
        _, loss_minus = self.loss_function(params_minus)

        # 计算梯度
        gradient = np.zeros_like(params)
        gradient[indices] = (loss_plus - loss_minus) / (2 * epsilon)

        return gradient


    def loss_function(self, phases):
        # 当前的评价参数
        current_params = self.asm.compute_all(phases)

        # 提取并计算差异
        # abs
        # diff_fwhm = np.max(abs(current_params["FWHM"])) - self.target_params["TargetFWHM"]
        # diff_sidelobe_ratio = np.max(current_params["side_lobe_ratio"]) - self.target_params["TargetSideLobeRatio"]
        # diff_peak_intensity = np.min(current_params["intensity_peak"]) - self.target_params["TargetPeakIntensity"]
        # diff_focal_offset = np.max(current_params["focal_offset"]) - self.target_params["TargetFocalOffset"]
        # DOF_loss = np.std(current_params["DOF"])
        # intensity_sum_loss = 10 ** 4 / np.min(current_params["intensity_sum"])
        #
        TargetFWHM = self.target_params["TargetFWHM"]
        TargetSideLobeRatio = self.target_params["TargetSideLobeRatio"]
        TargetPeakIntensity = self.target_params["TargetPeakIntensity"]
        TargetFocalOffset = self.target_params["TargetFocalOffset"]
        TargetIntensitySum = 10 ** 4  # 假设这是目标强度和的目标值

        # 计算每个参数的差值
        diff_fwhm = np.max(abs(current_params["FWHM"])) - TargetFWHM
        diff_sidelobe_ratio = np.max(abs(current_params["side_lobe_ratio"])) - TargetSideLobeRatio
        diff_peak_intensity = np.min(abs(current_params["intensity_peak"])) - TargetPeakIntensity
        diff_focal_offset = np.max(abs(current_params["focal_offset"])) - TargetFocalOffset
        DOF_loss = np.std(current_params["DOF"])
        intensity_sum_loss = 10 ** 4 / np.min(current_params["intensity_sum"])

        # 将差值转换为百分比
        percent_diff_fwhm = (diff_fwhm / TargetFWHM) * 100 if TargetFWHM != 0 else 0
        percent_diff_sidelobe_ratio = (
                                              diff_sidelobe_ratio / TargetSideLobeRatio) * 100 if TargetSideLobeRatio != 0 else 0
        percent_diff_peak_intensity = (
                                              diff_peak_intensity / TargetPeakIntensity) * 100 if TargetPeakIntensity != 0 else 0
        percent_diff_focal_offset = (diff_focal_offset / TargetFocalOffset) * 100 if TargetFocalOffset != 0 else 0
        percent_DOF_loss = (DOF_loss / TargetFWHM) * 100 if TargetFWHM != 0 else 0  # 假设DOF的目标值与FWHM相同
        percent_intensity_sum_loss = (
                (TargetIntensitySum / np.min(current_params["intensity_sum"]) - 1) * 100) if np.min(
            current_params["intensity_sum"]) != 0 else 0
        # 定义损失权重
        # loss_weights = LossWeights(fwhm=10, sidelobe_ratio=0.1, peak_intensity=10, focal_offset=2, DOF=0.01,intensity_sum = 1)

        # 计算损失项
        # losses = {
        #     'fwhm_loss': diff_fwhm ** 2,
        #     'sidelobe_ratio_loss': diff_sidelobe_ratio ** 2,
        #     'peak_intensity_loss': diff_peak_intensity ** 2,
        #     'focal_offset_loss': diff_focal_offset ** 2,
        #     'DOF_loss': DOF_loss,
        #     'intensity_sum_loss': intensity_sum_loss
        # }
        # losses_without_suffix = {key.replace('_loss', ''): value for key, value in losses.items()}
        #
        # total_loss = sum(loss_weights._asdict()[key] * losses_without_suffix[key] for key in losses_without_suffix)
        #
        # self.total_loss = total_loss
        #
        # return current_params, total_loss
        loss_weights = LossWeights(fwhm=10, sidelobe_ratio=0.1, peak_intensity=10, focal_offset=2, DOF=0.01,
                                   intensity_sum=1)

        # 计算损失项
        losses = {
            'fwhm_loss': percent_diff_fwhm ** 2 * loss_weights.fwhm,
            'sidelobe_ratio_loss': percent_diff_sidelobe_ratio ** 2 * loss_weights.sidelobe_ratio,
            'peak_intensity_loss': percent_diff_peak_intensity ** 2 * loss_weights.peak_intensity,
            'focal_offset_loss': percent_diff_focal_offset ** 2 * loss_weights.focal_offset,
            'DOF_loss': percent_DOF_loss * loss_weights.DOF,
            'intensity_sum_loss': percent_intensity_sum_loss * loss_weights.intensity_sum
        }

        total_loss = sum(losses.values())

        self.total_loss = total_loss
        return current_params, total_loss

    def save_results(self, cur_params, code=0):

        """
        保存结果
        :param self:
        :param cur_params:保存的参数
        :param code:
        :return:
        """

        #  只需输出最优值
        # 一次优化后，保存当前的参数
        df = pd.DataFrame({key: np.asnumpy(value) for key, value in cur_params.items()})

        print("df:", df)
        if not self.file_exists:
            df.to_csv(self.file_name+"param.csv", mode='w', header=True, index=False)
            self.file_exists = True
        else:
            df.to_csv(self.file_name+"param.csv", mode='a', header=False, index=False)
        print(f"数据已追加保存到 {self.file_name}param.csv")

    def update_learning_rate(self, loss, learning_rate=0.01, decay_factor=0.1, min_learning_rate=1e-6,
                             patience=0,
                             epoch=0,
                             best_loss=float('inf')):
        """
          更新学习率
          :param loss:
          :param learning_rate:
          :param decay_factor:
          :param min_learning_rate:
          :param patience:
          :param epoch:
          :param best_loss:
          :return:
          """
        if loss < best_loss:
            best_loss = loss
            patience = 0
        else:
            patience += 1
            decay_patience = 10  # 衰减耐心的阈值
            if patience > decay_patience:
                learning_rate *= decay_factor
                learning_rate = max(learning_rate, min_learning_rate)
                patience = 0

        return learning_rate, patience, best_loss
