"""
预警性能评估模块

对检测到的预警点进行性能评估，计算：
- FAR (False Alarm Rate): 误报率 = 误报数 / 总预警数
- ALT (Average Lead Time): 平均提前期 = 最早真实预警到疫情峰的平均提前天数
- EWR (Early Warning Rate): 提前成功率 = 成功预警的疫情数 / 总疫情数

评估逻辑基于 Matlab 版本的预警性能评估算法：
  1. 对每个预警点，检查其后 eval_window 窗口内 ILI 是否出现显著增长
  2. 增长判定条件：Ymax > Y_thr AND rmax > r_c AND deltaY > delta
  3. 统计误报/真报，计算 FAR、ALT、EWR
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
import warnings


class WarningEvaluator:
    """
    预警性能评估器

    根据真实疫情暴发点（ground truth），对预警系统的检测结果进行量化评估。

    Parameters 与 MATLAB 版本对照:
       Y_thr    — ILI 阈值，窗口内最大值超过此值才认为可能是疫情
       r_c      — 增长率阈值，窗口内最大增长速率超过此值才确认
       delta    — 最小增幅阈值，窗口内最大增长对应的绝对增量超过此值才确认
       eval_window — 评估窗口大小（预警点后检测多少天）
       max_lead    — 最大提前期（仅在此范围内的预警视为有效）
    """

    def __init__(self,
                 ili_series: np.ndarray,
                 outbreak_points: List[int],
                 eval_window: int = 60,
                 Y_thr: Optional[float] = None,
                 r_c: float = 0.02,
                 delta: float = 10.0,
                 max_lead: int = 60):
        """
        初始化评估器

        Args:
            ili_series: ILI 原始数据序列
            outbreak_points: 真实疫情暴发峰值点索引列表（ground truth）
            eval_window: 评估窗口大小（预警点后检测窗口，单位：时间步），默认 60
            Y_thr: ILI 阈值。None 时自动取 ILI 的 75% 分位数
            r_c: 增长率阈值，默认 0.02
            delta: 最小增幅阈值，默认 10.0
            max_lead: 最大提前期（单位：时间步），默认 60
        """
        self.ili_series = np.asarray(ili_series, dtype=float)
        self.outbreak_points = sorted(outbreak_points)
        self.eval_window = int(eval_window)
        self.Y_thr = Y_thr if Y_thr is not None else float(np.percentile(self.ili_series, 75))
        self.r_c = float(r_c)
        self.delta = float(delta)
        self.max_lead = int(max_lead)

        if len(self.ili_series) == 0:
            raise ValueError("ILI series is empty")
        if len(self.outbreak_points) == 0:
            warnings.warn("No outbreak points provided — ALT/EWR will be NaN")

    # ============================================================
    # 核心方法
    # ============================================================

    def evaluate(self, anomaly_points: List[int]) -> Dict:
        """
        对预警点进行完整性能评估

        Args:
            anomaly_points: 模型检测到的预警点索引列表（支持 Python list 或 numpy array）

        Returns:
            Dict: 包含以下评估指标
                - total_alarms:      总预警数
                - true_alarms:       真报数
                - false_alarms:      误报数
                - unevaluated:       无法评估的预警数（窗口越界）
                - FAR:               误报率 = false / (true + false)
                - ALT:               平均提前期（天），无有效提前期时为 NaN
                - EWR:               提前成功率 = detected / N_outbreak
                - detected_outbreaks:成功预警的疫情数
                - lead_times:        各成功预警的提前期列表
                - false_alarm_indices: 误报预警点索引
                - true_alarm_indices:  真报预警点索引
                - unevaluated_indices: 越界未评估的预警点索引
                - outbreak_points:   真实疫情暴发峰索引
                - params:            使用的评估参数副本
        """
        anomaly_points = np.asarray(anomaly_points, dtype=int)
        # 去重并排序
        anomaly_points = np.sort(np.unique(anomaly_points))

        N = len(self.ili_series)
        false_alarms = []      # 误报预警点索引
        true_alarms = []       # 真报预警点索引
        unevaluated = []       # 越界未评估的预警点索引

        # ============================================================
        # Step 1: 逐预警点评估 — 是否真实触发疫情信号
        # ============================================================
        for t_star in anomaly_points:
            # ---- 越界检查 ----
            if t_star + self.eval_window > N:
                unevaluated.append(int(t_star))
                continue

            # 取评估窗口：预警点之后 eval_window 天的 ILI 数据
            window = self.ili_series[t_star + 1: t_star + self.eval_window]

            # ---- 增长检测（双循环找最大增长速率） ----
            rmax = -np.inf
            deltaY = 0.0

            for t1 in range(len(window) - 1):
                for t2 in range(t1 + 1, len(window)):
                    if window[t2] > window[t1]:
                        # 分母设下限 5，防止 log(0)
                        denominator = max(window[t1], 5.0)
                        r = np.log(window[t2] / denominator) / (t2 - t1)
                        if r > rmax:
                            rmax = r
                            deltaY = window[t2] - window[t1]

            Ymax = np.max(window)

            # ---- 疫情信号判定 ----
            epidemic_signal = (Ymax > self.Y_thr) and (rmax > self.r_c) and (deltaY > self.delta)

            if not epidemic_signal:
                false_alarms.append(int(t_star))
            else:
                true_alarms.append(int(t_star))

        # ============================================================
        # Step 2: FAR — 误报率
        # ============================================================
        N_alarm = len(true_alarms) + len(false_alarms)
        N_false = len(false_alarms)
        FAR = N_false / N_alarm if N_alarm > 0 else 0.0

        # ============================================================
        # Step 3: ALT — 平均提前期  &  EWR — 提前成功率
        # ============================================================
        N_outbreak = len(self.outbreak_points)
        lead_times = []
        detected_outbreaks = 0

        for t_out in self.outbreak_points:
            # 找在该疫情暴发之前、且不超过 max_lead 的真报预警
            valid_alarms = [
                ta for ta in true_alarms
                if ta < t_out and (t_out - ta) <= self.max_lead
            ]
            if valid_alarms:
                detected_outbreaks += 1
                t_first = min(valid_alarms)
                lead_times.append(t_out - t_first)

        ALT = float(np.mean(lead_times)) if len(lead_times) > 0 else float('nan')
        EWR = detected_outbreaks / N_outbreak if N_outbreak > 0 else 0.0

        # ============================================================
        # 返回结构化结果
        # ============================================================
        return {
            # 计数
            "total_alarms": N_alarm,
            "true_alarms": len(true_alarms),
            "false_alarms": N_false,
            "unevaluated": len(unevaluated),
            "detected_outbreaks": detected_outbreaks,
            "total_outbreaks": N_outbreak,

            # 指标
            "FAR": FAR,
            "ALT": ALT,
            "EWR": EWR,

            # 详细数据
            "lead_times": lead_times,
            "false_alarm_indices": false_alarms,
            "true_alarm_indices": true_alarms,
            "unevaluated_indices": unevaluated,
            "outbreak_points": self.outbreak_points,

            # 参数
            "params": {
                "eval_window": self.eval_window,
                "Y_thr": self.Y_thr,
                "r_c": self.r_c,
                "delta": self.delta,
                "max_lead": self.max_lead,
            },
        }

    # ============================================================
    # 便捷方法
    # ============================================================

    def evaluate_and_print(self, anomaly_points: List[int],
                           label: str = "评估结果") -> Dict:
        """
        评估并打印格式化结果

        Args:
            anomaly_points: 预警点索引列表
            label: 打印时的标题标签

        Returns:
            Dict: 评估结果（同 evaluate）
        """
        result = self.evaluate(anomaly_points)
        self.print_summary(result, label)
        return result

    @staticmethod
    def print_summary(result: Dict, label: str = "预警评估结果") -> None:
        """
        打印评估结果摘要

        Args:
            result: evaluate() 返回的结果字典
            label: 标题
        """
        print(f"\n{'=' * 50}")
        print(f"  {label}")
        print(f"{'=' * 50}")
        print(f"  总预警数:             {result['total_alarms']}")
        print(f"  真报数:               {result['true_alarms']}")
        print(f"  误报数:               {result['false_alarms']}")
        print(f"  越界未评估:           {result['unevaluated']}")
        print(f"  ——")
        print(f"  误报率 FAR:           {result['FAR']:.3f}")
        print(f"  平均提前期 ALT:       {result['ALT']:.2f} 天" if not np.isnan(result['ALT'])
              else "  平均提前期 ALT:       NaN")
        print(f"  提前成功率 EWR:       {result['EWR']:.3f}")
        print(f"  成功预警疫情数:       {result['detected_outbreaks']}/{result['total_outbreaks']}")
        if result['lead_times']:
            print(f"  各提前期:             {result['lead_times']}")
        if result['false_alarm_indices']:
            print(f"  误报索引:             {result['false_alarm_indices']}")
        if result['true_alarm_indices']:
            print(f"  真报警索引:           {result['true_alarm_indices']}")
        if result['unevaluated_indices']:
            print(f"  越界未评估索引:       {result['unevaluated_indices']}")
        print(f"{'=' * 50}\n")


# ============================================================
# 便捷函数
# ============================================================

def evaluate_alarms(ili_series: np.ndarray,
                    anomaly_points: List[int],
                    outbreak_points: List[int],
                    eval_window: int = 60,
                    Y_thr: Optional[float] = None,
                    r_c: float = 0.02,
                    delta: float = 10.0,
                    max_lead: int = 60,
                    verbose: bool = True) -> Dict:
    """
    快捷函数：创建评估器、执行评估、可选打印结果

    Args:
        ili_series: ILI 原始数据
        anomaly_points: 预警点索引列表
        outbreak_points: 真实疫情暴发索引列表
        eval_window: 评估窗口大小
        Y_thr: ILI 阈值
        r_c: 增长率阈值
        delta: 最小增幅阈值
        max_lead: 最大提前期
        verbose: 是否打印结果

    Returns:
        Dict: 评估结果
    """
    evaluator = WarningEvaluator(
        ili_series=ili_series,
        outbreak_points=outbreak_points,
        eval_window=eval_window,
        Y_thr=Y_thr,
        r_c=r_c,
        delta=delta,
        max_lead=max_lead,
    )

    if verbose:
        return evaluator.evaluate_and_print(anomaly_points)
    else:
        return evaluator.evaluate(anomaly_points)


# ============================================================
# 独立运行示例
# ============================================================

if __name__ == "__main__":
    # 简单演示：使用正弦波 + 噪声模拟 ILI 数据
    np.random.seed(42)
    t = np.arange(500)
    ili = 50 + 30 * np.sin(2 * np.pi * t / 365) + np.random.randn(500) * 5
    ili[ili < 0] = 0

    # 模拟 3 个真实疫情峰
    outbreak_pts = [70, 200, 365]

    # 模拟一些预警点
    alarm_pts = sorted([50, 65, 80, 185, 195, 300, 350, 400])

    result = evaluate_alarms(
        ili_series=ili,
        anomaly_points=alarm_pts,
        outbreak_points=outbreak_pts,
        verbose=True
    )
