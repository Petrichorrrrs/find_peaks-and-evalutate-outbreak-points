"""
预警性能评估独立模块

对齐其他模块（CNEMonitor / tRCEMonitor / FusionWarning）的标准接口：
  __init__(数据, 参数) → run() → 返回结构化结果 dict

功能：
  1. 自动检测 ILI 暴发峰值，支持用户确认 / 手动修正
  2. 对多个预警源分别评估（FAR / ALT / EWR）
  3. 绘制评估结果图（真报 / 误报 / 未评估 区分着色）

用法示例：
    from evaluate_alerts import AlertEvaluator

    evaluator = AlertEvaluator(
        ili_raw=data["ili_raw"],
        date_list=data["date_list"],
        data_path=args.data_path,
    )
    evaluator.add_source("CNE", [50, 65, ...])
    evaluator.add_source("tRCE", [48, 62, ...])
    evaluator.add_source("Fusion(严重)", [55, 70, ...])

    results = evaluator.run()        # 交互式峰值确认 + 评估
    evaluator.print_comparison()     # 打印对比表
    evaluator.plot_results(save_path="eval_result.png")  # 结果图
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any
from evaluation import WarningEvaluator
from find_peaks import detect_ili_peaks

# ======================================
# matplotlib 中文字体配置
# ======================================
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


class AlertEvaluator:
    """
    多源预警评估器

    标准接口：__init__() → add_source() → run() → 结构化结果

    Attributes:
        ili_raw: ILI 原始病例数据
        date_list: 日期字符串列表
        data_path: Excel 数据文件路径（用于峰值检测）
        alert_sources: 预警源字典 {名称: 预警点索引列表}
        outbreak_points: 最终确认的暴发峰值索引（ground truth）
        results: 评估结果字典 {名称: evaluate() 返回的 dict}
    """

    def __init__(
        self,
        ili_raw: np.ndarray,
        date_list: List[str],
        data_path: Optional[str] = None,
        eval_window: int = 60,
        r_c: float = 0.02,
        delta: float = 10.0,
        max_lead: int = 60,
    ):
        """
        Args:
            ili_raw: ILI 原始病例数据（1D array）
            date_list: 与 ili_raw 等长的日期字符串列表
            data_path: Excel 数据文件路径（传给 detect_ili_peaks）
            eval_window: 评估窗口大小，默认 60
            r_c: 增长率阈值，默认 0.02
            delta: 最小增幅阈值，默认 10.0
            max_lead: 最大提前期（天），默认 60
        """
        self.ili_raw = np.asarray(ili_raw, dtype=float)
        self.date_list = date_list
        self.data_path = data_path
        self.eval_window = eval_window
        self.r_c = r_c
        self.delta = delta
        self.max_lead = max_lead

        self.alert_sources: Dict[str, List[int]] = {}
        self.outbreak_points: List[int] = []
        self.results: Dict[str, dict] = {}

    # ---------------------------------------------------------------
    # 预警源管理
    # ---------------------------------------------------------------

    def add_source(self, name: str, alert_indices: List[int]) -> None:
        """添加一个预警源及其预警点索引"""
        self.alert_sources[name] = sorted(set(int(i) for i in alert_indices))

    @property
    def source_count(self) -> int:
        """已注册的预警源数量"""
        return len(self.alert_sources)

    # ---------------------------------------------------------------
    # 核心流程 run()
    # ---------------------------------------------------------------

    def run(self, skip_confirm: bool = False) -> Dict[str, Any]:
        """
        执行完整评估流程（标准接口，对标 CNEMonitor.run / tRCEMonitor.run）

        步骤：
          1. 自动检测 ILI 暴发峰
          2. 展示图表 + 文字 → 用户确认 / 手动修正（skip_confirm=True 时跳过）
          3. 对每个已注册的预警源执行 WarningEvaluator 评估
          4. 返回结构化结果 dict

        Args:
            skip_confirm: True 时跳过交互确认，直接使用自动检测的峰值

        Returns:
            results: 与 _build_results() 相同，包含 outbreak_points 和各源评估指标
        """
        # ---- Step 1 & 2: 峰值检测 + 用户确认 ----
        if skip_confirm:
            # 跳过交互：自动检测 + 直接使用
            from find_peaks import detect_ili_peaks
            peaks, _ = detect_ili_peaks(self.data_path)
            self.outbreak_points = peaks
            print(f"   (跳过确认) 自动检测到 {len(peaks)} 个暴发峰: {peaks}")
        else:
            self._detect_and_confirm_peaks()

        # ---- Step 3: 评估 ----
        if not self.outbreak_points:
            raise ValueError("暴发峰值为空，请确认峰值检测结果")

        self.results = {}
        for name, indices in self.alert_sources.items():
            if not indices:
                continue
            evaluator = WarningEvaluator(
                ili_series=self.ili_raw,
                outbreak_points=self.outbreak_points,
                eval_window=self.eval_window,
                r_c=self.r_c,
                delta=self.delta,
                max_lead=self.max_lead,
            )
            self.results[name] = evaluator.evaluate(indices)

        return self._build_results()

    def _detect_and_confirm_peaks(self) -> None:
        """
        内部：自动检测暴发峰 → 图表展示 → 用户确认 / 手动修正
        """
        if self.data_path is None:
            raise ValueError("data_path 未设置，无法检测暴发峰")

        peaks, _ = detect_ili_peaks(self.data_path)

        # ---- 文字输出 ----
        print("\n" + "=" * 60)
        print("  【ILI 暴发峰值检测结果】")
        print("=" * 60)
        for i, p in enumerate(peaks):
            date_str = self.date_list[p] if p < len(self.date_list) else "N/A"
            print(f"    第{i + 1:>2d}次: 索引 {p:>4d}, 日期 {date_str},  ILI = {self.ili_raw[p]:.1f}")
        print("=" * 60)

        # ---- 图表展示 ----
        fig, ax = plt.subplots(figsize=(14, 5))
        x = np.arange(len(self.ili_raw))

        ax.plot(x, self.ili_raw, color="black", linewidth=1.5, label="ILI 病例数")
        for i, p in enumerate(peaks):
            if p >= len(self.ili_raw):
                continue
            ax.axvline(x=p, color="#d62728", linestyle="--", alpha=0.5, linewidth=1)
            ax.scatter(p, self.ili_raw[p], color="#d62728", s=180,
                       marker="v", zorder=5, label="检测峰" if i == 0 else "")
            ax.annotate(
                f"#{i + 1}\n{self.date_list[p] if p < len(self.date_list) else ''}",
                (p, self.ili_raw[p]),
                textcoords="offset points", xytext=(0, 14),
                ha="center", fontsize=9, color="#d62728", fontweight="bold",
            )

        ax.set_title("ILI 暴发峰值确认 — 是否使用以上峰值作为评估依据？",
                      fontsize=13, fontweight="bold")
        ax.set_ylabel("ILI 病例数", fontsize=11)
        ax.set_xlabel("时间索引", fontsize=11)

        n = len(self.date_list)
        step = max(1, n // 10)
        tick_pos = list(range(0, n, step))
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(
            [self.date_list[i] if i < n else "" for i in tick_pos],
            rotation=30, ha="right", fontsize=8,
        )
        ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.25)
        ax.set_xlim(0, n - 1)
        plt.tight_layout()
        plt.show()

        # ---- 用户确认 ----
        answer = input("  是否使用以上峰值作为评估依据？(y/n, 默认 y): ").strip().lower()
        if answer in ("y", "yes", ""):
            self.outbreak_points = peaks
        else:
            manual = input("  请手动输入峰值索引（逗号分隔，例如 70,200,365）: ").strip()
            if manual:
                try:
                    pts = [int(x.strip()) for x in manual.split(",") if x.strip()]
                    self.outbreak_points = [p for p in pts if 0 <= p < len(self.ili_raw)]
                    if not self.outbreak_points:
                        print("  输入索引均越界，使用自动检测结果")
                        self.outbreak_points = peaks
                except ValueError:
                    print("  输入格式错误，使用自动检测结果")
                    self.outbreak_points = peaks
            else:
                self.outbreak_points = peaks

        print(f"  最终使用 {len(self.outbreak_points)} 个暴发峰: {self.outbreak_points}")

    # ---------------------------------------------------------------
    # 结果构建
    # ---------------------------------------------------------------

    def _build_results(self) -> Dict[str, Any]:
        """构建 run() 返回的结构化结果 dict"""
        out = {
            "summary": {
                "n_outbreaks": len(self.outbreak_points),
                "n_sources": len(self.results),
                "outbreak_points": self.outbreak_points,
            },
            "sources": {},
        }
        for name, r in self.results.items():
            out["sources"][name] = {
                "total_alarms":         r["total_alarms"],
                "true_alarms":          r["true_alarms"],
                "false_alarms":         r["false_alarms"],
                "unevaluated":          r["unevaluated"],
                "detected_outbreaks":   r["detected_outbreaks"],
                "total_outbreaks":      r["total_outbreaks"],
                "FAR":                  r["FAR"],
                "ALT":                  r["ALT"],
                "EWR":                  r["EWR"],
                "lead_times":           r["lead_times"],
                "true_alarm_indices":   r["true_alarm_indices"],
                "false_alarm_indices":  r["false_alarm_indices"],
                "unevaluated_indices":  r["unevaluated_indices"],
            }
        return out

    def get_plot_data(self) -> Dict[str, Any]:
        """
        获取绘图数据（供 visualization 模块使用）

        Returns:
            dict: 包含 dates / ili_values / outbreak_points / eval_results
        """
        eval_plot = {}
        for name, r in self.results.items():
            eval_plot[name] = {
                "true_alarm_indices":   r["true_alarm_indices"],
                "false_alarm_indices":  r["false_alarm_indices"],
                "unevaluated_indices":  r["unevaluated_indices"],
                "FAR":                  r["FAR"],
                "ALT":                  r["ALT"],
                "EWR":                  r["EWR"],
            }
        return {
            "dates": self.date_list,
            "ili_values": self.ili_raw.tolist(),
            "outbreak_points": self.outbreak_points,
            "eval_results": eval_plot,
        }

    def get_results(self) -> Dict[str, Any]:
        """获取结构化评估结果（_build_results 的别名）"""
        return self._build_results()

    # ---------------------------------------------------------------
    # 结果输出
    # ---------------------------------------------------------------

    def print_comparison(self) -> None:
        """打印多源对比汇总表及各源详细信息"""
        if not self.results:
            print("  （无评估结果）")
            return

        # ---- 汇总表 ----
        print(f"\n   {'预警源':<20s} {'总数':>5s} {'真报':>5s} {'误报':>5s} "
              f"{'FAR':>7s} {'ALT':>7s} {'EWR':>7s} {'检出/总疫情':>12s}")
        print(f"   {'-' * 70}")
        for name, r in self.results.items():
            far_s = f"{r['FAR']:.1%}" if r["total_alarms"] > 0 else "-"
            alt_s = f"{r['ALT']:.1f}d" if not np.isnan(r["ALT"]) else "-"
            ewr_s = f"{r['EWR']:.0%}" if r["total_outbreaks"] > 0 else "-"
            det_s = f"{r['detected_outbreaks']}/{r['total_outbreaks']}"
            print(f"   {name:<20s} {r['total_alarms']:>5d} {r['true_alarms']:>5d} "
                  f"{r['false_alarms']:>5d} {far_s:>7s} {alt_s:>7s} "
                  f"{ewr_s:>7s} {det_s:>12s}")

        # ---- 各源详情 ----
        for name, r in self.results.items():
            print(f"\n   ═══ {name} ═══")
            print(f"      总预警: {r['total_alarms']} | 真报: {r['true_alarms']} | "
                  f"误报: {r['false_alarms']} | 越界: {r['unevaluated']}")
            print(f"      FAR = {r['FAR']:.1%} | ALT = {r['ALT']:.1f}d | "
                  f"EWR = {r['EWR']:.0%} | 检出疫情: "
                  f"{r['detected_outbreaks']}/{r['total_outbreaks']}")
            if r["lead_times"]:
                print(f"      提前期: {r['lead_times']}")
            if r["true_alarm_indices"]:
                display = r["true_alarm_indices"][:20]
                suffix = " ..." if len(r["true_alarm_indices"]) > 20 else ""
                print(f"      真报索引: {display}{suffix}")

    # ---------------------------------------------------------------
    # 可视化
    # ---------------------------------------------------------------

    def plot_results(self, save_path: Optional[str] = None) -> None:
        """
        绘制评估结果图

        每个预警源占一个子图，区分：
          - ● 绿色圆形  = 真报（成功预警）
          - ✗ 红色叉形  = 误报
          - ◇ 灰色菱形  = 未评估（窗口越界）
          - ▼ 红色三角  = ILI 暴发峰（Ground Truth）

        Args:
            save_path: 图片保存路径（None 则只显示不保存）
        """
        if not self.results:
            print("  无评估结果可绘制")
            return

        n_sources = len(self.results)
        fig, axes = plt.subplots(
            n_sources + 1, 1,
            figsize=(14, 3 * (n_sources + 1)),
            sharex=True,
            constrained_layout=True,
        )

        # 兼容单源情况
        if n_sources == 1:
            axes_list = [axes[0], axes[1]]
        else:
            axes_list = list(axes)

        x = np.arange(len(self.ili_raw))

        # === 顶图：ILI 曲线 + 暴发峰 ===
        ax0 = axes_list[0]
        ax0.plot(x, self.ili_raw, color="black", linewidth=1.5, label="ILI 病例数")
        for i, p in enumerate(self.outbreak_points):
            if p >= len(self.ili_raw):
                continue
            label = "暴发峰 (Ground Truth)" if i == 0 else ""
            ax0.axvline(x=p, color="#d62728", linestyle="--", alpha=0.5, linewidth=1)
            ax0.scatter(p, self.ili_raw[p], color="#d62728", s=180,
                        marker="v", zorder=5, label=label)
            ax0.annotate(
                f"#{i + 1}", (p, self.ili_raw[p]),
                textcoords="offset points", xytext=(0, 12),
                ha="center", fontsize=10, color="#d62728", fontweight="bold",
            )
        ax0.set_ylabel("ILI 病例数", fontsize=11)
        ax0.set_title("ILI 病例趋势与暴发峰值 — 预警评估", fontsize=14, fontweight="bold")
        ax0.legend(loc="upper left", fontsize=9, framealpha=0.9)
        ax0.grid(True, alpha=0.25)
        ax0.set_xlim(0, len(self.ili_raw) - 1)

        # === 各源子图 ===
        style = {
            "true":        {"color": "#2ca02c", "marker": "o", "label": "真报 (True Alarm)",  "s": 80},
            "false":       {"color": "#d62728", "marker": "X", "label": "误报 (False Alarm)", "s": 90},
            "unevaluated": {"color": "#7f7f7f", "marker": "D", "label": "未评估 (Unevaluated)", "s": 70},
        }

        for idx, (name, result) in enumerate(self.results.items()):
            ax = axes_list[idx + 1]

            # 浅色 ILI 背景
            ax.plot(x, self.ili_raw, color="lightgray", linewidth=0.8, alpha=0.5)

            # 暴发峰垂线
            for p in self.outbreak_points:
                if p < len(self.ili_raw):
                    ax.axvline(x=p, color="#d62728", linestyle="--", alpha=0.3, linewidth=0.7)

            # 按类别绘制预警点
            cat_keys = [
                ("true", "true_alarm_indices"),
                ("false", "false_alarm_indices"),
                ("unevaluated", "unevaluated_indices"),
            ]
            plotted = set()
            for cat, key in cat_keys:
                indices = result.get(key, [])
                if not indices:
                    continue
                valid = [i for i in indices if 0 <= i < len(self.ili_raw)]
                if not valid:
                    continue
                s = style[cat]
                lbl = s["label"] if cat not in plotted else ""
                plotted.add(cat)
                ax.scatter(
                    valid, self.ili_raw[valid],
                    color=s["color"], marker=s["marker"], s=s["s"],
                    edgecolors="black", linewidth=0.4, zorder=4, label=lbl,
                )

            ax.set_ylabel("ILI", fontsize=10)
            ax.set_title(f"{name}", fontsize=12, fontweight="bold")
            ax.legend(loc="upper left", fontsize=9, framealpha=0.9, ncol=3)
            ax.grid(True, alpha=0.2)
            ax.set_xlim(0, len(self.ili_raw) - 1)

        # X 轴日期标签
        n = len(self.date_list)
        step = max(1, n // 12)
        tick_pos = list(range(0, n, step))
        axes_list[-1].set_xticks(tick_pos)
        axes_list[-1].set_xticklabels(
            [self.date_list[i] if i < n else "" for i in tick_pos],
            rotation=35, ha="right", fontsize=8,
        )
        axes_list[-1].set_xlabel("日期", fontsize=11)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  评估结果图已保存: {save_path}")
        plt.show()

    # ---------------------------------------------------------------
    # 便捷方法（快速评估，一步到位）
    # ---------------------------------------------------------------

    @classmethod
    def quick_evaluate(
        cls,
        ili_raw: np.ndarray,
        date_list: List[str],
        alert_sources: Dict[str, List[int]],
        data_path: str,
        **kwargs,
    ) -> "AlertEvaluator":
        """
        快捷工厂方法：创建 + 添加源 + run 一步完成

        Args:
            ili_raw: ILI 原始数据
            date_list: 日期列表
            alert_sources: {源名称: 预警点索引列表}
            data_path: Excel 数据文件路径
            **kwargs: 其他传给 __init__ 的参数

        Returns:
            AlertEvaluator: 已执行完 run() 的实例
        """
        evaluator = cls(ili_raw=ili_raw, date_list=date_list, data_path=data_path, **kwargs)
        for name, indices in alert_sources.items():
            evaluator.add_source(name, indices)
        evaluator.run()
        return evaluator


# ================================================================
# 独立运行示例
# ================================================================
if __name__ == "__main__":
    # 模拟数据
    np.random.seed(42)
    t = np.arange(500)
    ili = 50 + 30 * np.sin(2 * np.pi * t / 365) + np.random.randn(500) * 5
    ili[ili < 0] = 0
    dates = [f"2020-{i//30+1:02d}-{i%30+1:02d}" for i in t]

    demo_sources = {
        "CNE":             [50, 65, 185, 195, 300, 350, 400],
        "tRCE":            [48, 62, 190, 310, 345, 398],
        "Fusion(严重)":    [55, 70, 188, 340, 355],
    }

    # 演示标准接口
    evaluator = AlertEvaluator(ili, dates, data_path="测试数据全国_填充.xlsx")
    for name, pts in demo_sources.items():
        evaluator.add_source(name, pts)

    results = evaluator.run()
    evaluator.print_comparison()
    evaluator.plot_results(save_path="experiment_results/demo_evaluation.png")
