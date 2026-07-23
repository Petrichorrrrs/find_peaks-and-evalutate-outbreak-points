import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from typing import List, Tuple, Optional


def detect_ili_peaks(filepath: str = "测试数据全国_填充.xlsx",
                     sigma: float = 5,
                     prominence_factor: float = 1.0,
                     distance: int = 60,
                     width: int = 7) -> Tuple[List[int], dict]:
    """
    从流感数据中检测暴发峰值点（供 run_experiment.py 等模块复用）

    Args:
        filepath:     Excel 数据文件路径
        sigma:        高斯平滑参数，默认 5
        prominence_factor: 峰值显著性系数（乘以标准差），默认 1.0
        distance:     峰之间最小间隔（天），默认 60
        width:        峰最小宽度（天），默认 7

    Returns:
        (peak_indices, peak_properties)
        peak_indices: 峰值在原始数组中的索引列表
        peak_properties: scipy find_peaks 返回的 properties 字典
    """
    df = pd.read_excel(filepath)
    df["date"] = pd.to_datetime(df["date"])
    cases = df["ILI"].values

    # 高斯平滑
    smooth = gaussian_filter1d(cases, sigma=sigma)

    # 峰值检测
    peaks, properties = find_peaks(
        smooth,
        prominence=prominence_factor * np.std(smooth),
        distance=distance,
        width=width
    )

    return peaks.tolist(), {"peaks": peaks, "properties": properties, "smooth": smooth, "cases": cases, "df": df}


# ==========================
# 独立运行：读取数据 → 检测峰值 → 绘图
# ==========================
if __name__ == "__main__":
    peak_indices, data = detect_ili_peaks()
    peaks = data["peaks"]
    properties = data["properties"]
    smooth = data["smooth"]
    cases = data["cases"]
    df = data["df"]

    print(f"检测到 {len(peaks)} 次流感暴发：\n")
    for i, p in enumerate(peaks):
        print(f"第{i+1}次暴发")
        print("日期：", df.loc[p, "date"].date())
        print("峰值：", int(cases[p]))
        print("显著性：", round(properties["prominences"][i], 1))
        print()

    # 绘图
    plt.figure(figsize=(15, 5))
    plt.plot(df["date"], cases, color="lightgray", label="Daily Cases")
    plt.plot(df["date"], smooth, linewidth=2, label="Smoothed")
    plt.scatter(df.loc[peaks, "date"], smooth[peaks],
                color="red", s=80, zorder=5, label="Epidemic Peak")
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Cases")
    plt.title("Influenza Epidemic Peaks")
    plt.tight_layout()
    plt.show()