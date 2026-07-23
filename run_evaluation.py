"""
独立评估实验脚本

在已有上游预警结果的基础上，独立运行评估模块。
不依赖完整 pipeline，只需原始数据和预警结果文件即可运行。

执行流程：
  1. 加载原始 ILI 数据（data_processor）
  2. 加载上游预警结果（all_results.json）
  3. 交互式峰值确认 → 多源评估 → 比对表 → 结果图

用法：
    python run_evaluation.py                                        # 默认读取 experiment_results/
    python run_evaluation.py --data-path 测试数据全国_填充.xlsx      # 指定数据文件
    python run_evaluation.py --results-path path/to/results.json     # 指定结果文件
    python run_evaluation.py --skip-confirm                          # 跳过峰值确认(用自动检测结果)

输出：
    - experiment_results/evaluation_result.png   — 评估结果图
    - experiment_results/evaluation_metrics.json — 评估指标 JSON
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
import time
from datetime import datetime


def load_alert_sources_from_json(results_path: str) -> dict:
    """
    从 all_results.json 中提取各预警源的预警点索引

    Returns:
        {"CNE": [50, 64, ...], "tRCE": [...], "Fusion(严重)": [...]}
    """
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sources = {}

    # CNE
    if "cne" in data and "alert_indices" in data["cne"]:
        pts = data["cne"]["alert_indices"]
        if pts:
            sources["CNE"] = pts

    # tRCE
    if "trce" in data and "alert_indices" in data["trce"]:
        pts = data["trce"]["alert_indices"]
        if pts:
            sources["tRCE"] = pts

    # Fusion(严重) — critical 是橙+红合并去重后的预警点
    if "fusion" in data and "alert_indices" in data["fusion"]:
        alert_idx = data["fusion"]["alert_indices"]
        if isinstance(alert_idx, dict) and "critical" in alert_idx:
            pts = alert_idx["critical"]
            if pts:
                sources["Fusion(严重)"] = pts

    return sources


def load_alert_sources_from_csv(output_dir: str) -> dict:
    """从 CSV 文件加载预警源（all_results.json 不存在时的降级方案）"""
    sources = {}
    csv_cfg = [
        ("CNE", "cne_alerts.csv", "alert_index"),
        ("tRCE", "trce_alerts.csv", "alert_index"),
        ("Fusion(严重)", "critical_alerts.csv", "index"),
    ]
    for name, filename, col in csv_cfg:
        path = os.path.join(output_dir, filename)
        if os.path.exists(path):
            df = pd.read_csv(path)
            if col in df.columns:
                pts = df[col].dropna().astype(int).tolist()
                if pts:
                    sources[name] = pts
    return sources


def main():
    parser = argparse.ArgumentParser(
        description="独立预警评估实验 — 加载已有预警结果，执行峰值检测与多源评估",
    )
    parser.add_argument(
        "--data-path", type=str, default="测试数据全国_填充.xlsx",
        help="原始 ILI 数据文件路径（默认: 测试数据全国_填充.xlsx）",
    )
    parser.add_argument(
        "--results-path", type=str, default="experiment_results/all_results.json",
        help="上游预警结果 JSON 路径（默认: experiment_results/all_results.json）",
    )
    parser.add_argument(
        "--output-dir", type=str, default="experiment_results",
        help="输出目录（默认: experiment_results）",
    )
    parser.add_argument(
        "--skip-confirm", action="store_true",
        help="跳过交互式峰值确认，直接使用自动检测结果",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("=" * 70)
    print("独立预警评估实验")
    print("开始时间:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 70)
    print(f"   数据文件: {args.data_path}")
    print(f"   结果文件: {args.results_path}")
    if args.skip_confirm:
        print("   峰值确认: 跳过交互，使用自动检测结果")

    timing = {}

    # ============================================================
    # Step 1: 加载原始 ILI 数据
    # ============================================================
    print("\n[1/3] 加载数据...")
    t0 = time.time()
    try:
        from data_processor import load_and_process_data
        data = load_and_process_data(args.data_path)
        if data is None:
            print("   错误: 数据加载失败")
            return
        timing["data_loading"] = time.time() - t0
        print(f"   完成! 耗时: {timing['data_loading']:.2f}秒")
        print(f"   - 时间序列长度: {len(data['date_list'])}")
        print(f"   - 搜索指数数量: {data['search_data_raw'].shape[0]}")
    except Exception as e:
        print(f"   错误: {e}")
        import traceback
        traceback.print_exc()
        return

    # ============================================================
    # Step 2: 加载上游预警结果
    # ============================================================
    print("\n[2/3] 加载上游预警结果...")
    t0 = time.time()
    try:
        if os.path.exists(args.results_path):
            alert_sources = load_alert_sources_from_json(args.results_path)
            print(f"   从 {args.results_path} 加载")
        else:
            print(f"   未找到 {args.results_path}，尝试从 CSV 加载...")
            alert_sources = load_alert_sources_from_csv(output_dir)

        if not alert_sources:
            print("   错误: 未找到任何预警源数据")
            print("   请先运行 python run_experiment.py 生成预警结果")
            return

        timing["loading_results"] = time.time() - t0
        print(f"   完成! 耗时: {timing['loading_results']:.2f}秒")
        for name, pts in alert_sources.items():
            print(f"   - {name}: {len(pts)} 个预警点")
    except Exception as e:
        print(f"   错误: {e}")
        import traceback
        traceback.print_exc()
        return

    # ============================================================
    # Step 3: 执行评估
    # ============================================================
    print("\n[3/3] 执行预警评估...")
    t0 = time.time()
    try:
        from evaluate_alerts import AlertEvaluator

        evaluator = AlertEvaluator(
            ili_raw=data["ili_raw"],
            date_list=data["date_list"],
            data_path=args.data_path,
        )

        # 注册预警源
        for name, pts in alert_sources.items():
            evaluator.add_source(name, pts)

        # 执行评估（run() 内部：弹图 + 询问；skip_confirm=True 时跳过交互）
        results = evaluator.run(skip_confirm=args.skip_confirm)

        # 输出
        print("\n" + "=" * 70)
        print("  评估结果")
        print("=" * 70)
        evaluator.print_comparison()

        # 保存结果图
        plot_path = os.path.join(output_dir, "evaluation_result.png")
        evaluator.plot_results(save_path=plot_path)

        # 保存指标 JSON
        metrics_path = os.path.join(output_dir, "evaluation_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n   指标已保存: {metrics_path}")

        timing["evaluation"] = time.time() - t0
        print(f"   耗时: {timing['evaluation']:.2f}秒")

    except Exception as e:
        print(f"   错误: {e}")
        import traceback
        traceback.print_exc()
        timing["evaluation"] = time.time() - t0
        return

    # ========== 汇总 ==========
    total = sum(timing.values())
    print("\n" + "=" * 70)
    print("实验完成!")
    print("=" * 70)
    print("\n耗时统计:")
    for mod, t in timing.items():
        print(f"   {mod}: {t:.2f}秒")
    print(f"   总计: {total:.2f}秒")
    print(f"\n输出文件:")
    print(f"   {plot_path}")
    print(f"   {metrics_path}")
    print("\n结束时间:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    main()
