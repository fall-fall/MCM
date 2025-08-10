# -*- coding: utf-8 -*-
# 文件名: run_validation_chart.py
# 功能: 对问题一和问题二的结果进行蒙特卡洛模拟，并绘制利润分布图。
# 版本: 2.0 (适配新文件结构)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import re

# --- 1. 参数配置 ---
# 蒙特卡洛模拟次数
N_SIMULATIONS = 10000

# 问题二描述的不确定性参数
COST_GROWTH_RATE = 0.05
YIELD_SHOCK_RANGE = 0.10      # 亩产量每年 +/-10%
GRAIN_DEMAND_GROWTH_RANGE = (0.05, 0.10) # 小麦玉米年增长率 5%~10%
OTHER_DEMAND_SHOCK_RANGE = 0.05 # 其他作物年变化 +/-5%
VEG_PRICE_GROWTH_RATE = 0.05
FUNGI_PRICE_DROP_RANGE = (0.01, 0.05)
MOREL_PRICE_DROP_RATE = 0.05
DISCOUNT_RATIO = 0.5          # 降价销售比例

# --- 2. 数据加载模块 ---

def load_base_parameters(data_path_f1, data_path_f2):
    """加载附件1和2中的基础参数，为模拟做准备"""
    try:
        plots_df = pd.read_excel(data_path_f1, sheet_name='乡村的现有耕地')
        crops_info_df = pd.read_excel(data_path_f1, sheet_name='乡村种植的农作物')
        stats_df = pd.read_excel(data_path_f2, sheet_name='2023年统计的相关数据')
        
        params = {}
        params['P_plot_type'] = dict(zip(plots_df['地块名称'], plots_df['地块类型']))
        params['P_crop_type'] = dict(zip(crops_info_df['作物名称'], crops_info_df['作物类型']))
        
        def clean_and_convert_price(value):
            if isinstance(value, str) and any(c in value for c in '-–—'):
                parts = re.split(r'[-–—]', value.strip())
                try: return (float(parts[0]) + float(parts[1])) / 2
                except (ValueError, IndexError): return np.nan
            return pd.to_numeric(value, errors='coerce')

        stats_df['销售单价/(元/斤)'] = stats_df['销售单价/(元/斤)'].apply(clean_and_convert_price)
        stats_df['亩产量/千克'] = pd.to_numeric(stats_df['亩产量/斤'], errors='coerce') / 2
        stats_df['销售单价/(元/千克)'] = pd.to_numeric(stats_df['销售单价/(元/斤)'], errors='coerce') * 2
        stats_df['种植成本/(元/亩)'] = pd.to_numeric(stats_df['种植成本/(元/亩)'], errors='coerce')
        stats_df.dropna(subset=['亩产量/千克', '种植成本/(元/亩)', '销售单价/(元/千克)'], inplace=True)
        
        params['P_cost_base'], params['P_yield_base'], params['P_price_base'] = {}, {}, {}
        for _, row in stats_df.iterrows():
            key = (row['作物名称'], row['地块类型'])
            params['P_cost_base'][key] = row['种植成本/(元/亩)']
            params['P_yield_base'][key] = row['亩产量/千克']
            if row['作物名称'] not in params['P_price_base']:
                params['P_price_base'][row['作物名称']] = row['销售单价/(元/千克)']

        past_planting_df = pd.read_excel(data_path_f2, sheet_name='2023年的农作物种植情况')
        params['P_demand_base'] = {j: 0 for j in crops_info_df['作物名称'].unique()}
        temp_details = pd.merge(past_planting_df, plots_df, left_on='种植地块', right_on='地块名称')
        for j in params['P_demand_base'].keys():
            total_yield = sum(
                params['P_yield_base'].get((j, row['地块类型']), 0) * row['种植面积/亩']
                for _, row in temp_details[temp_details['作物名称'] == j].iterrows()
            )
            params['P_demand_base'][j] = total_yield if total_yield > 0 else 1000
        
        return params
    except FileNotFoundError as e:
        print(f"错误：无法找到数据文件 {e}。请确保文件路径正确。")
        return None

# --- 3. 蒙特卡洛模拟器 ---

def run_monte_carlo_simulation(planting_plan_df, base_params):
    """对给定的种植方案进行蒙特卡洛模拟"""
    total_profits = []
    
    grain_crops = [j for j, t in base_params['P_crop_type'].items() if '粮食' in str(t)]
    veg_crops = [j for j, t in base_params['P_crop_type'].items() if '蔬菜' in str(t)]
    fungi_crops = [j for j, t in base_params['P_crop_type'].items() if '食用菌' in str(t)]

    for i in range(N_SIMULATIONS):
        if (i + 1) % 1000 == 0:
            print(f"  正在进行第 {i+1}/{N_SIMULATIONS} 次模拟...")
            
        sim_total_profit = 0
        for year in range(2024, 2031):
            year_plan = planting_plan_df[planting_plan_df['年份'] == year]
            total_production = {j: 0 for j in base_params['P_demand_base']}
            total_cost, total_revenue = 0, 0

            for _, row in year_plan.iterrows():
                crop, plot, area = row['作物名称'], row['地块编号'], row['种植面积（亩）']
                plot_type = base_params['P_plot_type'].get(plot)
                
                base_yield = base_params['P_yield_base'].get((crop, plot_type), 0)
                yield_shock = np.random.uniform(1 - YIELD_SHOCK_RANGE, 1 + YIELD_SHOCK_RANGE)
                total_production[crop] += (base_yield * yield_shock) * area
                
                base_cost = base_params['P_cost_base'].get((crop, plot_type), 0)
                total_cost += (base_cost * ((1 + COST_GROWTH_RATE) ** (year - 2023))) * area

            for crop, produced_qty in total_production.items():
                if produced_qty == 0: continue
                
                base_price = base_params['P_price_base'].get(crop, 0)
                sim_price, year_factor = base_price, year - 2023
                if crop in veg_crops: sim_price *= ((1 + VEG_PRICE_GROWTH_RATE) ** year_factor)
                elif crop == '羊肚菌': sim_price *= ((1 - MOREL_PRICE_DROP_RATE) ** year_factor)
                elif crop in fungi_crops:
                    drop_rate = np.random.uniform(FUNGI_PRICE_DROP_RANGE[0], FUNGI_PRICE_DROP_RANGE[1])
                    sim_price *= ((1 - drop_rate) ** year_factor)
                
                base_demand = base_params['P_demand_base'].get(crop, 0)
                sim_demand = base_demand
                if crop in grain_crops:
                    growth_rate = np.random.uniform(GRAIN_DEMAND_GROWTH_RANGE[0], GRAIN_DEMAND_GROWTH_RANGE[1])
                    sim_demand *= ((1 + growth_rate) ** year_factor)
                else:
                    shock_rate = np.random.uniform(-OTHER_DEMAND_SHOCK_RANGE, OTHER_DEMAND_SHOCK_RANGE)
                    sim_demand *= (1 + shock_rate)
                
                qty_normal_sale = min(produced_qty, sim_demand)
                qty_discount_sale = produced_qty - qty_normal_sale
                total_revenue += (qty_normal_sale * sim_price) + (qty_discount_sale * sim_price * DISCOUNT_RATIO)
            
            sim_total_profit += (total_revenue - total_cost)
        total_profits.append(sim_total_profit)
    return total_profits

# --- 4. 绘图模块 ---

def plot_profit_distribution(results_dict, output_path):
    """绘制学术风格的利润分布直方图"""
    print("\n正在生成最终对比图表...")
    
    # 遵照您的要求，设置学术风格图表
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(12, 7))

    colors = {'方案一 (确定性)': '#4c72b0', '方案二 (鲁棒性)': '#55a868'}
    
    for name, profits in results_dict.items():
        if not profits: continue
        
        sns.histplot(profits, kde=True, ax=ax, label=name, color=colors.get(name, 'gray'),
                     bins=50, stat='density', alpha=0.6, edgecolor=None)
        
        mean_profit, std_profit = np.mean(profits), np.std(profits)
        ax.axvline(mean_profit, color=colors.get(name, 'gray'), linestyle='--', lw=2,
                    label=f'{name} 均值: {mean_profit:,.0f}')
        
        ax.text(0.02, 0.95 - len(ax.texts) * 0.1, 
                f'{name}:\n  均值 (期望利润): {mean_profit:,.0f} 元\n  标准差 (风险): {std_profit:,.0f} 元',
                transform=ax.transAxes, verticalalignment='top', 
                bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7))

    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax.get_yaxis().set_visible(False)
    ax.set_xlabel('7年总利润 (元)', fontsize=14)
    ax.set_title('不同策略下总利润的蒙特卡洛模拟分布', fontsize=18, pad=20)
    ax.legend(fontsize=12, loc='upper right')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n图表已成功保存至: {output_path}")

# --- 5. 主程序 ---

if __name__ == '__main__':
    try:
        # --- 【关键修改】根据您的文件结构调整路径 ---
        # 1. 获取当前脚本所在目录 (Code/Chart/)
        script_dir = Path(__file__).parent
        # 2. 获取项目根目录 (向上两级)
        project_root = script_dir.parent.parent
        # 3. 构建数据输入和图表输出目录
        data_dir = project_root / 'Data'
        output_dir = script_dir / 'result'
        
        # 确保输出目录存在
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 4. 定义所有文件路径
        data_f1 = data_dir / '附件1.xlsx'
        data_f2 = data_dir / '附件2.xlsx'
        result2_file = data_dir / 'result2.xlsx'
        result1_file = data_dir / 'result1_2.xlsx' # 假设对比方案也放在Data目录
        chart_output_path = output_dir / "profit_distribution_comparison.png"
        
        print("--- 开始执行鲁棒性验证程序 ---")
        print(f"数据来源: {data_dir}")
        print(f"图表输出: {output_dir}")
        
        base_params = load_base_parameters(data_f1, data_f2)
        
        if base_params:
            simulation_results = {}
            
            # 加载并模拟问题二方案
            try:
                plan2_df = pd.read_excel(result2_file)
                print(f"\n成功加载问题二方案: {result2_file.name}")
                print("开始对问题二方案进行模拟...")
                profits_q2 = run_monte_carlo_simulation(plan2_df, base_params)
                simulation_results['方案二 (鲁棒性)'] = profits_q2
            except FileNotFoundError:
                print(f"错误：未找到问题二的结果文件 {result2_file.name}，无法进行分析。")
                exit()
            
            # 加载并模拟问题一方案
            try:
                plan1_df = pd.read_excel(result1_file)
                print(f"\n成功加载问题一方案: {result1_file.name}")
                print("开始对问题一方案进行模拟...")
                profits_q1 = run_monte_carlo_simulation(plan1_df, base_params)
                simulation_results['方案一 (确定性)'] = profits_q1
            except FileNotFoundError:
                print(f"\n警告：未找到问题一的结果文件 {result1_file.name}，将仅分析问题二的方案。")
                
            # 绘图
            plot_profit_distribution(simulation_results, chart_output_path)

    except Exception as e:
        print(f"\n程序发生未知错误: {e}")
        import traceback
        traceback.print_exc()