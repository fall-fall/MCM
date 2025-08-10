# -*- coding: utf-8 -*-
# 文件名: run_final_analysis.py
# 功能: (一体化脚本) 完整执行蒙特卡洛模拟、指标计算，并生成所有图表
# 版本: 4.0

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
YIELD_SHOCK_RANGE = 0.10
GRAIN_DEMAND_GROWTH_RANGE = (0.05, 0.10)
OTHER_DEMAND_SHOCK_RANGE = 0.05
VEG_PRICE_GROWTH_RATE = 0.05
FUNGI_PRICE_DROP_RANGE = (0.01, 0.05)
MOREL_PRICE_DROP_RATE = 0.05
DISCOUNT_RATIO = 0.5

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

# --- 4. 分析与可视化模块 ---

def analyze_and_visualize_results(results_dict):
    """
    (一体化函数) 计算指标，打印并保存指标表格图片，最后生成并保存利润分布图。
    """
    if not results_dict:
        print("没有可供分析的结果。")
        return

    # --- Part 1: 计算指标并准备DataFrame ---
    summary_data = []
    name_map = {'profits_q1': '方案一 (确定性)', 'profits_q2': '方案二 (鲁棒性)'}
    plan_order = ['profits_q1', 'profits_q2']

    for key in plan_order:
        if key in results_dict:
            profits = results_dict[key]
            name = name_map.get(key, key)
            profits_in_wan = np.array(profits) / 10000
            mean_profit, std_profit = np.mean(profits_in_wan), np.std(profits_in_wan)
            var_95 = np.percentile(profits_in_wan, 5)
            cvar_95 = np.mean(profits_in_wan[profits_in_wan <= var_95])
            summary_data.append({
                "度量指标": name, "平均总利润 (万元)": f"{mean_profit:,.2f}",
                "利润标准差 (万元)": f"{std_profit:,.2f}", "95% VaR (万元)": f"{var_95:,.2f}",
                "95% CVaR (万元)": f"{cvar_95:,.2f}"
            })
            
    summary_df = pd.DataFrame(summary_data).set_index("度量指标")
    
    # --- Part 2: 打印Markdown表格到控制台 ---
    print("\n\n" + "="*80 + "\n--- 关键绩效与风险指标量化分析 ---\n" + "="*80)
    print(summary_df.to_string())

    # --- Part 3: 生成指标表格图片 ---
    print("\n正在生成指标表格图片...")
    plt.style.use('default')
    plt.rcParams['font.family'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig_table, ax_table = plt.subplots(figsize=(10, 3.5))
    ax_table.axis('tight'); ax_table.axis('off')
    table = ax_table.table(cellText=summary_df.values, colLabels=summary_df.columns,
                         rowLabels=summary_df.index, cellLoc='center', loc='center')
    table.auto_set_font_size(False); table.set_fontsize(12); table.scale(1.2, 2.0)
    for (i, j), cell in table.get_celld().items():
        if i == 0 or j == -1:
            cell.set_facecolor("#f0f0f0"); cell.set_text_props(weight='bold')
            
    fig_table.suptitle('两种种植方案的关键绩效与风险指标对比', fontsize=16, weight='bold')
    fig_table.tight_layout(pad=1.5)
    table_output_path = "kpi_risk_table.png"
    plt.savefig(table_output_path, dpi=300, bbox_inches='tight')
    print(f"指标表格图片已保存至: {table_output_path}")
    plt.close(fig_table)

    # --- Part 4: 生成利润分布直方图 ---
    print("\n正在生成利润分布直方图...")
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = ['SimHei']
    fig_hist, ax_hist = plt.subplots(figsize=(12, 7))
    colors = {'方案一 (确定性)': '#4c72b0', '方案二 (鲁棒性)': '#55a868'}
    
    for name, profits in results_dict.items():
        if not profits: continue
        sns.histplot(profits, kde=True, ax=ax_hist, label=name, color=colors.get(name.split(" ")[0], 'gray'),
                     bins=50, stat='density', alpha=0.6, edgecolor=None)
        mean_profit = np.mean(profits)
        ax_hist.axvline(mean_profit, color=colors.get(name.split(" ")[0], 'gray'), linestyle='--', lw=2,
                        label=f'{name} 均值: {mean_profit:,.0f}')
    
    ax_hist.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax_hist.get_yaxis().set_visible(False)
    ax_hist.set_xlabel('7年总利润 (元)', fontsize=14)
    ax_hist.set_title('不同策略下总利润的蒙特卡洛模拟分布', fontsize=18, pad=20)
    ax_hist.legend(fontsize=12, loc='upper right')
    ax_hist.spines['top'].set_visible(False); ax_hist.spines['right'].set_visible(False); ax_hist.spines['left'].set_visible(False)
    plt.tight_layout()
    hist_output_path = "profit_distribution_comparison.png"
    plt.savefig(hist_output_path, dpi=300, bbox_inches='tight')
    print(f"利润分布图已保存至: {hist_output_path}")
    plt.close(fig_hist)

# --- 5. 主程序 ---

if __name__ == '__main__':
    try:
        # 假设所有文件都在脚本的同一级目录中
        current_dir = Path(__file__).parent /'..'/'..'/'Data'
        
        data_f1 = current_dir / '附件1.xlsx'
        data_f2 = current_dir / '附件2.xlsx'
        result2_file = current_dir / 'result2.xlsx'
        result1_file = current_dir / 'result1_2.xlsx'
        
        print("--- 开始执行一体化分析程序 ---")
        
        base_params = load_base_parameters(data_f1, data_f2)
        
        if base_params:
            simulation_results = {}
            
            # 加载并模拟问题二方案
            try:
                plan2_df = pd.read_excel(result2_file)
                print(f"\n成功加载问题二方案: {result2_file.name}")
                profits_q2 = run_monte_carlo_simulation(plan2_df, base_params)
                simulation_results['profits_q2'] = profits_q2
            except FileNotFoundError:
                print(f"错误：未找到问题二的结果文件 {result2_file.name}。")
                exit()
            
            # 加载并模拟问题一方案
            try:
                plan1_df = pd.read_excel(result1_file)
                print(f"\n成功加载问题一方案: {result1_file.name}")
                profits_q1 = run_monte_carlo_simulation(plan1_df, base_params)
                simulation_results['profits_q1'] = profits_q1
            except FileNotFoundError:
                print(f"\n警告：未找到问题一的结果文件 {result1_file.name}，部分对比分析将无法进行。")
                
            # 调用统一的分析和可视化函数
            analyze_and_visualize_results(simulation_results)
            print("\n--- 所有任务执行完毕 ---")

    except Exception as e:
        print(f"\n程序发生未知错误: {e}")
        import traceback
        traceback.print_exc()