# -*- coding: utf-8 -*-
# 文件名: run_q3_with_convergence_plot.py
# 功能: 问题三最终版，增加遗传算法收敛曲线的自动绘制与保存功能
# 版本: 7.0

import pandas as pd
import numpy as np
import os
import time
import re
import random
import copy
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. 模型核心参数配置区 ---
# (为了快速看到结果，您可以适当减少代数和种群大小，但在最终报告中建议使用较大数值)
POP_SIZE = 40
MAX_GEN = 120
CX_PROB = 0.8
MUT_PROB = 0.3
TOURNAMENT_SIZE = 3
N_SIMULATIONS = 25

# (市场经济模型参数与之前相同)
SUPPLY_PRICE_ELASTICITY = 0.4
SURPLUS_SALE_PRICE_RATIO = 0.5
YIELD_SHOCK_RANGE = 0.15
PRICE_SHOCK_STD = 0.05
DEMAND_BASE_GROWTH = 1.02
DEMAND_SHOCK_RANGE = 0.20

# --- 2. 核心功能函数 (与之前版本相同) ---
# (为保证完整性，此处包含所有函数，未作省略)

def load_and_prepare_data(data_path_f1, data_path_f2):
    """加载并准备数据"""
    try:
        print("（1）正在读取Excel文件...")
        plots_df = pd.read_excel(data_path_f1, sheet_name='乡村的现有耕地')
        crops_info_df = pd.read_excel(data_path_f1, sheet_name='乡村种植的农作物')
        stats_df = pd.read_excel(data_path_f2, sheet_name='2023年统计的相关数据')
        past_planting_df = pd.read_excel(data_path_f2, sheet_name='2023年的农作物种植情况')
        
        for df in [plots_df, crops_info_df, stats_df, past_planting_df]:
            df.columns = df.columns.str.strip()
        
        params = {}
        params['I_plots'] = sorted(plots_df['地块名称'].tolist())
        params['P_area'] = dict(zip(plots_df['地块名称'], plots_df['地块面积/亩']))
        params['P_plot_type'] = dict(zip(plots_df['地块名称'], plots_df['地块类型']))
        
        params['J_crops'] = sorted(crops_info_df['作物名称'].dropna().unique().tolist())
        params['P_crop_type'] = dict(zip(crops_info_df['作物名称'], crops_info_df['作物类型']))
        params['J_bean'] = [j for j, ctype in params['P_crop_type'].items() if isinstance(ctype, str) and '豆' in ctype]
        
        params['P_past'] = {i: {1: None, 2: None} for i in params['I_plots']}
        for _, row in past_planting_df.iterrows():
            plot, crop, season = row['种植地块'], row['作物名称'], row.get('种植季节', 1)
            if plot in params['I_plots']:
                params['P_past'][plot][season] = crop

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
        
        params['P_yield_base'], params['P_cost_base'], params['P_price_base'] = {}, {}, {}
        for _, row in stats_df.iterrows():
            key = (row['作物名称'], row['地块类型'])
            params['P_cost_base'][key] = row['种植成本/(元/亩)']
            params['P_yield_base'][key] = row['亩产量/千克']
            if row['作物名称'] not in params['P_price_base']:
                params['P_price_base'][row['作物名称']] = row['销售单价/(元/千克)']

        params['P_demand_base'] = {j: 0 for j in params['J_crops']}
        temp_details = pd.merge(past_planting_df, plots_df, left_on='种植地块', right_on='地块名称')
        for j in params['J_crops']:
            total_yield = sum(
                params['P_yield_base'].get((j, row['地块类型']), 0) * row['种植面积/亩']
                for _, row in temp_details[temp_details['作物名称'] == j].iterrows()
            )
            params['P_demand_base'][j] = total_yield if total_yield > 0 else 1000
            
        params['S_suitability'] = {}
        for i in params['I_plots']:
            plot_t = params['P_plot_type'].get(i, '')
            for j in params['J_crops']:
                crop_t = params['P_crop_type'].get(j, '')
                is_bean = j in params['J_bean']
                is_veg = '蔬菜' in str(crop_t)
                for k in [1, 2]:
                    suitable = 0
                    if plot_t in ['平旱地', '梯田', '山坡地'] and ('粮食' in str(crop_t) or is_bean) and k == 1: suitable = 1
                    elif plot_t == '水浇地' and ('水稻' in str(crop_t) or (is_veg and '大白菜' not in j and '萝卜' not in j)): suitable = 1
                    elif plot_t == '水浇地' and is_veg and ('大白菜' in j or '萝卜' in j) and k == 2: suitable = 1
                    elif plot_t == '普通大棚' and is_veg and k == 1: suitable = 1
                    elif plot_t == '普通大棚' and '食用菌' in str(crop_t) and k == 2: suitable = 1
                    elif plot_t == '智慧大棚' and is_veg: suitable = 1
                    params['S_suitability'][(i, j, k)] = suitable
        print(" -> 数据参数准备完成。")
        return params
    except Exception as e:
        print(f"错误: 加载数据失败: {e}")
        raise

def create_initial_solution(params):
    # (此处省略函数内部代码，与上一版相同)
    solution = {y: {k: {i: None for i in params['I_plots']} for k in [1, 2]} for y in range(2024, 2031)}
    for y in range(2024, 2031):
        for k in [1, 2]:
            for i in params['I_plots']:
                possible_crops = [j for j in params['J_crops'] if params['S_suitability'].get((i, j, k), 0) == 1]
                if possible_crops:
                    solution[y][k][i] = random.choice(possible_crops)
    return repair_solution(solution, params)

def repair_solution(solution, params):
    # (此处省略函数内部代码，与上一版相同)
    def get_crops_in_year(sol, y, i):
        crops = set()
        if y in sol:
            for k in [1, 2]:
                crop = sol.get(y, {}).get(k, {}).get(i)
                if crop: crops.add(crop)
        return list(crops)
    for i in params['I_plots']:
        for y in range(2024, 2031):
            crops_this_year = get_crops_in_year(solution, y, i)
            crops_last_year = get_crops_in_year(solution, y - 1, i) if y > 2024 else get_crops_in_year(params['P_past'], 2023, i)
            common_crops = set(crops_this_year) & set(crops_last_year)
            if common_crops:
                for k in [1, 2]:
                    if solution[y][k][i] in common_crops:
                        possible_crops = [j for j in params['J_crops'] if params['S_suitability'].get((i, j, k), 0) == 1 and j not in crops_last_year]
                        solution[y][k][i] = random.choice(possible_crops) if possible_crops else None
        all_years = [2023] + sorted(solution.keys())
        for y_start_idx in range(len(all_years) - 2):
            window = all_years[y_start_idx : y_start_idx + 3]
            contains_bean = False
            for y_win in window:
                crops_in_win = get_crops_in_year(solution if y_win > 2023 else params['P_past'], y_win, i)
                if any(c in params['J_bean'] for c in crops_in_win):
                    contains_bean = True
                    break
            if not contains_bean:
                for _ in range(5):
                    y_fix = random.choice([y for y in window if y > 2023])
                    k_fix = random.choice([1, 2])
                    crops_last_year = get_crops_in_year(solution, y_fix - 1, i) if y_fix > 2024 else get_crops_in_year(params['P_past'], 2023, i)
                    possible_beans = [b for b in params['J_bean'] if params['S_suitability'].get((i, b, k_fix), 0) == 1 and b not in crops_last_year]
                    if possible_beans:
                        solution[y_fix][k_fix][i] = random.choice(possible_beans)
                        break
    return solution

def crossover(p1, p2, params):
    # (此处省略函数内部代码，与上一版相同)
    child = copy.deepcopy(p1)
    for i in params['I_plots']:
        if random.random() < 0.5:
            for y in range(2024, 2031):
                for k in [1, 2]:
                    child[y][k][i] = p2[y][k][i]
    return child

def mutate(solution, params):
    # (此处省略函数内部代码，与上一版相同)
    mut_sol = copy.deepcopy(solution)
    for _ in range(random.randint(1, 5)):
        y, k, i = random.choice(list(range(2024, 2031))), random.choice([1, 2]), random.choice(params['I_plots'])
        possible_crops = [j for j in params['J_crops'] if params['S_suitability'].get((i, j, k), 0) == 1]
        if possible_crops:
            mut_sol[y][k][i] = random.choice(possible_crops)
    return mut_sol

def evaluate_fitness(solution, params, cov_matrix_L):
    # (此处省略函数内部代码，与上一版相同)
    simulation_total_profits = []
    for _ in range(N_SIMULATIONS):
        yearly_profits = []
        for y in range(2024, 2031):
            uncorr_shocks = np.random.randn(len(params['J_crops']))
            price_corr_shocks = cov_matrix_L @ uncorr_shocks
            total_supply, year_cost = {j: 0 for j in params['J_crops']}, 0
            for k in [1, 2]:
                for i in params['I_plots']:
                    crop = solution[y][k][i]
                    if not crop: continue
                    plot_type = params['P_plot_type'][i]
                    base_yield = params['P_yield_base'].get((crop, plot_type), 0)
                    yield_shock = 1 + np.random.uniform(-YIELD_SHOCK_RANGE, YIELD_SHOCK_RANGE)
                    total_supply[crop] += params['P_area'][i] * base_yield * yield_shock
                    base_cost = params['P_cost_base'].get((crop, plot_type), 0)
                    year_cost += params['P_area'][i] * base_cost * (1.05 ** (y - 2023))
            year_revenue = 0
            for idx, j in enumerate(params['J_crops']):
                if total_supply.get(j, 0) <= 0: continue
                base_price = params['P_price_base'].get(j, 0)
                price_shock = 1 + price_corr_shocks[idx] * PRICE_SHOCK_STD
                supply_ratio = params['P_demand_base'].get(j, 1) / (total_supply[j] if total_supply[j] > 0 else 1)
                sim_price_primary = base_price * price_shock * (supply_ratio ** SUPPLY_PRICE_ELASTICITY)
                demand_shock = 1 + np.random.uniform(-DEMAND_SHOCK_RANGE, DEMAND_SHOCK_RANGE)
                sim_demand = params['P_demand_base'].get(j, 1) * (DEMAND_BASE_GROWTH ** (y - 2023)) * demand_shock
                qty_produced, qty_sold_primary = total_supply[j], min(total_supply[j], sim_demand)
                qty_sold_surplus, sim_price_surplus = qty_produced - qty_sold_primary, sim_price_primary * SURPLUS_SALE_PRICE_RATIO
                year_revenue += (qty_sold_primary * sim_price_primary) + (qty_sold_surplus * sim_price_surplus)
            yearly_profits.append(year_revenue - year_cost)
        simulation_total_profits.append(sum(yearly_profits))
    mean_profit, std_profit = np.mean(simulation_total_profits), np.std(simulation_total_profits)
    return mean_profit, std_profit

def run_genetic_algorithm(params, cov_matrix_L):
    """【修改】遗传算法运行器，增加历史记录功能"""
    print("\n--- 开始执行遗传算法优化 ---")
    population = [create_initial_solution(params) for _ in range(POP_SIZE)]
    
    # 【新增】用于记录历史数据的列表
    history_best_fitness = []
    history_avg_fitness = []

    best_solution_overall = None
    best_fitness_overall = -np.inf

    for gen in range(MAX_GEN):
        start_time = time.time()
        eval_results = [evaluate_fitness(sol, params, cov_matrix_L) for sol in population]
        fitnesses = [profit for profit, risk in eval_results] # 适应度仅基于平均利润

        # 【新增】记录当前代的最佳和平均适应度
        current_best = np.max(fitnesses)
        current_avg = np.mean(fitnesses)
        history_best_fitness.append(current_best)
        history_avg_fitness.append(current_avg)

        if current_best > best_fitness_overall:
            best_fitness_overall = current_best
            best_solution_overall = copy.deepcopy(population[np.argmax(fitnesses)])
        
        end_time = time.time()
        if (gen + 1) % 5 == 0: 
            print(f"  第 {gen+1}/{MAX_GEN} 代, 本代最优: {current_best:,.0f}, "
                  f"本代平均: {current_avg:,.0f}, "
                  f"历史最优: {best_fitness_overall:,.0f}, "
                  f"耗时: {end_time - start_time:.2f} 秒")
        
        # 进化过程 (精英保留 + 锦标赛选择)
        new_population = [best_solution_overall]
        def tournament_selection(pop, fits, k):
            best = random.randrange(len(pop))
            for _ in range(k - 1):
                i = random.randrange(len(pop))
                if fits[i] > fits[best]: best = i
            return pop[best]

        while len(new_population) < POP_SIZE:
            p1 = tournament_selection(population, fitnesses, TOURNAMENT_SIZE)
            p2 = tournament_selection(population, fitnesses, TOURNAMENT_SIZE)
            child = crossover(p1, p2, params) if random.random() < CX_PROB else copy.deepcopy(p1)
            if random.random() < MUT_PROB: child = mutate(child, params)
            new_population.append(repair_solution(child, params))
        population = new_population
        
    print("--- 遗传算法优化完成 ---")
    return best_solution_overall, best_fitness_overall, history_best_fitness, history_avg_fitness

# --- 3. 【新增】绘图函数 ---

def plot_ga_convergence(history_best, history_avg, output_path):
    """
    绘制并保存遗传算法的收敛曲线图
    """
    print("\n正在生成GA收敛曲线图...")
    
    # 遵照您的要求，设置学术风格图表
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(12, 7))

    generations = range(1, len(history_best) + 1)
    
    # 绘制曲线
    ax.plot(generations, history_best, color='#4c72b0', linestyle='-', marker='o', markersize=4, label='每代最佳适应度')
    ax.plot(generations, history_avg, color='#55a868', linestyle='--', label='每代平均适应度')
    
    # 美化图表
    ax.set_title('遗传算法收敛曲线', fontsize=18, pad=20, weight='bold')
    ax.set_xlabel('迭代代数', fontsize=14)
    ax.set_ylabel('适应度 (7年平均总利润 / 元)', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # 格式化Y轴，显示为“xx千万”
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e7:,.1f} 千万'))
    
    ax.legend(fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"收敛曲线图已成功保存至: {output_path}")
    plt.close(fig)

# --- 4. 主程序 ---

if __name__ == '__main__':
    try:
        current_dir = Path(__file__).parent
        data_path_f1 = current_dir / '..'/'..'/'Data'/'附件1.xlsx'
        data_path_f2 = current_dir / '..'/'..'/'Data'/ '附件2.xlsx'
        output_dir = current_dir / 'results'
        output_dir.mkdir(parents=True, exist_ok=True)

        params = load_and_prepare_data(data_path_f1, data_path_f2)
        
        if params:
            # 构建一个简单但合理的相关性矩阵
            n_crops = len(params['J_crops'])
            corr_matrix = np.eye(n_crops) # 默认为单位矩阵
            # 您可以根据需要定义更复杂的相关性
            cov_matrix_L = np.linalg.cholesky(corr_matrix)
            
            # 运行遗传算法
            best_sol, best_fit, hist_best, hist_avg = run_genetic_algorithm(params, cov_matrix_L)
            
            # 【新增】调用绘图函数
            chart_output_path = output_dir / 'ga_convergence_curve.png'
            plot_ga_convergence(hist_best, hist_avg, chart_output_path)
            
            # (可选) 保存最终找到的最佳方案
            output_list = []
            for y in sorted(best_sol.keys()):
                for k in sorted(best_sol[y].keys()):
                    for i in sorted(best_sol[y][k].keys()):
                        crop = best_sol[y][k][i]
                        if crop:
                            output_list.append({
                                '年份': y, '季节': k, '地块编号': i, '作物名称': crop, 
                                '种植面积（亩）': params['P_area'][i]
                            })
            result_df = pd.DataFrame(output_list)
            result_df.to_excel(output_dir / 'result3_plan.xlsx', index=False)
            print(f"问题三的最优方案已保存至: {output_dir / 'result3_plan.xlsx'}")


    except Exception as e:
        print(f"\n程序主流程发生错误: {e}")
        import traceback
        traceback.print_exc()