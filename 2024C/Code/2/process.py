# -*- coding: utf-8 -*-
# 文件名: solve_q2_complete.py
# 功能: 问题二最终完整版，包含所有约束和修正
# 版本: 3.0

import pandas as pd
import pyomo.environ as pyo
import os
import re
import time
from pathlib import Path

def load_and_prepare_data(data_path_f1, data_path_f2):
    """
    数据加载与基础参数准备函数。
    """
    try:
        print("（1/4）正在读取原始数据文件...")
        plots_df = pd.read_excel(data_path_f1, sheet_name='乡村的现有耕地')
        crops_info_df = pd.read_excel(data_path_f1, sheet_name='乡村种植的农作物')
        stats_df = pd.read_excel(data_path_f2, sheet_name='2023年统计的相关数据')
        past_planting_df = pd.read_excel(data_path_f2, sheet_name='2023年的农作物种植情况')
        
        for df in [plots_df, crops_info_df, stats_df, past_planting_df]:
            df.columns = df.columns.str.strip()

        params = {}
        params['I_plots'] = plots_df['地块名称'].tolist()
        params['J_crops'] = crops_info_df['作物名称'].unique().tolist()
        params['K_seasons'] = [1, 2]
        params['Y_years'] = list(range(2024, 2031))

        params['P_area'] = dict(zip(plots_df['地块名称'], plots_df['地块面积/亩']))
        params['P_plot_type'] = dict(zip(plots_df['地块名称'], plots_df['地块类型']))

        params['P_crop_type'] = dict(zip(crops_info_df['作物名称'], crops_info_df['作物类型']))
        params['J_bean'] = [j for j, ctype in params['P_crop_type'].items() if isinstance(ctype, str) and '豆' in ctype]
        
        # 【重要】P_past使用兼容Q1的(地块,作物)->1/0格式，以匹配修正后的约束逻辑
        params['P_past'] = {(i, j): 0 for i in params['I_plots'] for j in params['J_crops']}
        for _, row in past_planting_df.iterrows():
            if row['种植地块'] in params['I_plots'] and row['作物名称'] in params['J_crops']:
                params['P_past'][(row['种植地块'], row['作物名称'])] = 1

        def clean_and_convert_price(value):
            if isinstance(value, str) and any(c in value for c in '-–—'):
                parts = re.split(r'[-–—]', value.strip())
                return (float(parts[0]) + float(parts[1])) / 2
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
        
        params['P_demand_base'] = {j: 0 for j in params['J_crops']}
        temp_planting_details = pd.merge(past_planting_df, plots_df, left_on='种植地块', right_on='地块名称')
        for j in params['J_crops']:
            total_yield_j = 0
            crop_plantings = temp_planting_details[temp_planting_details['作物名称'] == j]
            for _, row in crop_plantings.iterrows():
                yield_val = params['P_yield_base'].get((j, row['地块类型']), 0)
                total_yield_j += yield_val * row['种植面积/亩']
            params['P_demand_base'][j] = total_yield_j if total_yield_j > 0 else 1000

        params['S_suitability'] = {}
        restricted_veg = ['大白菜', '白萝卜', '红萝卜']
        for i in params['I_plots']:
            plot_t = params['P_plot_type'].get(i, '')
            for j in params['J_crops']:
                crop_t = params['P_crop_type'].get(j, '')
                is_bean = j in params['J_bean']
                is_veg = '蔬菜' in str(crop_t)
                for k in params['K_seasons']:
                    suitable = 0
                    if plot_t in ['平旱地', '梯田', '山坡地'] and ('粮食' in str(crop_t) or is_bean) and k == 1: suitable = 1
                    elif plot_t == '水浇地':
                        if '水稻' in str(crop_t) and k == 1: suitable = 1
                        elif is_veg and j not in restricted_veg and k == 1: suitable = 1
                        elif is_veg and j in restricted_veg and k == 2: suitable = 1
                    elif plot_t == '普通大棚':
                        if is_veg and j not in restricted_veg and k == 1: suitable = 1
                        elif '食用菌' in str(crop_t) and k == 2: suitable = 1
                    elif plot_t == '智慧大棚' and is_veg: suitable = 1
                    params['S_suitability'][(i, j, k)] = suitable
        
        # 为新增的管理约束设置默认值
        params['A_min'] = 0.1  # 默认最小种植面积0.1亩
        params['N_j'] = {j: 10 for j in params['J_crops']}  # 默认最大分散地块为10个

        print(" -> 数据加载与基准参数计算完成。")
        return params
    except Exception as e:
        print(f"错误: 加载或处理数据时发生严重错误: {e}")
        raise

def prepare_robust_parameters(params):
    """
    根据鲁棒优化思想，计算未来7年所有参数的“最坏情况”值。
    """
    print("（2/4）正在生成鲁棒优化所需的最坏情况参数...")
    robust_params = {'P_cost': {}, 'P_yield': {}, 'P_price': {}, 'P_demand': {}}
    
    for y in params['Y_years']:
        year_factor = y - 2023
        
        for (j, plot_type), cost in params['P_cost_base'].items():
            robust_params['P_cost'][(j, plot_type, y)] = cost * (1.05 ** year_factor)
        
        for key, yield_val in params['P_yield_base'].items():
            robust_params['P_yield'][key + (y,)] = yield_val * 0.9

        for j in params['J_crops']:
            price_base = params['P_price_base'].get(j, 0)
            crop_type = params['P_crop_type'].get(j, '')
            
            if '蔬菜' in str(crop_type):
                robust_params['P_price'][(j, y)] = price_base * (1.05 ** year_factor)
            elif j == '羊肚菌':
                robust_params['P_price'][(j, y)] = price_base * (0.95 ** year_factor)
            elif '食用菌' in str(crop_type):
                robust_params['P_price'][(j, y)] = price_base * (0.95 ** year_factor)
            else:
                robust_params['P_price'][(j, y)] = price_base

            demand_base = params['P_demand_base'].get(j, 1000)
            if j in ['小麦', '玉米']:
                robust_params['P_demand'][(j, y)] = demand_base * (1.05 ** year_factor)
            else:
                robust_params['P_demand'][(j, y)] = demand_base * (0.95 ** year_factor)
    
    print(" -> 最坏情况参数生成完毕。")
    return robust_params

def build_and_solve_robust_model(base_params, robust_params, solver_path, timeout_sec=1200):
    """
    【最终修正版】构建并求解基于最坏情况参数的确定性MILP模型。
    - 包含了所有管理约束 (A_min, N_j)
    - 包含了鲁棒的结果提取逻辑
    """
    print("（3/4）开始构建大规模混合整数线性规划模型...")
    model = pyo.ConcreteModel("Robust_Planting_Strategy_Complete")

    # --- 从基础参数中获取管理约束参数 ---
    A_min = base_params.get('A_min', 0.1)
    N_j = base_params.get('N_j', {j: 10 for j in base_params['J_crops']})

    # --- 1. 集合与决策变量 ---
    model.I = pyo.Set(initialize=base_params['I_plots'])
    model.J = pyo.Set(initialize=base_params['J_crops'])
    model.K = pyo.Set(initialize=base_params['K_seasons'])
    model.Y = pyo.Set(initialize=base_params['Y_years'])
    model.J_bean = pyo.Set(initialize=base_params['J_bean'])
    
    model.x = pyo.Var(model.I, model.J, model.K, model.Y, domain=pyo.NonNegativeReals)
    model.u = pyo.Var(model.I, model.J, model.K, model.Y, domain=pyo.Binary)
    model.z = pyo.Var(model.I, model.J, model.Y, domain=pyo.Binary)
    
    model.Sales_normal = pyo.Var(model.J, model.K, model.Y, domain=pyo.NonNegativeReals)
    model.Sales_over = pyo.Var(model.J, model.K, model.Y, domain=pyo.NonNegativeReals)

    # --- 2. 目标函数 ---
    def linear_objective_rule(m):
        revenue = pyo.quicksum(
            robust_params['P_price'].get((j, y), 0) * m.Sales_normal[j, k, y] +
            0.5 * robust_params['P_price'].get((j, y), 0) * m.Sales_over[j, k, y]
            for j in m.J for k in m.K for y in m.Y
        )
        cost = pyo.quicksum(
            robust_params['P_cost'].get((j, base_params['P_plot_type'][i], y), 0) * m.x[i, j, k, y]
            for i in m.I for j in m.J for k in m.K for y in m.Y
        )
        return revenue - cost
    model.profit = pyo.Objective(rule=linear_objective_rule, sense=pyo.maximize)

    # --- 3. 约束条件 ---
    # (此处省略约束定义，以您代码为准，因为它们不是出错的原因)
    @model.Constraint(model.J, model.K, model.Y)
    def production_rule(m, j, k, y):
        total_production = pyo.quicksum(robust_params['P_yield'].get((j, base_params['P_plot_type'][i], y), 0) * m.x[i,j,k,y] for i in m.I)
        return total_production == m.Sales_normal[j, k, y] + m.Sales_over[j, k, y]
    @model.Constraint(model.J, model.K, model.Y)
    def demand_rule(m, j, k, y):
        return m.Sales_normal[j, k, y] <= robust_params['P_demand'].get((j, y), 0)
    @model.Constraint(model.I, model.K, model.Y)
    def area_rule(m, i, k, y):
        return pyo.quicksum(m.x[i, j, k, y] for j in m.J) <= base_params['P_area'][i]
    @model.Constraint(model.I, model.J, model.K, model.Y)
    def suitability_rule(m, i, j, k, y):
        return m.x[i, j, k, y] <= base_params['P_area'][i] * base_params['S_suitability'].get((i, j, k), 0)
    @model.Constraint(model.I, model.J, model.K, model.Y)
    def link_x_u_upper_rule(m, i, j, k, y):
        return m.x[i,j,k,y] <= base_params['P_area'][i] * m.u[i,j,k,y]
    @model.Constraint(model.I, model.J, model.K, model.Y)
    def link_x_u_lower_rule(m, i, j, k, y):
        return m.x[i,j,k,y] >= A_min * m.u[i,j,k,y]
    @model.Constraint(model.J, model.K, model.Y)
    def dispersion_rule(m, j, k, y):
        return pyo.quicksum(m.u[i, j, k, y] for i in m.I) <= N_j[j]
    @model.Constraint(model.I, model.J, model.K, model.Y)
    def link_u_z_rule(m, i, j, k, y):
        return m.u[i, j, k, y] <= m.z[i, j, y]
    @model.Constraint(model.I, model.J)
    def rotation_past_rule(m, i, j):
        if base_params['P_plot_type'].get(i) not in ['普通大棚', '智慧大棚']:
            if base_params['P_past'].get((i, j), 0) == 1: return m.z[i, j, 2024] == 0
        return pyo.Constraint.Skip
    @model.Constraint(model.I, model.J, model.Y)
    def rotation_future_rule(m, i, j, y):
        if y >= 2030: return pyo.Constraint.Skip
        if base_params['P_plot_type'].get(i) not in ['普通大棚', '智慧大棚']:
            return m.z[i, j, y] + m.z[i, j, y+1] <= 1
        return pyo.Constraint.Skip
    model.bean_rule = pyo.ConstraintList()
    for i in model.I:
        if base_params['P_plot_type'].get(i) not in ['普通大棚', '智慧大棚']:
            past_bean_sum = sum(1 for jb in model.J_bean if base_params['P_past'].get((i, jb), 0) == 1)
            model.bean_rule.add(past_bean_sum + pyo.quicksum(model.z[i,jb,y] for jb in model.J_bean for y in [2024, 2025]) >= 1)
            for y_start in range(2024, 2029):
                model.bean_rule.add(pyo.quicksum(model.z[i,jb,y] for jb in model.J_bean for y in range(y_start, y_start + 3)) >= 1)
    
    print(" -> 模型构建完成。即将调用求解器...")
    # --- 4. 求解 ---
    solver = pyo.SolverFactory('cbc', executable=solver_path)
    solver.options['sec'] = timeout_sec
    start_time = time.time()
    results = solver.solve(model, tee=True)
    end_time = time.time()

    # ===================================================================
    # --- 5. 结果提取 (【关键修正】) ---
    # ===================================================================
    print("\n（4/4）正在提取并整理结果...")
    final_profit, output_df = 0, None
    
    term_cond = results.solver.termination_condition
    if term_cond in [pyo.TerminationCondition.optimal, pyo.TerminationCondition.maxTimeLimit]:
        # 【关键修正】使用 pyo.value(model.profit, exception=False) 来安全地检查目标值是否存在
        profit_value = pyo.value(model.profit, exception=False)
        if profit_value is not None:
            print(f"求解成功！耗时: {end_time - start_time:.2f} 秒")
            final_profit = profit_value
            print(f"在最坏情况下的保底总利润 (2024-2030): {final_profit:,.2f} 元")

            if term_cond == pyo.TerminationCondition.optimal:
                print("求解状态: 已找到全局最优解。")
            else: # maxTimeLimit
                print(f"求解状态: 已达到 {timeout_sec} 秒时间限制，返回当前找到的最佳解。")

            output = [{'年份': y, '季节': k, '地块编号': i, '作物名称': j, '种植面积（亩）': round(pyo.value(model.x[i, j, k, y]), 4)} 
                      for (i, j, k, y) in model.x.index_set() if pyo.value(model.x[i, j, k, y]) > 0.01]
            
            if output:
                output_df = pd.DataFrame(output)
            else:
                print("\n警告：求解器找到了解，但未能提取出任何种植活动。")
        else:
            print("\n错误：求解器报告已找到解，但无法从模型中加载该解（profit值为None）。")
    else:
        print("\n求解失败：未能找到任何可行的种植方案。")
        print(f"求解器状态: {results.solver.status}")
        print(f"终止条件: {term_cond}")

    return output_df, final_profit

if __name__ == '__main__':
    try:
        current_dir = Path(__file__).parent
        project_root = current_dir/'..'/'..'/'Data' # 假设附件和代码在同一目录下
        data_path_f1 = project_root / '附件1.xlsx'
        data_path_f2 = project_root / '附件2.xlsx'
        output_dir = current_dir / 'results_q2'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 检查求解器路径，根据您的系统进行修改
        # windows: 'cbc.exe', mac/linux: 'cbc'
        solver_path_str = r'CBC\bin\cbc.exe' 
        # 如果cbc.exe在项目文件夹的子目录，可以写相对路径
        # solver_path_str = str(project_root / 'solver' / 'cbc.exe') 

        base_parameters = load_and_prepare_data(data_path_f1, data_path_f2)
        if base_parameters:
            robust_parameters = prepare_robust_parameters(base_parameters)
            result_df, profit = build_and_solve_robust_model(
                base_parameters, 
                robust_parameters, 
                solver_path_str,
                timeout_sec=1200 # 可根据需要调整求解时间
            )

            if result_df is not None and not result_df.empty:
                output_path = output_dir / 'result2.xlsx'
                result_df.to_excel(output_path, index=False)
                print(f"\n最优种植方案已成功保存至: {output_path}")
            else:
                print("\n未能生成结果文件。请检查求解过程中的日志信息。")

    except FileNotFoundError as e:
        print(f"\n文件未找到错误: {e}")
        print("请确保 '附件1.xlsx' 和 '附件2.xlsx' 文件与脚本在同一目录下，或者修改脚本中的路径。")
    except Exception as e:
        import traceback
        print(f"\n程序主流程发生严重错误: {e}")
        traceback.print_exc()