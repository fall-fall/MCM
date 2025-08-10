# -*- coding: utf-8 -*-
# 文件名: plot_strategy_chart.py
# 功能: (独立脚本) 对比分析两种方案的年均作物组合策略，并绘制堆叠条形图

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_data_for_strategy_plot(current_dir):
    """加载绘图所需的数据文件"""
    try:
        # 加载作物分类信息
        crops_info_df = pd.read_excel(current_dir / '附件1.xlsx', sheet_name='乡村种植的农作物')
        
        # 加载两种方案
        plan1_df = pd.read_excel(current_dir / 'result1_2.xlsx')
        plan2_df = pd.read_excel(current_dir / 'result2.xlsx')
        
        print("所有必需的数据文件加载成功。")
        return crops_info_df, plan1_df, plan2_df

    except FileNotFoundError as e:
        print(f"错误：找不到必需的数据文件 {e}。")
        print("请确保 '附件1.xlsx', 'result1_2.xlsx', 'result2.xlsx' 文件与脚本在同一目录下。")
        return None, None, None

def process_planting_plan(plan_df, crop_info_df):
    """处理种植方案，计算各类作物的年均种植面积"""
    
    # 1. 创建作物名称到类别的映射
    crop_to_category = {}
    for _, row in crop_info_df.iterrows():
        name = row['作物名称']
        crop_type = str(row['作物类型'])
        if '粮食' in crop_type:
            crop_to_category[name] = '粮食作物'
        elif '蔬菜' in crop_type:
            crop_to_category[name] = '蔬菜作物'
        elif '食用菌' in crop_type:
            crop_to_category[name] = '食用菌'
        else:
            # 默认为粮食类（例如豆类在某些分类下算杂粮）
            crop_to_category[name] = '粮食作物'
            
    # 2. 将类别映射到方案数据中
    plan_df['作物类别'] = plan_df['作物名称'].map(crop_to_category)

    # 3. 计算每类作物的总种植面积
    # 我们需要按年份和类别分组求和，然后计算7年的平均值
    total_years = plan_df['年份'].nunique()
    
    # 计算每年的总种植面积
    yearly_area = plan_df.groupby(['年份', '作物类别'])['种植面积（亩）'].sum().unstack(fill_value=0)
    
    # 计算年均种植面积
    avg_area_per_category = yearly_area.mean()
    
    return avg_area_per_category

def plot_strategy_comparison_chart(data_dict, output_path):
    """
    根据处理后的数据，绘制学术风格的堆叠条形图
    """
    print("正在生成作物组合策略对比图...")

    # 遵照您的要求，设置学术风格图表
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = ['SimHei'] # 设置中文字体
    plt.rcParams['axes.unicode_minus'] = False # 正常显示负号

    labels = list(data_dict.keys())
    categories = ['粮食作物', '蔬菜作物', '食用菌']
    colors = ['#4c72b0', '#55a868', '#c44e52'] # 专业蓝、绿、橙红色系
    
    # 修正后
    plot_data = pd.DataFrame(data_dict).T.reindex(categories, axis=1, fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 10))

    # 绘制堆叠条形图
    bottom = np.zeros(len(labels))
    for i, category in enumerate(categories):
        values = plot_data[category].values
        bars = ax.bar(labels, values, label=category, bottom=bottom, color=colors[i], width=0.6)
        bottom += values
        
        # 在每个色块上添加数据标签
        for bar in bars:
            yval = bar.get_height()
            xpos = bar.get_x() + bar.get_width() / 2
            bottom_pos = bar.get_y()
            # 计算百分比
            total_height_for_bar = plot_data.loc[labels[int(round(xpos))]].sum()
            percentage = yval / total_height_for_bar * 100
            
            # 只有当色块高度足够时才显示标签，防止重叠
            if yval > 0:
                ax.text(xpos, bottom_pos + yval / 2, f"{yval:.0f} 亩\n({percentage:.1f}%)", 
                        ha='center', va='center', color='white', fontsize=12, weight='bold')

    # --- 图表美化 ---
    ax.set_ylabel('年均总种植面积 (亩)', fontsize=14)
    ax.set_title('确定性与鲁棒方案的年均作物组合策略对比', fontsize=18, pad=20, weight='bold')
    
    # 调整Y轴刻度，使其更美观
    ax.set_ylim(0, bottom.max() * 1.1)
    
    # 调整X轴标签
    ax.tick_params(axis='x', labelsize=14)
    
    # 美化图例
    ax.legend(title='作物类别', fontsize=12, title_fontsize=13, bbox_to_anchor=(1.02, 1), loc='upper left')

    # 移除顶部和右侧的边框线，使图表更简洁
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # 调整布局为图例留出空间
    
    # 保存图表
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"策略对比图已成功保存至: {output_path}")
    plt.close(fig)


if __name__ == '__main__':
    # 假设所有文件都在脚本的同一级目录中
    current_dir = Path(__file__).parent/'..'/'..'/'Data'
    
    # 1. 加载数据
    crop_info_df, plan1_df, plan2_df = load_data_for_strategy_plot(current_dir)
    
    if crop_info_df is not None:
        # 2. 处理数据
        avg_area_plan1 = process_planting_plan(plan1_df, crop_info_df)
        avg_area_plan2 = process_planting_plan(plan2_df, crop_info_df)

        plot_data = {
            '确定性方案': avg_area_plan1,
            '鲁棒方案': avg_area_plan2
        }

        # 3. 绘图
        output_path = current_dir / "2-3.png"
        plot_strategy_comparison_chart(plot_data, output_path)