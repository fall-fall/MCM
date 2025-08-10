# ==============================================================================
# 0. 导入所有需要的库
# ==============================================================================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import sys

# --- 中文字体设置 ---
# 根据操作系统设置合适的字体
if sys.platform.startswith('win'):
    # Windows系统
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
elif sys.platform.startswith('darwin'):
    # Mac OS系统
    plt.rcParams['font.sans-serif'] = ['PingFang SC']
elif sys.platform.startswith('linux'):
    # Linux系统，需要确认已安装中文字体，如'WenQuanYi Zen Hei'
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
else:
    print("未知的操作系统，请手动设置中文字体")
print("--- 所有库已成功导入 ---")

# ==============================================================================
# 1. 数据预处理
# ==============================================================================
print("\n--- [阶段一] 开始数据预处理 ---")

# 定义一个标志，用于判断数据是否成功加载
data_loaded_successfully = False

try:
    # (1) 加载与探查 -【修正】修改文件路径
    df_stroke = pd.read_csv('data/stroke.csv')
    print(f"成功加载 data/stroke.csv，数据集包含 {df_stroke.shape[0]} 行 和 {df_stroke.shape[1]} 列。")

    # (2) 数据清洗与转换
    df_stroke_cleaned = df_stroke.copy()
    df_stroke_cleaned = df_stroke_cleaned[df_stroke_cleaned['gender'] != 'Other']
    df_stroke_cleaned = df_stroke_cleaned.drop('id', axis=1)
    bmi_median = df_stroke_cleaned['bmi'].median()
    df_stroke_cleaned['bmi'].fillna(bmi_median, inplace=True)
    print("数据清洗完成：移除了异常值，删除了id列，填充了bmi缺失值。")

    # (3) 分类变量编码
    categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    df_stroke_processed = pd.get_dummies(df_stroke_cleaned, columns=categorical_cols, drop_first=True)
    print("分类变量已转换为独热编码格式。")
    print("--- [阶段一] 数据预处理完成 ---\n")
    
    # 如果代码成功运行到这里，设置标志为True
    data_loaded_successfully = True

except FileNotFoundError:
    print("错误: 文件 'data/stroke.csv' 未找到。请再次确认你的文件结构是否正确。")
except Exception as e:
    print(f"处理过程中出现错误: {e}")


# ==============================================================================
# 2. 探索性数据分析 (EDA) - 【优化】只有在数据加载成功后才执行
# ==============================================================================
if data_loaded_successfully:
    print("\n--- [阶段二] 开始探索性数据分析 (EDA) ---")

    # --- 设置全局可视化风格 ---
    sns.set_theme(style="ticks", palette="muted")

    # --- (1) 单变量分析 ---
    print("正在生成：单变量分析图表...")
    numerical_features = ['age', 'avg_glucose_level', 'bmi']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Distribution of Numerical Features in Stroke Dataset', fontsize=16, fontweight='bold')
    for i, feature in enumerate(numerical_features):
        sns.histplot(df_stroke_processed[feature], kde=True, ax=axes[i])
        axes[i].set_title(f'Distribution of {feature.replace("_", " ").title()}', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    plt.figure(figsize=(6, 4))
    ax = sns.countplot(x='stroke', data=df_stroke_processed)
    plt.title('Distribution of Stroke Outcome', fontsize=16, fontweight='bold')
    plt.xlabel('Stroke Status (0: No, 1: Yes)', fontsize=12)
    plt.ylabel('Count of Patients', fontsize=12)
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 5), textcoords='offset points')
    sns.despine()
    plt.show()

    # --- (2) 双变量分析 ---
    print("正在生成：双变量分析图表...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Numerical Features vs. Stroke Status', fontsize=16, fontweight='bold')
    for i, feature in enumerate(numerical_features):
        sns.boxplot(x='stroke', y=feature, data=df_stroke_processed, ax=axes[i], palette="pastel")
        axes[i].set_title(f'{feature.replace("_", " ").title()} by Stroke Status', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    original_categorical_cols = ['gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    fig.suptitle('Stroke Rate by Categorical Features', fontsize=20, fontweight='bold')
    for i, col in enumerate(original_categorical_cols):
        stroke_rate = df_stroke_cleaned.groupby(col)['stroke'].mean().sort_values(ascending=False) * 100
        sns.barplot(x=stroke_rate.index, y=stroke_rate.values, ax=axes[i], palette='viridis')
        axes[i].set_title(f'Stroke Rate by {col.replace("_", " ").title()}', fontsize=12)
        axes[i].set_ylabel('Stroke Rate (%)', fontsize=10)
        axes[i].set_xlabel('')
        axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, horizontalalignment='right')
    axes[-1].set_visible(False)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    # --- (3) 多变量分析 ---
    print("正在生成：多变量分析图表 (相关性热力图)...")
    correlation_matrix = df_stroke_processed.corr()
    plt.figure(figsize=(16, 12))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=.5)
    plt.title('Correlation Matrix of All Features in Stroke Dataset', fontsize=18, fontweight='bold')
    plt.show()

    print("\n--- [阶段二] 探索性数据分析 (EDA) 完成 ---")
else:
    print("\n由于数据加载失败，无法进行探索性数据分析。")