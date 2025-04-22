import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import argparse
from datetime import datetime
from pathlib import Path

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决保存图像时负号'-'显示为方块的问题

class ResultAnalyzer:
    def __init__(self, csv_path):
        """初始化分析器"""
        self.csv_path = csv_path
        
        # 读取CSV文件
        try:
            self.df = pd.read_csv(csv_path)
            print(f"成功加载数据: {csv_path}")
            print(f"共有 {len(self.df)} 条测试记录")
        except Exception as e:
            print(f"加载CSV文件时出错: {e}")
            sys.exit(1)
        
        # 获取当前脚本的绝对路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 创建结果目录
        self.output_dir = os.path.join(current_dir, "figures/test")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 生成输出文件名前缀
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = Path(csv_path).stem
        self.output_prefix = os.path.join(self.output_dir, f"{csv_filename}_{timestamp}")
    
    def generate_basic_stats(self):
        """生成基本统计信息"""
        # 计算成功率
        success_rate = self.df['success'].mean() * 100
        
        # 计算平均悬停时间
        avg_hover_time = self.df['hover_time'].mean()
        
        # 计算平均恢复稳定时间(只考虑成功恢复稳定的试验)
        stabilization_df = self.df[self.df['stabilization_time'] > 0]
        avg_stabilization_time = stabilization_df['stabilization_time'].mean() if len(stabilization_df) > 0 else 0
        
        # 计算平均位置误差
        avg_position_error = self.df['position_error_total'].mean()
        
        # 打印基本统计信息
        print("\n=== 基本统计信息 ===")
        print(f"成功率: {success_rate:.2f}%")
        print(f"平均悬停时间: {avg_hover_time:.2f} 秒")
        print(f"平均恢复稳定时间: {avg_stabilization_time:.2f} 秒")
        print(f"平均位置误差: {avg_position_error:.4f} 米")
        
        # 创建统计信息的表格图
        fig, ax = plt.figure(figsize=(10, 3)), plt.subplot(111)
        
        stats = {
            '指标': ['成功率', '平均悬停时间', '平均恢复稳定时间', '平均位置误差'],
            '值': [f"{success_rate:.2f}%", f"{avg_hover_time:.2f} 秒", 
                  f"{avg_stabilization_time:.2f} 秒", f"{avg_position_error:.4f} 米"]
        }
        
        # 隐藏轴线
        ax.axis('tight')
        ax.axis('off')
        
        # 创建表格
        table = ax.table(cellText=list(zip(stats['指标'], stats['值'])),
                          colLabels=['指标', '值'],
                          loc='center',
                          cellLoc='center')
        
        # 调整表格样式
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        
        plt.title('模型测试基本统计信息', fontsize=16)
        plt.tight_layout()
        
        # 保存图表
        plt.savefig(f"{self.output_prefix}_basic_stats.png", dpi=300, bbox_inches='tight')
        print(f"基本统计信息已保存至 {self.output_prefix}_basic_stats.png")
        plt.close()
        
        return {
            'success_rate': success_rate,
            'avg_hover_time': avg_hover_time,
            'avg_stabilization_time': avg_stabilization_time,
            'avg_position_error': avg_position_error
        }
    
    def plot_wind_effect(self):
        """分析风力对性能的影响"""
        if 'wind_strength' not in self.df.columns or self.df['wind_strength'].max() == 0:
            print("数据中没有风力信息或风力均为0，跳过风力分析")
            return
        
        # 创建图形，只包含两个子图
        fig, axs = plt.subplots(1, 2, figsize=(16, 7))
        
        # 1. 风力强度与位置误差的关系
        ax = axs[0]
        sns.regplot(x='wind_strength', y='position_error_total', data=self.df, ax=ax,
                     scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'})
        ax.set_title('风力强度与位置误差的关系', fontsize=14)
        ax.set_xlabel('风力强度 (牛顿)', fontsize=12)
        ax.set_ylabel('位置误差 (米)', fontsize=12)
        
        # 2. 风力强度与姿态误差的关系
        ax = axs[1]
        sns.regplot(x='wind_strength', y='average_attitude_error', data=self.df, ax=ax,
                     scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'})
        ax.set_title('风力强度与姿态误差的关系', fontsize=14)
        ax.set_xlabel('风力强度 (牛顿)', fontsize=12)
        ax.set_ylabel('姿态误差 (弧度)', fontsize=12)
        
        # 设置全局标题
        fig.suptitle('风力对四旋翼悬停性能的影响', fontsize=18)
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # 调整布局，为全局标题留出空间
        
        # 保存图表
        plt.savefig(f"{self.output_prefix}_wind_effect.png", dpi=300, bbox_inches='tight')
        print(f"风力影响分析已保存至 {self.output_prefix}_wind_effect.png")
        plt.close()
    
    def plot_error_distribution(self):
        """绘制误差分布图"""
        # 创建图形和子图
        fig, axs = plt.subplots(2, 2, figsize=(16, 14))
        
        # 1. 位置误差的直方图
        ax = axs[0, 0]
        sns.histplot(self.df['position_error_total'], kde=True, ax=ax)
        ax.set_title('位置误差分布', fontsize=14)
        ax.set_xlabel('位置误差 (米)', fontsize=12)
        ax.set_ylabel('频次', fontsize=12)
        
        # 2. 各坐标轴误差的箱线图
        ax = axs[0, 1]
        error_df = pd.DataFrame({
            'X轴': self.df['position_error_x'].abs(),
            'Y轴': self.df['position_error_y'].abs(),
            'Z轴': self.df['position_error_z'].abs()
        })
        sns.boxplot(data=error_df, ax=ax)
        ax.set_title('各坐标轴位置误差分布', fontsize=14)
        ax.set_ylabel('位置误差 (米)', fontsize=12)
        
        # 3. 速度误差与姿态误差的散点图
        ax = axs[1, 0]
        sns.scatterplot(x='average_velocity_error', y='average_attitude_error', 
                        hue='success', data=self.df, ax=ax, palette={0: 'red', 1: 'green'})
        ax.set_title('速度误差与姿态误差的关系', fontsize=14)
        ax.set_xlabel('平均速度误差 (米/秒)', fontsize=12)
        ax.set_ylabel('平均姿态误差 (弧度)', fontsize=12)
        ax.legend(title='是否成功', labels=['失败', '成功'])
        
        # 4. 悬停时间的直方图
        ax = axs[1, 1]
        sns.histplot(self.df['hover_time'], kde=True, ax=ax)
        ax.set_title('悬停时间分布', fontsize=14)
        ax.set_xlabel('悬停时间 (秒)', fontsize=12)
        ax.set_ylabel('频次', fontsize=12)
        
        # 设置全局标题
        fig.suptitle('四旋翼悬停性能误差分析', fontsize=18)
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # 调整布局，为全局标题留出空间
        
        # 保存图表
        plt.savefig(f"{self.output_prefix}_error_distribution.png", dpi=300, bbox_inches='tight')
        print(f"误差分布分析已保存至 {self.output_prefix}_error_distribution.png")
        plt.close()
    
    def plot_performance_metrics(self):
        """绘制性能指标的关系图"""
        # 创建相关性矩阵
        metrics = ['hover_time', 'stabilization_time', 'position_error_total', 
                  'average_velocity_error', 'average_attitude_error']
        
        # 替换-1的stabilization_time(未恢复稳定)为NaN，以便计算相关性
        corr_df = self.df.copy()
        corr_df.loc[corr_df['stabilization_time'] < 0, 'stabilization_time'] = np.nan
        
        # 计算相关性矩阵
        corr_matrix = corr_df[metrics].corr()
        
        # 创建图形
        plt.figure(figsize=(14, 12))
        
        # 绘制相关性热图
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
                    linewidths=0.5, fmt='.2f')
        plt.title('性能指标相关性矩阵', fontsize=16)
        
        # 调整标签
        metric_labels = {
            'hover_time': '悬停时间',
            'stabilization_time': '恢复稳定时间',
            'position_error_total': '位置误差',
            'average_velocity_error': '平均速度误差',
            'average_attitude_error': '平均姿态误差'
        }
        
        plt.xticks(np.arange(len(metrics)) + 0.5, [metric_labels[m] for m in metrics], rotation=45)
        plt.yticks(np.arange(len(metrics)) + 0.5, [metric_labels[m] for m in metrics], rotation=0)
        
        plt.tight_layout()
        
        # 保存图表
        plt.savefig(f"{self.output_prefix}_performance_correlation.png", dpi=300, bbox_inches='tight')
        print(f"性能指标相关性分析已保存至 {self.output_prefix}_performance_correlation.png")
        plt.close()
    
    def plot_success_factors(self):
        """分析影响成功率的因素"""
        # 创建图形和子图
        fig, axs = plt.subplots(2, 2, figsize=(16, 14))
        
        # 1. 位置误差与成功率的关系
        ax = axs[0, 0]
        error_bins = pd.cut(self.df['position_error_total'], bins=10)
        success_by_error = self.df.groupby(error_bins)['success'].mean() * 100
        
        # 获取位置误差区间的中点
        bin_centers = [(interval.left + interval.right) / 2 for interval in success_by_error.index]
        
        ax.bar(bin_centers, success_by_error.values, width=(bin_centers[1]-bin_centers[0])*0.8 if len(bin_centers) > 1 else 0.1)
        ax.set_title('位置误差与成功率的关系', fontsize=14)
        ax.set_xlabel('位置误差 (米)', fontsize=12)
        ax.set_ylabel('成功率 (%)', fontsize=12)
        ax.set_ylim(0, 100)
        
        # 2. 速度误差与成功率的关系
        ax = axs[0, 1]
        vel_bins = pd.cut(self.df['average_velocity_error'], bins=10)
        success_by_vel = self.df.groupby(vel_bins)['success'].mean() * 100
        
        # 获取速度误差区间的中点
        bin_centers = [(interval.left + interval.right) / 2 for interval in success_by_vel.index]
        
        ax.bar(bin_centers, success_by_vel.values, width=(bin_centers[1]-bin_centers[0])*0.8 if len(bin_centers) > 1 else 0.1)
        ax.set_title('速度误差与成功率的关系', fontsize=14)
        ax.set_xlabel('平均速度误差 (米/秒)', fontsize=12)
        ax.set_ylabel('成功率 (%)', fontsize=12)
        ax.set_ylim(0, 100)
        
        # 3. 姿态误差与成功率的关系
        ax = axs[1, 0]
        att_bins = pd.cut(self.df['average_attitude_error'], bins=10)
        success_by_att = self.df.groupby(att_bins)['success'].mean() * 100
        
        # 获取姿态误差区间的中点
        bin_centers = [(interval.left + interval.right) / 2 for interval in success_by_att.index]
        
        ax.bar(bin_centers, success_by_att.values, width=(bin_centers[1]-bin_centers[0])*0.8 if len(bin_centers) > 1 else 0.1)
        ax.set_title('姿态误差与成功率的关系', fontsize=14)
        ax.set_xlabel('平均姿态误差 (弧度)', fontsize=12)
        ax.set_ylabel('成功率 (%)', fontsize=12)
        ax.set_ylim(0, 100)
        
        # 4. 恢复稳定时间分布
        ax = axs[1, 1]
        # 只考虑成功恢复稳定的试验
        stab_df = self.df[self.df['stabilization_time'] > 0]
        if len(stab_df) > 0:
            sns.histplot(stab_df['stabilization_time'], kde=True, ax=ax)
            ax.set_title('恢复稳定时间分布', fontsize=14)
            ax.set_xlabel('恢复稳定时间 (秒)', fontsize=12)
            ax.set_ylabel('频次', fontsize=12)
        else:
            ax.text(0.5, 0.5, '没有成功恢复稳定的试验', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=14)
            ax.set_title('恢复稳定时间分布', fontsize=14)
        
        # 设置全局标题
        fig.suptitle('影响四旋翼悬停成功率的因素分析', fontsize=18)
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # 调整布局，为全局标题留出空间
        
        # 保存图表
        plt.savefig(f"{self.output_prefix}_success_factors.png", dpi=300, bbox_inches='tight')
        print(f"成功率因素分析已保存至 {self.output_prefix}_success_factors.png")
        plt.close()
    
    def run_all_analyses(self):
        """运行所有分析"""
        print("开始进行数据分析...")
        
        # 生成基本统计信息
        self.generate_basic_stats()
        
        # 绘制风力影响图
        self.plot_wind_effect()
        
        # 绘制误差分布图
        self.plot_error_distribution()
        
        # 绘制性能指标相关性图
        self.plot_performance_metrics()
        
        # 分析影响成功率的因素
        self.plot_success_factors()
        
        print(f"所有分析完成，图表已保存至 {self.output_dir} 目录")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="分析四旋翼悬停测试结果")
    parser.add_argument("csv_path", type=str, help="CSV文件路径")
    
    args = parser.parse_args()
    
    # 检查CSV文件是否存在
    if not os.path.exists(args.csv_path):
        print(f"错误: 找不到CSV文件 '{args.csv_path}'")
        sys.exit(1)
    
    # 创建分析器并运行分析
    analyzer = ResultAnalyzer(args.csv_path)
    analyzer.run_all_analyses() 