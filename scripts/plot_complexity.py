"""
理论复杂度可视化 - Baseline vs Hough算法

绘制时间和空间复杂度随hits数N的变化曲线
"""
import numpy as np
import matplotlib.pyplot as plt


def baseline_time_complexity(N, k=10, L=5):
    """Baseline算法时间复杂度估算"""
    edge_building = N**2 / L  # 边构建
    chain_building = N * k**2  # 链构建
    C = min(N, 10)  # 假设候选数
    post_processing = C * np.log(C + 1)  # 后处理
    return edge_building + chain_building + post_processing


def hough_time_complexity(N, P=10, H_ratio=0.3, B=80):
    """Hough算法时间复杂度估算"""
    voting = N**2  # 投票
    peak_finding = B**2  # 峰值查找（固定）
    H = max(int(N * H_ratio), 5)  # 每个峰值的hits数
    clustering = P * H**2  # 聚类
    C = min(N, 10)
    post_processing = C * np.log(C + 1)
    return voting + peak_finding + clustering + post_processing


def baseline_space_complexity(N, k=10):
    """Baseline算法空间复杂度"""
    return N * k


def hough_space_complexity(N, B=80):
    """Hough算法空间复杂度"""
    accumulator = B**2
    pair_bins = N**2 * 0.5  # 假设50%配对通过斜率过滤
    return accumulator + pair_bins


def plot_complexity():
    """绘制复杂度对比图"""
    N_values = np.arange(10, 201, 5)
    
    # 计算复杂度
    baseline_time = np.array([baseline_time_complexity(N) for N in N_values])
    hough_time = np.array([hough_time_complexity(N) for N in N_values])
    
    baseline_space = np.array([baseline_space_complexity(N) for N in N_values])
    hough_space = np.array([hough_space_complexity(N) for N in N_values])
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # ========== 图1: 时间复杂度（线性刻度）==========
    ax = axes[0, 0]
    ax.plot(N_values, baseline_time, label='Baseline', linewidth=2.5, color='blue')
    ax.plot(N_values, hough_time, label='Hough', linewidth=2.5, color='red')
    ax.set_xlabel('Number of hits (N)', fontsize=12)
    ax.set_ylabel('Time complexity (ops)', fontsize=12)
    ax.set_title('Time Complexity Comparison', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # ========== 图2: 时间复杂度（对数刻度）==========
    ax = axes[0, 1]
    ax.plot(N_values, baseline_time, label='Baseline', linewidth=2.5, color='blue')
    ax.plot(N_values, hough_time, label='Hough', linewidth=2.5, color='red')
    ax.set_xlabel('Number of hits (N)', fontsize=12)
    ax.set_ylabel('Time complexity (ops)', fontsize=12)
    ax.set_yscale('log')
    ax.set_title('Time Complexity (log scale)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which='both')
    
    # ========== 图3: 空间复杂度 ==========
    ax = axes[1, 0]
    ax.plot(N_values, baseline_space, label='Baseline', linewidth=2.5, color='blue')
    ax.plot(N_values, hough_space, label='Hough', linewidth=2.5, color='red')
    ax.set_xlabel('Number of hits (N)', fontsize=12)
    ax.set_ylabel('Space complexity (units)', fontsize=12)
    ax.set_title('Space Complexity Comparison', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # ========== 图4: 速度比 ==========
    ax = axes[1, 1]
    speedup = hough_time / baseline_time
    ax.plot(N_values, speedup, linewidth=2.5, color='green')
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5, label='Equal speed')
    ax.fill_between(N_values, 1, speedup, where=(speedup >= 1), 
                     alpha=0.3, color='red', label='Hough slower')
    ax.fill_between(N_values, 1, speedup, where=(speedup < 1), 
                     alpha=0.3, color='blue', label='Baseline slower')
    ax.set_xlabel('Number of hits (N)', fontsize=12)
    ax.set_ylabel('Speedup (Hough time / Baseline time)', fontsize=12)
    ax.set_title('Relative Speed', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存
    output_path = '/Users/IvanTang/hep/data_Run502/outputs/complexity_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Complexity comparison plot saved to: {output_path}")
    plt.close()
    
    # 打印关键统计
    print("\n" + "="*70)
    print("Theoretical Complexity Analysis")
    print("="*70)
    
    # 找到交叉点
    speedup = hough_time / baseline_time
    crossover_idx = np.where(np.diff(np.sign(speedup - 1)))[0]
    
    if len(crossover_idx) > 0:
        N_cross = N_values[crossover_idx[0]]
        print(f"\nCrossover point: N ≈ {N_cross}")
        print(f"  → Baseline faster when N > {N_cross}")
        print(f"  → Hough faster when N < {N_cross}")
    else:
        if speedup[0] > 1:
            print(f"\nBaseline is consistently faster across all N")
        else:
            print(f"\nHough is consistently faster across all N")
    
    # N=50的比较（典型值）
    idx_50 = np.argmin(np.abs(N_values - 50))
    N_50 = N_values[idx_50]
    print(f"\nAt N = {N_50} (typical event):")
    print(f"  Baseline time:  {baseline_time[idx_50]:,.0f} ops")
    print(f"  Hough time:     {hough_time[idx_50]:,.0f} ops")
    print(f"  Speedup:        {speedup[idx_50]:.2f}x")
    print(f"  Baseline space: {baseline_space[idx_50]:,.0f} units")
    print(f"  Hough space:    {hough_space[idx_50]:,.0f} units")
    print(f"  Space ratio:    {hough_space[idx_50] / baseline_space[idx_50]:.1f}x")
    
    # N=100的比较（高密度）
    idx_100 = np.argmin(np.abs(N_values - 100))
    N_100 = N_values[idx_100]
    print(f"\nAt N = {N_100} (high multiplicity):")
    print(f"  Baseline time:  {baseline_time[idx_100]:,.0f} ops")
    print(f"  Hough time:     {hough_time[idx_100]:,.0f} ops")
    print(f"  Speedup:        {speedup[idx_100]:.2f}x")
    print(f"  Baseline space: {baseline_space[idx_100]:,.0f} units")
    print(f"  Hough space:    {hough_space[idx_100]:,.0f} units")
    print(f"  Space ratio:    {hough_space[idx_100] / baseline_space[idx_100]:.1f}x")


def plot_breakdown():
    """绘制各阶段复杂度分解"""
    N_values = np.arange(10, 201, 10)
    
    # Baseline分解
    baseline_edge = N_values**2 / 5
    baseline_chain = N_values * 100  # k²=100
    baseline_post = 10 * np.log(11)  # 假设C=10
    
    # Hough分解
    hough_voting = N_values**2
    hough_peak = 6400 * np.ones_like(N_values)  # B²=6400
    H_vals = (N_values * 0.3).astype(int)
    H_vals[H_vals < 5] = 5
    hough_clustering = 10 * H_vals**2  # P=10
    hough_post = 10 * np.log(11)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Baseline分解
    ax1.fill_between(N_values, 0, baseline_edge, 
                      alpha=0.6, label='Edge building O(N²/L)', color='lightblue')
    ax1.fill_between(N_values, baseline_edge, baseline_edge + baseline_chain,
                      alpha=0.6, label='Chain building O(N·k²)', color='lightgreen')
    ax1.fill_between(N_values, baseline_edge + baseline_chain, 
                      baseline_edge + baseline_chain + baseline_post,
                      alpha=0.6, label='Post-processing O(C log C)', color='lightyellow')
    ax1.set_xlabel('Number of hits (N)', fontsize=12)
    ax1.set_ylabel('Operations', fontsize=12)
    ax1.set_yscale('log')
    ax1.set_title('Baseline Algorithm - Stage Breakdown', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, which='both')
    
    # Hough分解
    ax2.fill_between(N_values, 0, hough_voting,
                      alpha=0.6, label='Voting O(N²)', color='lightcoral')
    ax2.fill_between(N_values, hough_voting, hough_voting + hough_peak,
                      alpha=0.6, label='Peak finding O(B²)', color='lightgreen')
    ax2.fill_between(N_values, hough_voting + hough_peak, 
                      hough_voting + hough_peak + hough_clustering,
                      alpha=0.6, label='Clustering O(P·H²)', color='lightyellow')
    ax2.fill_between(N_values, hough_voting + hough_peak + hough_clustering,
                      hough_voting + hough_peak + hough_clustering + hough_post,
                      alpha=0.6, label='Post-processing O(C log C)', color='lightgray')
    ax2.set_xlabel('Number of hits (N)', fontsize=12)
    ax2.set_ylabel('Operations', fontsize=12)
    ax2.set_yscale('log')
    ax2.set_title('Hough Algorithm - Stage Breakdown', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    
    output_path = '/Users/IvanTang/hep/data_Run502/outputs/complexity_breakdown.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nStage breakdown plot saved to: {output_path}")
    plt.close()


if __name__ == "__main__":
    print("Plotting theoretical complexity curves...\n")
    plot_complexity()
    print("\n" + "="*70)
    plot_breakdown()
    
    print("\n" + "="*70)
    print("Note: These are theoretical estimates based on Big-O analysis.")
    print("Run compare_timing.py for actual measured performance.")
    print("="*70)
