import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.animation import FuncAnimation
from PIL import Image
import imageio
import torch.nn.functional as F
import math

# cmap RdBu_r  viridis
def visualize_autoregressive(input_frames, predictions, ground_truth, save_path, sample_idx=0):
    """
    可视化自回归预测结果（4行×T列布局）
    :param input_frames: 输入帧 [T_in, H, W] (numpy array)
    :param predictions: 预测序列 [T_total, H, W] (numpy array)
    :param ground_truth: 真实序列 [T_total, H, W] (numpy array)
    :param save_path: 保存路径
    :param sample_idx: 样本索引
    """
    total_frames = predictions.shape[0]
    input_length = input_frames.shape[0]

    # 创建4行×T列的子图
    fig, axes = plt.subplots(4, total_frames, figsize=(total_frames * 2, 8))
    fig.suptitle(f"Autoregressive Prediction - Sample {sample_idx}", fontsize=16)

    # 转换为张量计算误差
    predictions_tensor = torch.from_numpy(predictions)
    ground_truth_tensor = torch.from_numpy(ground_truth)

    # 计算该样本的整体 MSE 和 MAE
    error_tensor = torch.abs(predictions_tensor - ground_truth_tensor)
    mse = F.mse_loss(predictions_tensor, ground_truth_tensor).item()
    mae = error_tensor.mean().item()

    # 转换回numpy用于可视化
    error = error_tensor.numpy()

    # 设置统一颜色范围
    vmin = min(input_frames.min(), predictions.min(), ground_truth.min())
    vmax = max(input_frames.max(), predictions.max(), ground_truth.max())


    for t in range(total_frames):
        # 第一行：输入（仅前input_length帧有数据）
        if t < input_length:
            ax = axes[0, t]
            im0 = ax.imshow(input_frames[t], cmap='RdBu_r', vmin=vmin, vmax=vmax)
            ax.set_title(f"In {t}")
            ax.axis('off')
        else:
            # 对于预测帧，留空
            axes[0, t].axis('off')

        # 第二行：预测
        ax = axes[1, t]
        im1 = ax.imshow(predictions[t], cmap='RdBu_r', vmin=vmin, vmax=vmax)
        ax.set_title(f"Pred {t}")
        ax.axis('off')

        # 第三行：真实值
        ax = axes[2, t]
        im2 = ax.imshow(ground_truth[t], cmap='RdBu_r', vmin=vmin, vmax=vmax)
        ax.set_title(f"GT {t}")
        ax.axis('off')

        # 第四行：绝对误差
        ax = axes[3, t]
        im = ax.imshow(error[t], cmap='hot', vmin=0, vmax=error.max())
        ax.set_title(f"Err {t}")
        ax.axis('off')

    # 添加颜色条
    cbar_ax = fig.add_axes([0.9, 0.05, 0.001, 0.6])  # [left, bottom, width, height]
    fig.colorbar(im, cax=cbar_ax)

    # 在图底部中间显示 MSE / MAE
    fig.text(0.5, 0.01, f"MSE: {mse:.4f}    MAE: {mae:.4f}",
             ha='center', fontsize=12)

    plt.tight_layout(rect=[0, 0.03, 0.9, 1])
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved visualization to {save_path}")

# RdBu_r viridis
def visualize_trajectory(frames, save_path, title="visualize_trajectory", cmap='RdBu_r'):
    """
    可视化一个帧序列 (T, H, W)，并将其绘制在一个网格中，同时附带一个颜色条。

    :param frames: 一个形状为 [T, H, W] 的三维Numpy数组，其中 T 是帧数。
    :param save_path: 保存输出图像的完整路径（包含文件名）。
    :param title: 整个图表的主标题。
    :param cmap: 用于绘图的颜色映射 (例如, 'viridis', 'hot', 'RdBu_r')。
    """
    # 获取总帧数 (T)
    T = frames.shape[0]
    if T == 0:
        print("警告: 输入的 `frames` 为空，无法进行可视化。")
        return

    # --- 新增修改 1: 计算输入数据的全局最大值和最小值 ---
    vmin = frames.min()
    vmax = frames.max()
    print(vmin)
    print(vmax)

    # 计算一个最优的网格布局
    ncols = math.ceil(math.sqrt(T))
    nrows = math.ceil(T / ncols)

    # 创建图形和子图
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 2.5))
    if title:
        fig.suptitle(title, fontsize=16)

    # 为了便于一维迭代，将子图数组展平
    if T > 1:
        axes_flat = axes.flatten()
    else:
        axes_flat = [axes]

    # 循环遍历每一帧，并将其绘制在子图上
    for t in range(T):
        ax = axes_flat[t]
        # --- 新增修改 2: 在绘图时使用 vmin 和 vmax，并保存imshow的返回值 ---
        # im 对象对于后续创建颜色条至关重要
        im = ax.imshow(frames[t], cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"frame {t}")
        ax.axis('off')

    # 隐藏多余的、未使用的子图
    for t in range(T, len(axes_flat)):
        axes_flat[t].axis('off')

    # --- 新增修改 3: 创建并添加颜色条 ---
    # 定义颜色条的位置和大小: [左边距, 下边距, 宽度, 高度]
    # 这些值是相对于整个图窗的比例
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    # --- 新增修改 4: 调整整体布局以给颜色条留出空间 ---
    # rect 的最后一个参数 0.9 表示右边界只到图窗宽度的90%，留下右侧空间
    plt.tight_layout(rect=[0, 0.03, 0.9, 0.95])
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"轨迹可视化图像已保存至: {save_path}")

# 传入T*H*W的numpy数组，生成gif图像
def array_to_gif(input_array, output_path='output.gif', fps=20, dpi=100):
    """
    将3D NumPy数组(T,H,W)转换为高清GIF动画

    参数:
        input_array (np.ndarray): 输入数组，形状为(T,H,W)
        output_path (str): 输出GIF路径,默认'output.gif'
        fps (int): 帧率(帧/秒),默认10
        dpi (int): 输出分辨率(每英寸点数),默认100
    """
    # 验证输入数组
    if not isinstance(input_array, np.ndarray) or input_array.ndim != 3:
        raise ValueError("输入必须是3D NumPy数组 (T, H, W)")

    T, H, W = input_array.shape  # 获取帧数、高度和宽度

    # 创建图形和坐标轴,注意是先W后H
    fig, ax = plt.subplots(figsize=(W/dpi,H/dpi), dpi=dpi)
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # 初始化图像对象,修改cmap以改变颜色映射(gray)
    img = ax.imshow(input_array[0], cmap='viridis', vmin=input_array.min(), vmax=input_array.max())
    # img = ax.imshow(input_array[0], cmap='viridis', vmin=-1, vmax=1)

    # 更新函数
    def update(frame):
        img.set_array(input_array[frame])
        return [img]

    # 创建动画
    anim = FuncAnimation(fig, update, frames=T, interval=1000 / fps, blit=True)

    # 保存为GIF
    anim.save(output_path, writer='pillow', fps=fps, dpi=dpi)
    plt.close()

    # 使用imageio进一步优化(可选)
    gif = imageio.mimread(output_path)
    imageio.mimsave(output_path, gif, fps=fps,loop=0)

    print(f"GIF已保存至 {output_path} (尺寸: {W}x{H}, 帧数: {T}, 帧率: {fps}fps)")

# 传入T*H*W的numpy数组，生成gif图像,接受两个输入 array1为targets,array2为Predictions
def arrays_to_dual_gif(array1, array2, output_path='output.gif', fps=20, dpi=100, scale = 1,titles=('Targets', 'Predictions')):
    """
    将两个3D NumPy数组(T,H,W)合并为单个GIF,左右并排显示

    参数:
        array1 (np.ndarray): 第一个数组，形状(T,H,W)
        array2 (np.ndarray): 第二个数组，形状(T,H,W)
        output_path (str): 输出GIF路径
        fps (int): 帧率
        dpi (int): 分辨率
        titles (tuple): 两个子图的标题
    """
    # 验证输入
    for arr in (array1, array2):
        if not isinstance(arr, np.ndarray) or arr.ndim != 3:
            raise ValueError("输入必须是3D NumPy数组 (T, H, W)")

    # 确保两个数组时间长度相同
    if array1.shape[0] != array2.shape[0]:
        raise ValueError("两个数组的时间维度必须相同")

    T, H, W = array1.shape
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2 * W / dpi*scale, H / dpi*scale), dpi=dpi)

    # 设置子图
    for ax in (ax1, ax2):
        ax.axis('off')

    ax1.set_title(titles[0],fontsize=4,pad = 2)
    ax2.set_title(titles[1],fontsize=4,pad = 2)

    # 初始化图像
    img1 = ax1.imshow(array1[0], cmap='viridis', vmin=array1.min(), vmax=array1.max())
    img2 = ax2.imshow(array2[0], cmap='viridis', vmin=array2.min(), vmax=array2.max())

    def update(frame):
        img1.set_array(array1[frame])
        img2.set_array(array2[frame])
        return [img1, img2]

    # 创建动画
    anim = FuncAnimation(fig, update, frames=T, interval=1000 / fps, blit=True)
    anim.save(output_path, writer='pillow', fps=fps, dpi=dpi)
    plt.close()

    print(f"双视图GIF已保存至 {output_path}")