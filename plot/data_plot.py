import os
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator, FuncFormatter
from collections import Counter
import matplotlib.patches as patches
import matplotlib as mpl

def get_bbox_sizes(xml_path):
    """
    从XML文件中提取标注框的宽度、高度和类别
    返回: sizes列表 [(width, height, class_name), ...]
    """
    sizes = []
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # 获取图像原始尺寸
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    
    for obj in root.findall('object'):
        name = obj.find('name').text
        bbox = obj.find('bndbox')
        
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        
        # 计算框的宽度和高度
        box_width = xmax - xmin
        box_height = ymax - ymin
        
        sizes.append((box_width, box_height, name))
    
    return sizes, (width, height)

def normalize_sizes(sizes, original_size, target_size=(1920, 1080)):
    """
    将框尺寸归一化到目标尺寸
    
    Args:
        sizes: 原始尺寸列表 [(width, height, class_name), ...]
        original_size: 原始图像尺寸 (width, height)
        target_size: 目标尺寸，默认1920x1080
        
    Returns:
        normalized_sizes: 归一化后的尺寸列表
    """
    orig_width, orig_height = original_size
    target_width, target_height = target_size
    
    # 计算缩放比例
    width_scale = target_width / orig_width
    height_scale = target_height / orig_height
    
    # 对所有框尺寸进行转换
    normalized_sizes = []
    for w, h, class_name in sizes:
        norm_w = w * width_scale
        norm_h = h * height_scale
        normalized_sizes.append((norm_w, norm_h, class_name))
    
    return normalized_sizes

def get_fixed_colors():
    """
    返回固定的颜色列表，使用明亮的16进制颜色代码
    """
    return [
        '#FF3366',     # 亮玫红
        '#33B4FF',     # 天空蓝
        '#66CC33',     # 草绿色
        '#FF9933',     # 明橙色
        '#9966FF',     # 亮紫色
        '#FF6699',     # 粉红色
        '#33CCCC',     # 碧绿色
        '#FF9966',     # 珊瑚色
        '#99CC33',     # 柠檬绿
        '#6699FF'      # 天蓝色
    ]

def plot_box_sizes(xml_folder, output_folder, font_size):
    """
    处理文件夹中的所有XML文件并生成框尺寸分布图
    """
    os.makedirs(output_folder, exist_ok=True)
    
    all_sizes = []
    target_size = (1920, 1080)  # 目标尺寸
    max_box_size = 0  # 记录最大框尺寸
    
    # 收集所有框尺寸数据并归一化
    for filename in os.listdir(xml_folder):
        if filename.endswith('.xml'):
            xml_path = os.path.join(xml_folder, filename)
            sizes, img_size = get_bbox_sizes(xml_path)
            normalized_sizes = normalize_sizes(sizes, img_size, target_size)
            all_sizes.extend(normalized_sizes)
            
            # 更新最大框尺寸
            for w, h, _ in normalized_sizes:
                size = max(w, h)
                if size > max_box_size:
                    max_box_size = size
    
    # 统计每个类别的数量并排序
    class_counts = Counter(size[2] for size in all_sizes)
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    
    # 获取固定的颜色列表
    colors = get_fixed_colors()
    
    # 创建类别到颜色的映射
    class_to_color = {}
    for i, (class_name, _) in enumerate(sorted_classes):
        if i < len(colors):
            class_to_color[class_name] = colors[i]
        else:
            class_to_color[class_name] = 'darkgray'
    
    # 设置全局字体
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = font_size - 3
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
    
    # 按照类别数量从多到少的顺序绘制散点
    for class_name, count in sorted_classes:
        points = [(w, h) for w, h, name in all_sizes if name == class_name]
        if points:
            w, h = zip(*points)
            ax.scatter(w, h, c=class_to_color[class_name], 
                      label=f'{class_name} ({count})', alpha=0.7, s=20, edgecolors='none')
    
    # 添加参考线
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    
    # 设置坐标轴标签
    ax.set_xlabel('Bounding Box Width (pixels)', fontsize=font_size)
    ax.set_ylabel('Bounding Box Height (pixels)', fontsize=font_size)
    
    # 设置坐标轴范围
    max_val = max_box_size * 1.1
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    
    # 设置网格和刻度
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # 动态设置刻度间隔
    tick_interval = 200 if max_val < 1000 else 500
    ax.xaxis.set_major_locator(MultipleLocator(tick_interval))
    ax.yaxis.set_major_locator(MultipleLocator(tick_interval))
    
    # 添加比例线
    ratios = [0.5, 1, 2, 5]  # 宽高比
    for ratio in ratios:
        x = np.linspace(0, max_val, 100)
        y = x * ratio
        ax.plot(x, y, '--', color='gray', alpha=0.5, linewidth=1)
        ax.text(max_val, max_val * ratio, f'{ratio}:1', 
               verticalalignment='top', fontsize=font_size-5)
    
    # 添加图例
    ax.legend(loc='upper right', frameon=True, fontsize=font_size-5,
              title='Object Classes', title_fontsize=font_size-4,
              bbox_to_anchor=(1.0, 1.0), bbox_transform=ax.transAxes)
    
    # 添加标题
    plt.title('Distribution of Bounding Box Sizes', fontsize=font_size+2, pad=15)
    
    # 添加统计信息
    total_boxes = len(all_sizes)
    avg_width = sum(w for w, h, c in all_sizes) / total_boxes
    avg_height = sum(h for w, h, c in all_sizes) / total_boxes
    avg_ratio = avg_width / avg_height
    
    stats_text = f'Total Boxes: {total_boxes}\nAvg Width: {avg_width:.1f}px\nAvg Height: {avg_height:.1f}px\nAvg W/H Ratio: {avg_ratio:.2f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=font_size-4, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    output_path = os.path.join(output_folder, 'bbox_size_distribution.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Bounding box size distribution plot has been saved to: {output_path}")

def plot_size_distribution_histogram(xml_folder, output_folder, font_size):
    """
    生成框尺寸分布的直方图
    """
    os.makedirs(output_folder, exist_ok=True)
    
    all_sizes = []
    target_size = (1920, 1080)  # 目标尺寸
    
    # 收集所有框尺寸数据并归一化
    for filename in os.listdir(xml_folder):
        if filename.endswith('.xml'):
            xml_path = os.path.join(xml_folder, filename)
            sizes, img_size = get_bbox_sizes(xml_path)
            normalized_sizes = normalize_sizes(sizes, img_size, target_size)
            all_sizes.extend(normalized_sizes)
    
    # 提取宽度和高度
    widths = [w for w, h, c in all_sizes]
    heights = [h for w, h, c in all_sizes]
    
    # 设置全局字体
    plt.rcParams['font.family'] = 'Times New Roman'
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # 绘制宽度分布直方图
    ax1.hist(widths, bins=50, color='#33B4FF', alpha=0.7, edgecolor='white')
    ax1.set_ylabel('Count', fontsize=font_size)
    ax1.set_title('Bounding Box Width Distribution', fontsize=font_size)
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # 绘制高度分布直方图
    ax2.hist(heights, bins=50, color='#FF9933', alpha=0.7, edgecolor='white')
    ax2.set_xlabel('Size (pixels)', fontsize=font_size)
    ax2.set_ylabel('Count', fontsize=font_size)
    ax2.set_title('Bounding Box Height Distribution', fontsize=font_size)
    ax2.grid(True, linestyle='--', alpha=0.3)
    
    # 添加总框数
    total_boxes = len(all_sizes)
    fig.suptitle(f'Bounding Box Size Distribution (Total: {total_boxes} boxes)', fontsize=font_size+2)
    
    output_path = os.path.join(output_folder, 'bbox_size_histogram.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Bounding box size histogram has been saved to: {output_path}")

if __name__ == "__main__":
    # 配置参数
    FONT_SIZE = 18
    XML_FOLDER = r"/home/hipeson/ymn/dataset/no_Ho/ImageSets/annotations"
    OUTPUT_FOLDER = r"/home/hipeson/ymn/dataset/no_Ho/ImageSets"    
    # 生成散点图
    plot_box_sizes(XML_FOLDER, OUTPUT_FOLDER, FONT_SIZE)    
    # 生成直方图
    plot_size_distribution_histogram(XML_FOLDER, OUTPUT_FOLDER, FONT_SIZE)
