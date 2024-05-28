import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 为了生成类似于根号x那种整体平滑曲线，可以使用移动平均的方法来平滑数据。移动平均可以消除噪声并保留整体趋势。
# 此代码通过使用 moving_average 方法对每个数据集进行平滑处理。您可以调整 window_size 参数来控制平滑程度。较大的 window_size 会产生更平滑的曲线，较小的 window_size 会保留更多的原始数据特征。


def moving_average(data, window_size):
    """计算移动平均"""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_csv_files(file_prefix_mapping, window_size=10):
    # 创建图和轴对象，指定图的大小
    fig, ax = plt.subplots(figsize=(12, 8))

    # 遍历文件和前缀映射字典
    for file, prefix in file_prefix_mapping.items():
        # 读取CSV文件
        df = pd.read_csv(file)
        
        # 获取x和y的原始数据
        x = df['n']
        y1 = df['accuracy_with_referenceAnswer']
        y2 = df['accuracy_with_referenceAnswerValue']
        
        # 计算移动平均值
        y1_smooth = moving_average(y1, window_size)
        y2_smooth = moving_average(y2, window_size)
        
        # 调整x轴数据的长度以匹配移动平均值的长度
        x_smooth = x[:len(y1_smooth)]
        
        # 绘制平滑后的数据
        ax.plot(x_smooth, y1_smooth, label=f'{prefix}_accuracy_with_referenceAnswer')
        ax.plot(x_smooth, y2_smooth, label=f'{prefix}_accuracy_with_referenceAnswerValue')

    # 设置x轴标签
    ax.set_xlabel('n')
    # 设置y轴标签
    ax.set_ylabel('Accuracy')
    # 设置图的标题
    ax.set_title('Accuracy with Reference Answer and Value')
    # 显示网格线
    ax.grid(True)

    # 调整图的布局以腾出图例的空间
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.8])

    # 在图的下方添加图例，并设置图例为两列
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)

    # 应用紧凑布局
    plt.tight_layout()
    # 保存图像，确保图例不会被裁剪
    plt.savefig('/workspace/dujh22/math_feedback/prm_evaluation/data/test1_1/combined_accuracy_plot3.png', bbox_inches='tight')
    # 显示图像
    plt.show()

if __name__ == "__main__":
    # 指定文件路径
    dir_path = '/workspace/dujh22/math_feedback/prm_evaluation/data/test1_1/'
    # 创建文件和前缀的映射字典
    file_prefix_mapping = {
        dir_path + 'test_rm3.csv': 'Train_llama3-8b-instruct-prm(prod)',
        dir_path + 'test_rm3V2.csv': 'Train_llama3-8b-instruct-prm(mean)',
        dir_path + 'test_rm3_mathshepherd_prm.csv': 'Open_Mistral-7b-prm(prod)',
        dir_path + 'test_rm3_mathshepherd_prm_mean.csv': 'Open_Mistral-7b-prm(mean)'
    }

    # 调用绘图函数
    plot_csv_files(file_prefix_mapping)
