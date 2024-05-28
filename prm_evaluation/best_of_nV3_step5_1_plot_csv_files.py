import pandas as pd
import matplotlib.pyplot as plt

def plot_csv_files(file_prefix_mapping):
    # 创建图和轴对象，指定图的大小
    fig, ax = plt.subplots(figsize=(12, 8))

    # 遍历文件和前缀映射字典
    for file, prefix in file_prefix_mapping.items():
        # 读取CSV文件
        df = pd.read_csv(file)
        # 绘制accuracy_with_referenceAnswer列的数据
        ax.plot(df['n'], df['accuracy_with_referenceAnswer'], label=f'{prefix}_accuracy_with_referenceAnswer')
        # 绘制accuracy_with_referenceAnswerValue列的数据
        ax.plot(df['n'], df['accuracy_with_referenceAnswerValue'], label=f'{prefix}_accuracy_with_referenceAnswerValue')

    # 设置x轴标签
    ax.set_xlabel('n')
    # 设置y轴标签
    ax.set_ylabel('Accuracy')
    # 设置图的标题
    ax.set_title('Accuracy with Reference Answer and Value')
    # 显示网格线
    ax.grid(True)

    # 调整图的布局以腾出图例的空间
    # 获取当前图的位置
    box = ax.get_position()
    # 设置新的图位置
    # box.x0 和 box.y0 是图的左下角的坐标
    # box.width 和 box.height 是图的宽度和高度
    # 调整图的位置以给图例留出空间
    # 这里将图的高度减少20%（0.8），以在图下方留出空间
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.8])

    # 在图的下方添加图例，并设置图例为两列
    # loc='upper center' 将图例放置在图的上方中心
    # bbox_to_anchor 参数指定图例的锚点位置，这里设置为图的下方
    # ncol=2 设置图例为两列
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)

    # 应用紧凑布局
    plt.tight_layout()
    # 保存图像，确保图例不会被裁剪
    plt.savefig('/workspace/dujh22/math_feedback/prm_evaluation/data/test1_1/combined_accuracy_plot.png', bbox_inches='tight')
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
