import numpy as np
import matplotlib.pyplot as plt


def draw_value(avg_value, std_value, dataset):
    # 写入data
    # Quality(kbps) ↑	Var space(kbps) ↓	Var time(kbps) ↓	Stall time(ms) ↓	QoE ↑
    labels = ['Quality(x10)', 'Spatial var(x10)', 'Temporal var(x10)', 'Rebuffering']
    # 调整除stall time的data
    for i in range(len(avg_value)):
        for j in range(len(avg_value[0]) - 1):
            avg_value[i][j] /= 10
            std_value[i][j] /= 10

    # 设置柱形的间隔
    width = 0.18  # 柱形的宽度
    x_list, x1_list, x2_list, x3_list, x4_list, x5_list = [], [], [], [], [], []
    for i in range(4):
        x1_list.append(i)
        x2_list.append(i + width)
        x_list.append(i + 2.75 * width)
        x3_list.append(i + 2 * width)
        x4_list.append(i + 3 * width)
        x5_list.append(i + 4 * width)

    # 创建图层
    fig, ax1 = plt.subplots()

    ax1.bar(x1_list, avg_value[0], width=width, label='BS', color='#3366FF', align='edge',
            hatch='xx', edgecolor='k',
            zorder=100)
    # ax1.errorbar(x1_list, avg_value[0], yerr=std_value[0], fmt='-o')
    ax1.bar(x2_list, avg_value[1], width=width, label='STS', color='#8DE529', align='edge',
            hatch=r'\\', edgecolor='k',
            zorder=100)
    ax1.bar(x3_list, avg_value[2], width=width, label='TTS', color='#3FC9FC', align='edge',
            hatch='--', edgecolor='k',
            zorder=100)
    ax1.bar(x4_list, avg_value[3], width=width, label='RAM360', color='#FFD311', align='edge', hatch='//',
            edgecolor='k', zorder=100)
    ax1.bar(x5_list, avg_value[4], width=width, label='DCRL360', color='#FF6721', align='edge', edgecolor='k',
            zorder=100)

    plt.xticks(x_list, labels=labels, horizontalalignment='center', fontsize=15)
    plt.legend(bbox_to_anchor=(0.5, 1.), loc=8, ncol=10, fontsize=13)

    plt.tight_layout()
    plt.grid(axis="y", linestyle='-.', zorder=0)
    # plt.title("QoE in {}".format(dataset))
    plt.ylabel("Average value", fontsize=15)
    plt.savefig("./figure/value_in_{}.pdf".format(dataset), bbox_inches="tight")
    plt.show()


# quality	Var_space	Var_time	Stall_time

PROPOSED_in_fcc = [421.928, 109.430, 88.921, 1.004]
RAM360_in_fcc = [405.545, 197.775, 103.772, 1.440]
TTS_in_fcc = [313.414, 79.081, 90.530, 0.475]
STS_in_fcc = [375.299, 99.724, 80.230, 4.266]
BS_in_fcc = [50.000, 0.000, 0.000, 0.475]

PROPOSED_std_in_fcc = [113.715, 46.447, 23.975, 1.901]
RAM360_std_in_fcc = [116.332, 43.679, 26.143, 2.399]
TTS_std_in_fcc = [86.862, 35.883, 22.812, 0.397]
STS_std_in_fcc = [99.431, 42.204, 18.328, 8.913]
BS_std_in_fcc = [0.000, 0.000, 0.000, 0.397]

PROPOSED_in_norway = [384.414, 98.527, 85.471, 3.369]
RAM360_in_norway = [361.889, 175.083, 104.171, 3.423]
TTS_in_norway = [281.063, 69.050, 85.499, 0.987]
STS_in_norway = [316.808, 82.208, 76.271, 22.274]
BS_in_norway = [50.000, 0.000, 0.000, 0.607]

PROPOSED_std_in_norway = [107.151, 45.622, 21.599, 8.386]
RAM360_std_in_norway = [103.555, 39.776, 28.644, 7.838]
TTS_std_in_norway = [78.909, 33.495, 22.402, 2.560]
STS_std_in_norway = [86.671, 38.399, 18.573, 47.803]
BS_std_in_norway = [0.000, 0.000, 0.000, 0.677]

plt.rcParams['figure.figsize'] = (9.0, 4.0)

if __name__ == '__main__':
    avg_value = [BS_in_norway, STS_in_norway, TTS_in_norway, RAM360_in_norway, PROPOSED_in_norway]
    std_value = [BS_std_in_norway, STS_std_in_norway, TTS_std_in_norway, RAM360_std_in_norway, PROPOSED_std_in_norway]
    draw_value(avg_value, std_value, dataset="HSDPA")

    avg_value = [BS_in_fcc, STS_in_fcc, TTS_in_fcc, RAM360_in_fcc, PROPOSED_in_fcc]
    std_value = [BS_std_in_fcc, STS_std_in_fcc, TTS_std_in_fcc, RAM360_std_in_fcc, PROPOSED_std_in_fcc]
    draw_value(avg_value, std_value, dataset="Broadband")


