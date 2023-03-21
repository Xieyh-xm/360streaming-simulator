import numpy as np
import matplotlib.pyplot as plt


def draw_qoe(avg_value):
    # 写入data
    labels = ['Low', 'Medium', 'High']

    # 设置柱形的间隔
    width = 0.18  # 柱形的宽度
    x_list, x1_list, x2_list, x3_list, x4_list, x5_list = [], [], [], [], [], []
    for i in range(3):
        x1_list.append(1.5 * i)
        x2_list.append(1.5 * i + width)
        x_list.append(1.5 * i + 2.75 * width)
        x3_list.append(1.5 * i + 2 * width)
        x4_list.append(1.5 * i + 3 * width)
        x5_list.append(1.5 * i + 4 * width)

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

    plt.xticks(x_list, labels=labels, horizontalalignment='center', fontsize=16)
    plt.legend(bbox_to_anchor=(0.5, 1.), loc=8, ncol=10, fontsize=13)

    plt.tight_layout()
    plt.grid(axis="y", linestyle='-.', zorder=0)
    # plt.title("QoE in {}".format(dataset))
    plt.ylabel("Average QoE", fontsize=13)
    plt.savefig("./figure/qoe_in_diff_net.pdf", dpi=1000, bbox_inches="tight")
    plt.show()


# FCC Norway

PROPOSED_qoe = [300.0389874, 201.3453373, 153.7831649]
RAM360_qoe = [234.1366, 173.4347927, 83.99505393]
TTS_qoe = [218.18706, 153.0745273, 100.2073635]
STS_qoe = [238.2409437, 171.369722, 71.4227296]
BS_qoe = [48.67653576, 47.85678362, 48.63499584]

plt.rcParams['figure.figsize'] = (9.0, 4.0)

if __name__ == '__main__':
    avg_value = [BS_qoe, STS_qoe, TTS_qoe, RAM360_qoe, PROPOSED_qoe]
    # std_value = [BS_std, STS_std, TTS_std, RAM360_std, PROPOSED_std]
    draw_qoe(avg_value)
