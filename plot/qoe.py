import matplotlib.pyplot as plt
from mplfonts.bin.cli import init
from mplfonts import use_font

init()

def draw_qoe(version, avg_value):
    # 写入data
    if version == "ENGLISH":
        labels = ['Broadband', 'HSDPA', '5G']
    else:
        labels = ['宽带网络', 'LTE网络', '5G网络']

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


    plt.legend(bbox_to_anchor=(0.5, 1.), loc=8, ncol=10, fontsize=13)

    plt.tight_layout()
    plt.grid(axis="y", linestyle='-.', zorder=0)
    if version == "ENGLISH":
        plt.xticks(x_list, labels=labels, horizontalalignment='center', fontsize=16)
        plt.ylabel("Average QoE", fontsize=13)
        plt.savefig("./figure/overall_qoe.pdf", dpi=1000, bbox_inches="tight")
    else:
        use_font()
        plt.xticks(x_list, labels=labels, horizontalalignment='center', fontsize=16)
        plt.ylabel("平均QoE", fontsize=13)
        plt.savefig("./figure/不同网络下的QoE比较.pdf", dpi=1000, bbox_inches="tight")
    plt.show()


# FCC Norway

PROPOSED_qoe = [392.18, 349.59, 546.41]
RAM360_qoe = [368.191, 316.848, 447.40]
TTS_qoe = [294.079, 260.676, 518.70]
STS_qoe = [335.974, 274.388, 490.80]
BS_qoe = [47.205, 46.543, 48.31]

plt.rcParams['figure.figsize'] = (9.0, 4.0)

if __name__ == '__main__':
    version = "CHINESE"
    # version = "ENGLISH"
    avg_value = [BS_qoe, STS_qoe, TTS_qoe, RAM360_qoe, PROPOSED_qoe]
    draw_qoe(version, avg_value)
