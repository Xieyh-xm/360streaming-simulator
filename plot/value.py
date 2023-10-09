import matplotlib.pyplot as plt
from mplfonts.bin.cli import init
from mplfonts import use_font

init()

def draw_value(version, avg_value, dataset):
    # 写入data
    # Quality(kbps) ↑	Var space(kbps) ↓	Var time(kbps) ↓	Stall time(ms) ↓	QoE ↑
    if version == "ENGLISH":
        labels = ['Quality(x10)', 'Spatial var(x10)', 'Temporal var(x10)', 'Rebuffering']
    else:
        labels = ['平均质量(x10)', '空间平滑度(x10)', '时间平滑度(x10)', '卡顿时长']
    # 调整除stall time的data
    for i in range(len(avg_value)):
        for j in range(len(avg_value[0]) - 1):
            avg_value[i][j] /= 10

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
    plt.yticks(fontsize=13)
    # plt.legend(bbox_to_anchor=(0.5, 1.), loc=8, ncol=10, fontsize=13)
    plt.legend(bbox_to_anchor=(0.5, 1.), loc=8, ncol=10, fontsize=15)

    plt.tight_layout()
    plt.grid(axis="y", linestyle='-.', zorder=0)
    # plt.title("QoE in {}".format(dataset))
    if version == "ENGLISH":
        plt.xticks(x_list, labels=labels, horizontalalignment='center', fontsize=15.5)
        plt.ylabel("Average value", fontsize=15.5)
        plt.savefig("./figure/value_in_{}.pdf".format(dataset), bbox_inches="tight")
    else:
        use_font()
        plt.xticks(x_list, labels=labels, horizontalalignment='center', fontsize=15.5)
        plt.ylabel("各指标均值", fontsize=15.5)
        plt.savefig("./figure/均值-{}.pdf".format(dataset), bbox_inches="tight")
    plt.show()


# quality	Var_space	Var_time	Stall_time

PROPOSED_in_fcc = [420.32, 109.59, 105.10, 1.33]
RAM360_in_fcc = [405.545, 197.775, 103.772, 1.440]
TTS_in_fcc = [313.414, 79.081, 90.530, 0.475]
STS_in_fcc = [375.299, 99.724, 80.230, 4.266]
BS_in_fcc = [50.000, 0.000, 0.000, 0.475]

PROPOSED_in_norway = [383.82, 99.54, 95.55, 2.94]
RAM360_in_norway = [361.889, 175.083, 104.171, 3.423]
TTS_in_norway = [281.063, 69.050, 85.499, 0.987]
STS_in_norway = [316.808, 82.208, 76.271, 22.274]
BS_in_norway = [50.000, 0.000, 0.000, 0.607]

PROPOSED_in_5G = [567.18, 138.32, 56.68, 0.25]
RAM360_in_5G = [469.29, 155.65, 50.56, 0.25]
TTS_in_5G = [542.59, 140.05, 69.87, 0.58]
STS_in_5G = [536.92, 142.55, 63.10, 5.11]
BS_in_5G = [49.58, 0.00, 0.00, 0.25]

plt.rcParams['figure.figsize'] = (9.0, 4.0)

if __name__ == '__main__':
    version = "CHINESE"
    # version = "ENGLISH"
    avg_value = [BS_in_norway, STS_in_norway, TTS_in_norway, RAM360_in_norway, PROPOSED_in_norway]
    draw_value(version, avg_value, dataset="HSDPA")

    avg_value = [BS_in_fcc, STS_in_fcc, TTS_in_fcc, RAM360_in_fcc, PROPOSED_in_fcc]
    draw_value(version, avg_value, dataset="Broadband")

    avg_value = [BS_in_5G, STS_in_5G, TTS_in_5G, RAM360_in_5G, PROPOSED_in_5G]
    draw_value(version, avg_value, dataset="5G")
