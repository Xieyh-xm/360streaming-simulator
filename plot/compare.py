import matplotlib.pyplot as plt
import numpy as np
from mplfonts.bin.cli import init
from mplfonts import use_font

init()

''' 绘制QoE比较的百分比堆积图 '''

"""
    stage   DCRL360<Baseline  DCRL360>Baseline
0   stage1  128               272
1   stage2  71                329

"""
data = {
    "Training stage": ["DCRL360-1st", "DCRL360"],
    "better": np.array([294, 337]),
    "worse": np.array([128, 71])
}

def plot_compare(version):
    x = data["Training stage"]
    y1 = 100 * data['better'] / (data['better'] + data['worse'])
    y2 = 100 * data['worse'] / (data['better'] + data['worse'])

    if version == "ENGLISH":
        plt.bar(x, y1, width=0.45, label='Better', color='#f9766e', edgecolor='black', lw=1.5, zorder=5)
        plt.bar(x, y2, width=0.45, bottom=y1, label='Worse', color='#00bfc4', edgecolor='black', lw=1.5, zorder=5)
    else:
        use_font()
        plt.bar(x, y1, width=0.45, label='更好', color='#f9766e', edgecolor='black', lw=1.5, zorder=5)
        plt.bar(x, y2, width=0.45, bottom=y1, label='更差', color='#00bfc4', edgecolor='black', lw=1.5, zorder=5)

    # plt.xlabel('Training stage', fontsize=12)
    plt.ylabel('% traces', fontsize=15)
    plt.legend(fontsize=15)
    plt.xticks(size=18)
    plt.yticks(size=12)
    plt.grid(axis="y", linestyle='-.', zorder=0)
    plt.legend(fontsize=15)
    # plt.show()
    if version == "ENGLISH":
        plt.savefig('../plot/figure/stage_compare.pdf', dpi=600, bbox_inches="tight")
    else:
        plt.savefig('../plot/figure/阶段比较.pdf', dpi=600, bbox_inches="tight")


if __name__ == '__main__':
    version = "CHINESE"
    # version = "ENGLISH"
    plot_compare(version)
