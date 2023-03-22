import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

''' 绘制QoE比较的百分比堆积图 '''

"""
    stage   DCRL360<Baseline  DCRL360>Baseline
0   stage1  106               294
1   stage2  63                337

"""
data = {
    "Training stage": ["Stage 1", "Stage 2"],
    "better": np.array([294, 337]),
    "worse": np.array([106, 63])
}


def plot_compare():
    x = data["Training stage"]
    y1 = data['better'] / (data['better'] + data['worse'])
    y2 = data['worse'] / (data['better'] + data['worse'])

    plt.bar(x, y1, width=0.4, label='Better', color='#f9766e', edgecolor='grey', zorder=5)
    plt.bar(x, y2, width=0.4, bottom=y1, label='Worse', color='#00bfc4', edgecolor='grey', zorder=5)

    plt.xlabel('Training stage', fontsize=12)
    plt.ylabel('%', fontsize=12)
    plt.legend()
    # plt.show()
    plt.savefig('../plot/figure/stage_compare.pdf', dpi=600)

if __name__ == '__main__':
    plot_compare()
