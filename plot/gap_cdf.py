import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
import xlrd
from mplfonts.bin.cli import init
from mplfonts import use_font

init()

''' 用于绘制gap的cdf曲线 '''


# stage1 stage2

def read_excel(file_path):
    workbook = xlrd.open_workbook(file_path)
    worksheet = workbook.sheet_by_index(0)
    col_data_1 = worksheet.col_values(0)  # none
    col_data_2 = worksheet.col_values(1)  # 1st
    col_data_3 = worksheet.col_values(2)  # 2nd
    col_data_4 = worksheet.col_values(3)  # DCRL
    return col_data_1[1:], col_data_2[1:], col_data_3[1:], col_data_4[1:]


def plot_gap_cdf(version):
    none_gap, stage1_gap, stage2_gap, DCRL_gap = read_excel("../plot/test.xlsx")
    res = stats.relfreq(none_gap, numbins=15)
    x = res.lowerlimit + np.linspace(0, res.binsize * res.frequency.size, res.frequency.size)
    y = np.cumsum(res.frequency)
    plt.plot(x, y, label='DCRL360-none', linewidth=2)

    res = stats.relfreq(stage1_gap, numbins=15)
    x = res.lowerlimit + np.linspace(0, res.binsize * res.frequency.size, res.frequency.size)
    y = np.cumsum(res.frequency)
    plt.plot(x, y, label='DCRL360-1st', linewidth=2)

    res = stats.relfreq(stage2_gap, numbins=15)
    x = res.lowerlimit + np.linspace(0, res.binsize * res.frequency.size, res.frequency.size)
    y = np.cumsum(res.frequency)
    plt.plot(x, y, label='DCRL360-2nd', linewidth=2)

    res = stats.relfreq(DCRL_gap, numbins=15)
    x = res.lowerlimit + np.linspace(0, res.binsize * res.frequency.size, res.frequency.size)
    y = np.cumsum(res.frequency)
    plt.plot(x, y, label='DCRL360', linewidth=2)

    # plt.title("Gap between DCRL360 and baseline")
    plt.legend(fontsize=15)
    plt.xticks(size=12)
    plt.yticks(size=12)

    if version == "ENGLISH":
        plt.xlabel("Gap in QoE", fontsize=15)
    else:
        use_font()
        plt.xlabel("QoE差距", fontsize=15)
    plt.grid(axis="x", linestyle='-.', zorder=0)
    plt.grid(axis="y", linestyle='-.', zorder=0)
    plt.ylabel("CDF", fontsize=15)
    # plt.show()
    if version == "ENGLISH":
        plt.savefig("../plot/figure/Gap-CDF.pdf", bbox_inches="tight")
    else:
        plt.savefig("../plot/figure/差距-CDF.pdf", bbox_inches="tight")


if __name__ == '__main__':
    version = "CHINESE"
    # version = "ENGLISH"
    plot_gap_cdf(version)
