import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
import xlrd

''' 用于绘制gap的cdf曲线 '''


# stage1 stage2

def read_excel(file_path):
    workbook = xlrd.open_workbook(file_path)
    worksheet = workbook.sheet_by_index(0)
    col_data_1 = worksheet.col_values(0)  # stage 1
    col_data_2 = worksheet.col_values(1)  # stage 2
    return col_data_1[1:], col_data_2[1:]


def plot_gap_cdf():
    stage_1_gap, stage_2_gap = read_excel("../plot/test.xlsx")
    res = stats.relfreq(stage_1_gap, numbins=100)
    x = res.lowerlimit + np.linspace(0, res.binsize * res.frequency.size, res.frequency.size)
    y = np.cumsum(res.frequency)
    plt.plot(x, y, label='stage 1', linewidth=2)

    res = stats.relfreq(stage_2_gap, numbins=100)
    x = res.lowerlimit + np.linspace(0, res.binsize * res.frequency.size, res.frequency.size)
    y = np.cumsum(res.frequency)
    plt.plot(x, y, label='stage 2', linewidth=2)

    plt.title("Gap between DCRL360 and baseline")
    plt.legend(fontsize=13)
    # plt.show()
    plt.savefig("../plot/figure/Gap-CDF.pdf")


if __name__ == '__main__':
    plot_gap_cdf()
