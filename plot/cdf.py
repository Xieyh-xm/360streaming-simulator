import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats

''' 用于绘制gap的cdf曲线 '''

if __name__ == '__main__':
    rng = np.random.RandomState(seed=111)
    samples = stats.norm.rvs(size=1000, random_state=rng)
    res = stats.relfreq(samples, numbins=100)
    x = res.lowerlimit + np.linspace(0, res.binsize * res.frequency.size, res.frequency.size)
    y = np.cumsum(res.frequency)
    plt.plot(x, y)
    plt.show()
