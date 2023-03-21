import numpy as np
import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5, 6]
qoe = [372.856, 384.540, 391.699, 397.073, 397.646, 397.787]
wastage = [0.491, 0.489, 0.487, 0.488, 0.488, 0.487]

# 双坐标轴
fig = plt.figure()
ax1 = fig.add_subplot(111)
ln1 = ax1.plot(x, qoe, color="#930D1E", linewidth=4, marker='o', label='QoE', zorder=100)
ax1.set_ylabel(r"QoE", fontsize=13)
ax1.set_xlabel("Target threshold $b_{th}$", fontsize=13)
ax1.legend(loc=0, fontsize=13)
ax1.grid(axis="y", linestyle='-.', zorder=0)

ax2 = ax1.twinx()
ln2 = ax2.plot(x, wastage, color="#FF7F60", linewidth=4, marker='o', label='Wastage raio', zorder=100)
ax2.set_ylim(0.45, 0.53)
ax2.set_ylabel(r"Wastage raio", fontsize=13)

lns = ln1 + ln2
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, bbox_to_anchor=(0.5, 1.), loc=8, ncol=10, fontsize=13)

plt.tight_layout()
plt.savefig("./figure/param_study.pdf", dpi=1000, bbox_inches="tight")
plt.show()
