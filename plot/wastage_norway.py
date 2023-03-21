import matplotlib.pyplot as plt
import numpy as np

# PROPOSED = [373.1215288, 0.493688015]
# RAM360 = [342.5192654, 0.554569719]
# TTS = [277.3774076, 0.537294011]
# STS = [305.1805402, 0.47238032]
# BS = [46.87390636, 0.718173906]
#
# PROPOSED_std = [120.8479998, 0.105777632]
# RAM360_std = [117.5529533, 0.100880892]
# TTS_std = [84.43535559, 0.091249858]
# STS_std = [114.4885003, 0.109134169]
# BS_std = [3.269415944, 0.015234673]

PROPOSED = [349.170, 0.499]
RAM360 = [316.848, 0.562]
TTS = [260.676, 0.545]
STS = [274.388, 0.481]
BS = [46.543, 0.718]

# [FCC , NORWAY]
plt.scatter(PROPOSED[0], PROPOSED[1], color='#FF6721', s=200, label="DCRL360", zorder=100)
plt.annotate("DCRL360", xy=(PROPOSED[0], PROPOSED[1]), xytext=(PROPOSED[0] - 10, PROPOSED[1] - 0.035), color='#FF6721',
             fontsize=15, zorder=100)
# plt.errorbar(PROPOSED[0], PROPOSED[1], xerr=PROPOSED_std[0], yerr=PROPOSED_std[1], capsize=10, capthick=3, elinewidth=3,
#              color='#FF6721',
#              label="DCRL360")

plt.scatter(RAM360[0], RAM360[1], color='#FFD311', marker='v', s=200, label="RAM360", zorder=0)
plt.annotate("RAM360", xy=(RAM360[0], RAM360[1]), xytext=(RAM360[0], RAM360[1] + 0.015), color='#FFD311',
             fontsize=15, zorder=100)
# plt.errorbar(RAM360[0], RAM360[1], xerr=RAM360_std[0], yerr=RAM360_std[1], capsize=10, capthick=3, elinewidth=3,
#              color='#FFD311', label="RAM360")

plt.scatter(TTS[0], TTS[1], color='#3FC9FC', marker='X', s=200, label="TTS", zorder=100)
plt.annotate("TTS", xy=(TTS[0], TTS[1]), xytext=(TTS[0] - 45, TTS[1] + 0.005), color='#3FC9FC',
             fontsize=15, zorder=100)
# plt.errorbar(TTS[0], TTS[1], xerr=TTS_std[0], yerr=TTS_std[1], capsize=10, capthick=3, elinewidth=3, color='#3FC9FC',
#              label="TTS360")

plt.scatter(STS[0], STS[1], color='#8DE529', marker='D', s=200, label="STS", zorder=100)
plt.annotate("STS", xy=(STS[0], STS[1]), xytext=(STS[0] - 45, STS[1] + 0.005), color='#8DE529',
             fontsize=15, zorder=100)
# plt.errorbar(STS[0], STS[1], xerr=STS_std[0], yerr=STS_std[1], capsize=10, capthick=3, elinewidth=3, color='#8DE529',
#              label="STS")

plt.scatter(BS[0], BS[1], color='#3366FF', marker=',', s=200, label="BS", zorder=100)
plt.annotate("BS", xy=(BS[0], BS[1]), xytext=(BS[0] + 10, BS[1] - 0.035), color='#3366FF',
             fontsize=15, zorder=100)
# plt.errorbar(BS[0], BS[1], xerr=BS_std[0], yerr=BS_std[1], capsize=10, capthick=3, elinewidth=3, color='#3366FF',
#              label="BS")


plt.xlabel("Average QoE", fontsize=13)
plt.ylabel("Wastage ratio", fontsize=13)
# plt.title("QoE vs. Wastage ratio")
# plt.legend()
plt.xlim(20, 420)
plt.ylim(0.4, 0.75)
plt.xticks(size=12)
plt.yticks(size=12)
plt.grid(linestyle='-.', zorder=0)
# plt.legend(bbox_to_anchor=(0.5, 1.02), loc=8, ncol=10, fontsize=13)
plt.savefig("./figure/QoE-Wastage-HSDPA.pdf", dpi=1000, bbox_inches="tight")
plt.show()
