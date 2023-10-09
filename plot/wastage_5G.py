import matplotlib.pyplot as plt
from mplfonts.bin.cli import init
from mplfonts import use_font

init()

version = "CHINESE"
# version = "ENGLISH"

PROPOSED = [546.41, 0.47]
RAM360 = [447.40, 0.69]
TTS = [518.70, 0.46]
STS = [490.80, 0.43]
BS = [48.31, 0.72]

# [FCC , NORWAY]
plt.scatter(PROPOSED[0], PROPOSED[1], color='#FF6721', s=200, label="DCRL360", zorder=100)
plt.annotate("DCRL360", xy=(PROPOSED[0], PROPOSED[1]), xytext=(PROPOSED[0] - 80, PROPOSED[1] + 0.020), color='#FF6721',
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
plt.annotate("TTS", xy=(TTS[0], TTS[1]), xytext=(TTS[0] - 75, TTS[1] + 0.002), color='#3FC9FC',
             fontsize=15, zorder=100)
# plt.errorbar(TTS[0], TTS[1], xerr=TTS_std[0], yerr=TTS_std[1], capsize=10, capthick=3, elinewidth=3, color='#3FC9FC',
#              label="TTS360")

plt.scatter(STS[0], STS[1], color='#8DE529', marker='D', s=200, label="STS", zorder=100)
plt.annotate("STS", xy=(STS[0], STS[1]), xytext=(STS[0] - 75, STS[1] + 0.003), color='#8DE529',
             fontsize=15, zorder=100)
# plt.errorbar(STS[0], STS[1], xerr=STS_std[0], yerr=STS_std[1], capsize=10, capthick=3, elinewidth=3, color='#8DE529',
#              label="STS")

plt.scatter(BS[0], BS[1], color='#3366FF', marker=',', s=200, label="BS", zorder=100)
plt.annotate("BS", xy=(BS[0], BS[1]), xytext=(BS[0] + 10, BS[1] - 0.035), color='#3366FF',
             fontsize=15, zorder=100)
# plt.errorbar(BS[0], BS[1], xerr=BS_std[0], yerr=BS_std[1], capsize=10, capthick=3, elinewidth=3, color='#3366FF',
#              label="BS")

plt.xlim(20, 600)
plt.ylim(0.4, 0.75)
plt.xticks(size=12)
plt.yticks(size=12)
plt.grid(linestyle='-.', zorder=0)


if version == "ENGLISH":
    plt.xlabel("Average QoE", fontsize=13)
    plt.ylabel("Wastage ratio", fontsize=13)
    plt.savefig("./figure/QoE-Wastage-5G.pdf", dpi=1000, bbox_inches="tight")
else:
    use_font()
    plt.xlabel("平均QoE", fontsize=13)
    plt.ylabel("带宽浪费率", fontsize=13)
    plt.savefig("./figure/5G网络下QoE和带宽浪费率的比较.pdf", dpi=1000, bbox_inches="tight")


plt.show()
