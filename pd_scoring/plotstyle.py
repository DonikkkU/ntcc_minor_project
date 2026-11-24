import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#ffffff"
GOOD, WARN, SERIOUS, CRITICAL = "#0ca30c", "#fab219", "#ec835a", "#d03b3b"


def apply():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlesize": 10,
        "axes.titlecolor": INK, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "lines.linewidth": 2, "legend.frameon": False, "legend.fontsize": 8,
        "axes.prop_cycle": matplotlib.cycler(color=SERIES),
    })


def save(fig, path):
    fig.savefig(path)
    plt.close(fig)
