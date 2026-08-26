"""Stile condiviso dalle figure del progetto, cosi' che notebook e script coincidano."""
import matplotlib as mpl
import matplotlib.pyplot as plt

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#dcdcd8"


def use_project_style():
    """Applica la palette e i parametri tipografici del progetto a matplotlib."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_SECONDARY,
        "axes.titlecolor": TEXT_PRIMARY,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 12,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "font.size": 10,
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })


def bar_labels(ax, bars, fmt="{:.1f}%", pad=0.6):
    """Scrive il valore in fondo a ogni barra di un grafico a barre orizzontali."""
    for b in bars:
        w = b.get_width()
        ax.text(w + pad, b.get_y() + b.get_height() / 2, fmt.format(w),
                va="center", ha="left", fontsize=8, color=TEXT_SECONDARY)
