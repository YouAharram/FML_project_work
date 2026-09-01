"""Figura: come una molecola diventa l'input della GIN.

Prende una molecola del dataset (benzammide, indice 86 di ``ogbg-moltox21``) e mostra i
tre passaggi della rappresentazione: il disegno chimico, il grafo con i nodi numerati e i
tensori ``x`` / ``edge_index`` che la rete riceve davvero.

La figura e' disegnata alla larghezza con cui compare nella relazione (circa 6,2 pollici),
cosi' che i corpi del testo restino leggibili senza riscalature.

    python scripts/fig_rappresentazione.py
"""
import io
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle
from rdkit import Chem, RDLogger
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data import load_tox21  # noqa: E402
from plotstyle import SERIES, SURFACE, TEXT_PRIMARY, TEXT_SECONDARY, use_project_style  # noqa: E402

RDLogger.DisableLog("rdApp.*")

IDX = 86            # NC(=O)c1ccccc1, benzammide: 9 atomi, 9 legami
FIGDIR = ROOT / "results" / "figures"
COLORE_ATOMO = {"C": SERIES[0], "N": SERIES[2], "O": SERIES[1]}
COLONNE_X = ["numero atomico", "chiralità", "grado", "carica formale", "numero di H",
             "elettroni spaiati", "ibridazione", "aromatico", "in anello"]

TITOLO = dict(fontsize=8.5, color=TEXT_PRIMARY, fontweight="bold")
COL_X = [0.115 + k * 0.094 for k in range(9)]
MONO = dict(family="monospace", fontsize=7.0, color=TEXT_PRIMARY)
NOTA = dict(fontsize=6.3, color=TEXT_SECONDARY)


def disegno_molecola(smiles, larghezza=720, altezza=430):
    """Rende la molecola come immagine RGB, con gli indici degli atomi."""
    mol = Chem.MolFromSmiles(smiles)
    rdDepictor.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DCairo(larghezza, altezza)
    opts = drawer.drawOptions()
    opts.addAtomIndices = True
    opts.setBackgroundColour((0.988, 0.988, 0.984))
    opts.bondLineWidth = 2
    opts.baseFontSize = 0.7
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    return plt.imread(io.BytesIO(drawer.GetDrawingText())), mol


def coordinate(mol):
    """Coordinate 2D degli atomi, normalizzate, cosi' il grafo ricalca il disegno."""
    conf = mol.GetConformer()
    p = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y]
                  for i in range(mol.GetNumAtoms())])
    p -= p.mean(0)
    return p / np.abs(p).max()


def pannello_grafo(ax, mol, pos):
    """Il grafo: un nodo per atomo, un arco per legame."""
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        ax.plot(*zip(pos[i], pos[j]), color=TEXT_SECONDARY, lw=1.1, zorder=1)
    for i, atom in enumerate(mol.GetAtoms()):
        c = COLORE_ATOMO.get(atom.GetSymbol(), SERIES[0])
        ax.add_patch(Circle(pos[i], 0.155, facecolor=c, edgecolor=SURFACE, lw=1.2, zorder=2))
        ax.text(*pos[i], str(i), ha="center", va="center", color="white",
                fontsize=6, fontweight="bold", zorder=3)
    for simbolo, colore in COLORE_ATOMO.items():
        ax.scatter([], [], s=22, color=colore, label=simbolo)
    ax.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.01),
              handletextpad=0.1, columnspacing=0.9, fontsize=6.5)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.30, 1.30)
    ax.set_aspect("equal")


def tabella_x(ax, x):
    """La matrice degli attributi dei nodi, con i nomi delle colonne."""
    ax.text(0.0, 0.97, "x", **{**TITOLO, "family": "monospace"})
    ax.text(0.024, 0.97, "  —  9 attributi categorici per atomo   [9 x 9]", **TITOLO)

    for k, nome in enumerate(COLONNE_X):
        ax.text(COL_X[k], 0.785, nome, fontsize=5.6, color=TEXT_SECONDARY,
                ha="left", va="bottom", rotation=32)
    for r, (i, simbolo) in enumerate([(0, "N"), (1, "C"), (2, "O"), (3, "c")]):
        y = 0.725 - r * 0.066
        ax.text(0.0, y, f"{simbolo} {i}", **{**MONO, "color": TEXT_SECONDARY})
        for k in range(9):
            ax.text(COL_X[k], y, f"{int(x[i, k])}", ha="center", **MONO)
    ax.text(0.0, 0.461, "  ...", **{**MONO, "color": TEXT_SECONDARY})
    ax.text(0.36, 0.461, "(altre 5 righe: gli atomi 4-8 dell'anello)", **NOTA)


def tabella_edges(ax, edge_index):
    """La lista degli archi, con un arco per verso di percorrenza."""
    ax.text(0.0, 0.295, "edge_index", **{**TITOLO, "family": "monospace"})
    ax.text(0.148, 0.295, "  —  ogni legame in entrambi i versi   [2 x 18]", **TITOLO)
    for r, etichetta in enumerate(("da", "a")):
        y = 0.205 - r * 0.066
        ax.text(0.0, y, etichetta, **{**MONO, "color": TEXT_SECONDARY})
        for k in range(edge_index.shape[1]):
            ax.text(0.062 + k * 0.0505, y, f"{int(edge_index[r, k])}", ha="center", **MONO)


def pannello_tensori(ax, x, edge_index):
    """I due tensori che descrivono la molecola, piu' il passaggio agli embedding."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    tabella_x(ax, x)
    ax.text(0.0, 0.390, "ogni valore è un indice, non una quantità: nella prima colonna "
                        "6 vuol dire azoto e 5 carbonio, non sei e cinque", **NOTA)
    tabella_edges(ax, edge_index)
    ax.annotate("", xy=(0.5, 0.045), xytext=(0.5, 0.098),
                arrowprops=dict(arrowstyle="-|>", color=TEXT_SECONDARY, lw=1))
    ax.text(0.5, 0.030, "AtomEncoder: 9 tabelle di embedding, sommate      "
                        r"$h^{(0)}$: 9 $\times$ 300",
            fontsize=7.0, color=TEXT_PRIMARY, ha="center", va="top")


def main():
    use_project_style()
    dataset, _ = load_tox21()
    smiles = pd.read_csv(ROOT / "data/ogbg_moltox21/mapping/mol.csv.gz")["smiles"][IDX]
    d = dataset[IDX]
    img, mol = disegno_molecola(smiles)

    fig = plt.figure(figsize=(6.3, 5.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.80, 1.32], hspace=0.16, wspace=0.02,
                          left=0.01, right=0.99, top=0.94, bottom=0.01)
    ax_mol, ax_graph = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax_tens = fig.add_subplot(gs[1, :])
    for ax in (ax_mol, ax_graph, ax_tens):
        ax.set_axis_off()
        ax.grid(False)

    ax_mol.imshow(img)
    ax_mol.set_title(f"1. la molecola:  {smiles}", loc="center", **TITOLO)
    pannello_grafo(ax_graph, mol, coordinate(mol))
    ax_graph.set_title("2. il grafo: 9 nodi, 9 archi", loc="center", **TITOLO)
    pannello_tensori(ax_tens, d.x.numpy(), d.edge_index.numpy())
    ax_tens.set_title("3. i tensori che riceve la rete", loc="center", **TITOLO)

    FIGDIR.mkdir(parents=True, exist_ok=True)
    out = FIGDIR / "04_rappresentazione_grafo.png"
    fig.savefig(out, dpi=300, bbox_inches=None)
    print(f"figura salvata in {out}")


if __name__ == "__main__":
    main()
