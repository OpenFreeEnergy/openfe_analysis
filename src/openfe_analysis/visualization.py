from .utils import draw_multi_matches

ofe_colors =  [(49 / 256, 57 / 256, 77 / 256, 1),  # Badass Blue
                  (184 / 256, 87 / 256, 65 / 256, 1),  # Feeling spicy
                  (0, 147 / 256, 132 / 256, 1),  # Feeling sick
                  (217 / 256, 196 / 256, 177 / 256, 1),  # Beastlygrey
                  (102 / 256, 102 / 256, 102 / 256, 1),  # Sandy Sergio
                  (0 / 256, 47 / 256, 74 / 256, 1), ]  # otherBlue]


def plot_torsion_profiles(
        angles,
        torsions,
        rdmol,
        nbins=36,
        colors=None):

    if (colors is None):
        colors = ofe_colors

    fig, axes = plt.subplots(ncols=angles.shape[1] + 1, figsize=[angles.shape[1] * 9, 9])
    axes = np.array(axes, ndmin=1).flat

    # Draw Structure
    draw_mol = Chem.Mol(rdmol)
    # draw_mol = Chem.RemoveAllHs(draw_mol)
    for atom in draw_mol.GetAtoms():
        atom.SetProp("atomNote", str(atom.GetIdx()))

    Chem.rdDepictor.Compute2DCoords(draw_mol)

    highlightAtomLists = [[int(a) for a in t] for t in torsions]

    d = rdMolDraw2D.MolDraw2DCairo(400, 400)
    d.DrawMolecule(draw_mol)

    draw_multi_matches(d, draw_mol, highlightAtomLists, r_min=0.3, r_dist=0.13,
                       relative_bond_width=0.5, color_list=colors,
                       line_width=4)

    d.FinishDrawing()
    g = Image.open(io.BytesIO(d.GetDrawingText()))
    g = np.asarray(g)

    axes[0].imshow(g)
    axes[0].axis("off")

    # Plot
    for i, ax in enumerate(axes[1:]):
        ax.hist(angles[:, i - 1], bins=nbins, range=(-180, 180), density=True, color=colors[i % len(colors)]);

        ax.set_xlim([-190, 190])
        x_angles = np.linspace(-180, 180, 5)
        ax.set_xticks(x_angles)
        ax.set_xticklabels(x_angles, fontsize=20, rotation=90)

        ax.set_ylim(0, 1 / nbins)

        if (i == 0):
            ylims = np.round(np.linspace(0, 1 / nbins, 5), 2)
            ax.set_yticks(ylims)
            ylims = np.round(np.linspace(0, 1 / nbins, 5), 2)
            ax.set_yticklabels(ylims, fontsize=20, )
        else:
            ax.set_yticks([])

        ax.set_title("Tors " + str(torsions[i][1]) + " - " + str(torsions[i][2]), fontsize=26)

    axes[1].set_ylabel("occurance ratio", fontsize=24, )
    axes[1].set_xlabel("angles", fontsize=24, )
    axes[2].set_xlabel("angles", fontsize=24, )
    axes[3].set_xlabel("angles", fontsize=24, )

    # fig.subplots_adjust(wspace=0)
    fig.tight_layout()
    axes[0].margins(tight=True)

    return fig
