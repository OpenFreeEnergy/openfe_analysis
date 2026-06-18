# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import prolif as plf


def plot_2D_rmsd(data: list[np.ndarray] | list[list[float]], vmax: float = 5.0) -> plt.Figure:
    """Plots 2D RMSD for many states

    Parameters
    ----------
    data : list[np.ndarray] | list[list[float]]
      for each state, the 2D RMSD
    vmax : float, optional
      the value to consider "high" in the colourmap to flag bad values,
      defaults to 5.0 (A)

    Returns
    -------
    matplotlib Figure
    """
    twod_rmsd_arrs = []
    for state in data:
        # unpack 2D RMSD data
        # we store N(N-1)//2 values, so find N then make symmetric array
        N = int((1 + np.sqrt(8 * len(state) + 1)) / 2)
        arr = np.zeros((N, N))
        arr[np.triu_indices_from(arr, k=1)] = state
        arr += arr.T

        twod_rmsd_arrs.append(arr)

    nplots = len(data) + 1  # + colorbar

    # plot on 4 x n grid
    nrows = nplots // 4 + (1 if nplots % 4 else 0)

    fig, axes = plt.subplots(nrows, 4)

    for i, (arr, ax) in enumerate(zip(twod_rmsd_arrs, axes.flatten())):
        ax.imshow(arr, vmin=0, vmax=vmax, cmap=plt.get_cmap("cividis"))
        ax.axis("off")  # turn off ticks/labels
        ax.set_title(f"State {i}")

    # turn off unused axes between the last plot and the colorbar
    for i in range(len(twod_rmsd_arrs), len(axes.flatten()) - 1):
        axes.flatten()[i].set_axis_off()

    plt.colorbar(
        axes.flatten()[0].images[0],
        cax=axes.flatten()[-1],
        label=r"RMSD scale ($\AA$)",
        orientation="horizontal",
    )

    fig.suptitle("Protein 2D RMSD")
    fig.tight_layout()

    return fig


def _plot_timeseries(
    time: list[float],
    data: list[np.ndarray],
    ylabel: str,
    title: str,
) -> plt.Figure:
    """
    Plot a per-state timeseries quantity.

    Parameters
    ----------
    time : list[float]
        Time values in picoseconds.
    data : list[np.ndarray]
        Per-state timeseries data.
    ylabel : str
        Label for the y-axis.
    title : str
        Title of the plot.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots()
    for i, s in enumerate(data):
        ax.plot(time, s, label=f"State {i}")
    ax.legend(loc="upper left")
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return fig


def plot_ligand_COM_drift(time: list[float], data: list[np.ndarray]) -> plt.Figure:
    """
    Plot ligand center-of-mass drift over time for each thermodynamic state.

    Parameters
    ----------
    time : list[float]
        Time values in picoseconds.
    data : list[np.ndarray]
        Per-state ligand COM drift.

    Returns
    -------
    matplotlib.figure.Figure
    """
    return _plot_timeseries(
        time,
        data,
        ylabel=r"Distance ($\AA$)",
        title="Ligand COM drift",
    )


def plot_ligand_RMSD(time: list[float], data: list[np.ndarray]) -> plt.Figure:
    """
    Plot ligand RMSD over time for each thermodynamic state.

    Parameters
    ----------
    time : list[float]
        Time values in picoseconds.
    data : list[np.ndarray]
        Per-state ligand RMSD.

    Returns
    -------
    matplotlib.figure.Figure
    """
    return _plot_timeseries(
        time,
        data,
        ylabel=r"RMSD ($\AA$)",
        title="Ligand RMSD",
    )

def plot_prolif_lignetwork(
    analysis,
    ligand_mol=None,
    *,
    frame: Optional[int] = None,
    kind: Literal["aggregate", "frame"] = "frame",
    display_all: bool = False,
    threshold: float = 0.3,
    use_coordinates: bool = True,
    flatten_coordinates: bool = True,
    kekulize: bool = False,
    molsize: int = 35,
    rotation: float = 0,
    carbon: float = 0.16,
    width: str = "100%",
    height: str = "500px",
    fontsize: int = 20,
    show_interaction_data: bool = False,
):
    """
    2D ProLIF ligand-network visualization.
    """
    if not analysis.ifp:
        raise RuntimeError(
            "No ProLIF fingerprint data found. Run `analysis.run(...)` first."
        )

    available_frames = list(analysis.fp.ifp.keys())

    if frame is None:
        frame = available_frames[0]

    if kind == "frame" and frame not in analysis.fp.ifp:
        preview = available_frames[:10]
        suffix = " ..." if len(available_frames) > 10 else ""
        raise ValueError(
            f"frame={frame} not present in fingerprint results. "
            f"Available frames: {preview}{suffix}"
        )

    if frame is not None:
        analysis.universe.trajectory[frame]

    if ligand_mol is None:
        ligand_mol = plf.Molecule.from_mda(
            analysis.ligand_ag,
            inferrer=None,
            implicit_hydrogens=False,
            use_segid=analysis.fp.use_segid,
        )

    return analysis.fp.plot_lignetwork(
        ligand_mol,
        kind=kind,
        frame=frame,
        display_all=display_all,
        threshold=threshold,
        use_coordinates=use_coordinates,
        flatten_coordinates=flatten_coordinates,
        kekulize=kekulize,
        molsize=molsize,
        rotation=rotation,
        carbon=carbon,
        width=width,
        height=height,
        fontsize=fontsize,
        show_interaction_data=show_interaction_data,
    )

def plot_prolif_3d(
    analysis,
    ligand_mol=None,
    protein_mol=None,
    water_mol=None,
    *,
    frame: int = 0,
    size: tuple[int, int] = (650, 600),
    display_all: bool = False,
    only_interacting: bool = True,
    remove_hydrogens: bool | Literal["ligand", "protein", "water"] = True,
):
    """
    3D ProLIF interaction visualization using py3Dmol.
    """
    if not analysis.ifp:
        raise RuntimeError(
            "No ProLIF fingerprint data found. Run `analysis.run(...)` first."
        )

    if frame not in analysis.fp.ifp:
        raise ValueError(f"frame={frame} not present in fingerprint results.")

    analysis.universe.trajectory[frame]

    if ligand_mol is None:
        ligand_mol = plf.Molecule.from_mda(
            analysis.ligand_ag,
            inferrer=None,
            implicit_hydrogens=False,
            use_segid=analysis.fp.use_segid,
        )

    if protein_mol is None:
        protein_mol = plf.Molecule.from_mda(
            analysis.protein_ag,
            implicit_hydrogens=False,
            use_segid=analysis.fp.use_segid,
        )

    if water_mol is None and analysis.water_ag.n_atoms:
        water_mol = plf.Molecule.from_mda(
            analysis.water_ag,
            implicit_hydrogens=False,
            use_segid=analysis.fp.use_segid,
        )

    return analysis.fp.plot_3d(
        ligand_mol,
        protein_mol,
        water_mol=water_mol,
        frame=frame,
        size=size,
        display_all=display_all,
        only_interacting=only_interacting,
        remove_hydrogens=remove_hydrogens,
    )
