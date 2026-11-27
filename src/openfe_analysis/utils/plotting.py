# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.artist import Artist
from matplotlib.figure import Figure


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


def plot_boresch_restraint(
    time_ps: npt.ArrayLike,
    restraint_data: Mapping[str, Sequence[npt.ArrayLike]],
    reference_values: Mapping[str, float] | None = None,
    *,
    windows: Sequence[int] | None = None,
    window_labels: Sequence[str] | None = None,
) -> Figure:
    """
    Plot Boresch restraint analysis results for selected lambda windows.

    Parameters
    ----------
    time_ps : ArrayLike
        Time in picoseconds of the analysed frames, shared by all windows.
    restraint_data : Mapping[str, Sequence[ArrayLike]]
        The ``BoreschRestraintAnalysis`` results, keyed by ``bond``,
        ``angle1``, ``angle2``, ``dihedral1``, ``dihedral2`` and
        ``dihedral3``. Each value is a sequence with one time series
        per lambda window.
        Bonds are in Å, angles and dihedrals in radians.
    reference_values : Mapping[str, float] | None
        Equilibrium restraint values, keyed by the same quantities as
        ``restraint_data`` and expressed in the same units.
        Reference values are drawn as dashed horizontal lines.
    windows : Sequence[int] | None
        Lambda window indices to plot (negative indices count from the
        end). Defaults to the first and last window.
    window_labels : Sequence[str] | None
        Legend labels, one per entry of ``windows``.
        Defaults to ``lambda_{index}``, using non-negative indices.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the six restraint time-series plots.

    Raises
    ------
    ValueError
        If ``restraint_data`` contains no windows, ``windows`` is empty
        or contains out-of-range indices, ``window_labels`` does not
        match ``windows`` in length, or ``reference_values`` contains
        unknown keys.
    """
    quantities = {
        "bond": "Bond [Å]",
        "angle1": "Angle 1 [rad]",
        "angle2": "Angle 2 [rad]",
        "dihedral1": "Dihedral 1 [rad]",
        "dihedral2": "Dihedral 2 [rad]",
        "dihedral3": "Dihedral 3 [rad]",
    }

    n_windows = len(restraint_data["bond"])
    if not n_windows:
        raise ValueError("restraint_data contains no windows")

    if windows is None:
        windows = [0, n_windows - 1] if n_windows > 1 else [0]

    windows = list(windows)
    if not windows:
        raise ValueError("At least one window must be selected")

    if any(not -n_windows <= w < n_windows for w in windows):
        raise ValueError(
            f"Window indices must be in [-{n_windows}, {n_windows - 1}], got {windows}"
        )

    windows = [w % n_windows for w in windows]

    if window_labels is None:
        window_labels = [f"lambda_{w}" for w in windows]
    elif len(window_labels) != len(windows):
        raise ValueError("window_labels must match windows in length")

    reference_values = reference_values or {}
    unknown = reference_values.keys() - quantities.keys()
    if unknown:
        raise ValueError(f"Unknown keys in reference_values: {sorted(unknown)}")

    fig, axes = plt.subplots(2, 3, figsize=(8, 4), sharex=True, layout="constrained")

    for ax, (key, y_label) in zip(axes.flat, quantities.items()):
        for window, label in zip(windows, window_labels):
            ax.plot(time_ps, restraint_data[key][window], label=label)

        if key in reference_values:
            ax.axhline(reference_values[key], linestyle="--", color="black", label="Reference")

        ax.set_ylabel(y_label)

    for ax in axes[-1]:
        ax.set_xlabel("Time [ps]")

    legend_entries: dict[str, Artist] = {}
    for ax in axes.flat:
        handles, labels = ax.get_legend_handles_labels()
        legend_entries.update(zip(labels, handles))

    fig.legend(
        handles=legend_entries.values(),
        labels=legend_entries.keys(),
        loc="outside lower center",
        ncols=min(len(legend_entries), 4),
        frameon=False,
    )

    return fig
