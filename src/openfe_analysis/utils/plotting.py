# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import matplotlib.pyplot as plt
import numpy as np


def plot_2D_rmsd(data: list[np.ndarray], vmax: float = 5.0) -> plt.Figure:
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
