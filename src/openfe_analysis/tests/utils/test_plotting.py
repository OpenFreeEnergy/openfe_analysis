import matplotlib.pyplot as plt
import numpy as np
import pytest

from openfe_analysis.utils.plotting import (
    plot_2D_rmsd,
    plot_ligand_COM_drift,
    plot_ligand_RMSD,
)


@pytest.mark.parametrize("num", [i for i in range(1, 30)])
def test_plot_2D_rmsd(num):
    """
    Smoke test:
      Loop through and test plotting fictitious 2D data
    """
    points = num * (num - 1) // 2
    data = [[0.5 for x in range(points)] for i in range(num)]
    fig = plot_2D_rmsd(data)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


@pytest.mark.parametrize("num_states", [1, 3, 11])
def test_plot_ligand_COM_drift(num_states):
    """
    Smoke test: plot fictitious ligand COM drift for varying numbers of states.
    """
    time = list(np.arange(0, 500, 100, dtype=float))
    data = [np.random.rand(len(time)) for _ in range(num_states)]
    fig = plot_ligand_COM_drift(time, data)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


@pytest.mark.parametrize("num_states", [1, 3, 11])
def test_plot_ligand_RMSD(num_states):
    """
    Smoke test: plot fictitious ligand RMSD for varying numbers of states.
    """
    time = list(np.arange(0, 500, 100, dtype=float))
    data = [np.random.rand(len(time)) for _ in range(num_states)]
    fig = plot_ligand_RMSD(time, data)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
