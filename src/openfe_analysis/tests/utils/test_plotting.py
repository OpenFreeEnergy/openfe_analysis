import matplotlib.pyplot as plt
import numpy as np
import pytest

from openfe_analysis.utils.plotting import (
    plot_2D_rmsd,
    plot_boresch_restraint,
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


KEYS = ["bond", "angle1", "angle2", "dihedral1", "dihedral2", "dihedral3"]


@pytest.fixture
def time_ps():
    return np.arange(50) * 0.5


@pytest.fixture
def restraint_data(time_ps):
    return {key: [time_ps + 10 * i + j for j in range(4)] for i, key in enumerate(KEYS)}


def legend_labels(fig):
    return [text.get_text() for text in fig.legends[0].get_texts()]


def test_default_windows(time_ps, restraint_data):
    fig = plot_boresch_restraint(time_ps, restraint_data)

    assert len(fig.axes) == 6
    assert legend_labels(fig) == ["lambda_0", "lambda_3"]

    for ax, key in zip(fig.axes, KEYS):
        assert len(ax.lines) == 2
        np.testing.assert_array_equal(ax.lines[0].get_ydata(), restraint_data[key][0])
        np.testing.assert_array_equal(ax.lines[1].get_ydata(), restraint_data[key][-1])
    plt.close(fig)


def test_selected_windows_and_labels(time_ps, restraint_data):
    fig = plot_boresch_restraint(
        time_ps,
        restraint_data,
        windows=[1, -1],
        window_labels=["mid", "end"],
    )

    assert legend_labels(fig) == ["mid", "end"]

    for ax, key in zip(fig.axes, KEYS):
        first, last = ax.lines

        np.testing.assert_array_equal(first.get_xdata(), time_ps)
        np.testing.assert_array_equal(last.get_xdata(), time_ps)
        np.testing.assert_array_equal(first.get_ydata(), restraint_data[key][1])
        np.testing.assert_array_equal(last.get_ydata(), restraint_data[key][-1])
    plt.close(fig)


@pytest.mark.parametrize(
    "reference_values",
    [
        pytest.param(
            {key: float(i) for i, key in enumerate(KEYS)},
            id="full",
        ),
        pytest.param({"dihedral3": 1.0}, id="partial"),
        pytest.param({}, id="empty"),
    ],
)
def test_reference_values(time_ps, restraint_data, reference_values):
    fig = plot_boresch_restraint(time_ps, restraint_data, reference_values)

    for ax, key in zip(fig.axes, KEYS):
        if key in reference_values:
            assert len(ax.lines) == 3
            reference_line = ax.lines[-1]
            np.testing.assert_array_equal(reference_line.get_ydata(), reference_values[key])
            assert reference_line.get_linestyle() == "--"
        else:
            assert len(ax.lines) == 2

    expected_labels = ["lambda_0", "lambda_3"]
    if reference_values:
        expected_labels.append("Reference")

    assert legend_labels(fig) == expected_labels
    plt.close(fig)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"windows": [4]}, "Window indices must be in"),
        ({"windows": [-5]}, "Window indices must be in"),
        ({"windows": []}, "At least one window"),
        ({"window_labels": ["only one"]}, "window_labels must match"),
        (
            {"reference_values": {"dihedal1": 0.0}},
            "Unknown keys.*dihedal1",
        ),
    ],
)
def test_invalid_input_raises(time_ps, restraint_data, kwargs, match):
    with pytest.raises(ValueError, match=match):
        plot_boresch_restraint(time_ps, restraint_data, **kwargs)
