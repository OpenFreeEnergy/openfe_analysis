import matplotlib.pyplot as plt
import numpy as np
import pytest

from openfe_analysis.utils.plotting import (
    plot_2D_rmsd,
    plot_ligand_COM_drift,
    plot_ligand_RMSD,
    plot_prolif_3d,
    plot_prolif_barcode,
    plot_prolif_lignetwork,
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


class _Trajectory(dict):
    """Frame dict that remembers the last frame indexed."""

    last_frame = None

    def __getitem__(self, frame):
        self.last_frame = frame
        return dict.get(self, frame)


@pytest.fixture
def make_ag():
    """Factory for a dummy AtomGroup with a recording .universe.trajectory."""

    def _make(*, n_atoms=None, frames=(0,)):
        traj = _Trajectory({f: None for f in frames})
        attrs = {"universe": type("U", (), {"trajectory": traj})()}
        if n_atoms is not None:
            attrs["n_atoms"] = n_atoms
        return type("AG", (), attrs)()

    return _make


@pytest.fixture
def patch_from_mda(monkeypatch):
    """Patch plf.Molecule.from_mda; return the list of recorded calls."""
    calls = []

    def fake(atomgroup, **kwargs):
        mol = object()
        calls.append({"ag": atomgroup, "kwargs": kwargs, "mol": mol})
        return mol

    monkeypatch.setattr("openfe_analysis.utils.plotting.plf.Molecule.from_mda", fake)
    return calls


@pytest.fixture
def make_fp():
    """Factory for a dummy fingerprint that records its plot_* calls (fp.calls)."""

    def _make(ifp=None, use_segid=False):
        calls = {}

        class FP:
            def __init__(self):
                self.ifp = {0: {"x": []}} if ifp is None else ifp
                self.use_segid = use_segid
                self.calls = calls

            def plot_lignetwork(self, ligand_mol, **kw):
                calls["lignetwork"] = {"mol": ligand_mol, "kw": kw}
                return "fake-view"

            def plot_3d(self, lig, prot, **kw):
                calls["plot_3d"] = {"lig": lig, "prot": prot, "kw": kw}
                return "fake-3d"

            def plot_barcode(self, **kw):
                calls["barcode"] = {"kw": kw}
                return "fake-barcode"

        return FP()

    return _make


def test_plot_prolif_lignetwork_builds_ligand_mol_and_delegates(make_fp, make_ag, patch_from_mda):
    """plot_prolif_lignetwork builds a ligand mol and delegates to fp.plot_lignetwork."""
    fp = make_fp(ifp={0: {"dummy": []}})
    ligand_ag = make_ag()

    view = plot_prolif_lignetwork(fp, ligand_ag, frame=0, kind="frame")

    assert view == "fake-view"
    assert patch_from_mda[0]["ag"] is ligand_ag
    assert patch_from_mda[0]["kwargs"]["inferrer"] is None
    assert patch_from_mda[0]["kwargs"]["implicit_hydrogens"] is False
    assert patch_from_mda[0]["kwargs"]["use_segid"] == fp.use_segid
    assert fp.calls["lignetwork"]["mol"] is patch_from_mda[0]["mol"]
    assert fp.calls["lignetwork"]["kw"]["frame"] == 0
    assert fp.calls["lignetwork"]["kw"]["kind"] == "frame"
    assert ligand_ag.universe.trajectory.last_frame == 0


def test_plot_prolif_3d_builds_mols_and_delegates(make_fp, make_ag, patch_from_mda):
    """plot_prolif_3d builds ligand/protein/water mols and delegates to fp.plot_3d."""
    fp = make_fp()
    ligand_ag = make_ag(n_atoms=10)
    protein_ag = make_ag(n_atoms=100)
    water_ag = make_ag(n_atoms=3)

    assert plot_prolif_3d(fp, ligand_ag, protein_ag, water_ag, frame=0) == "fake-3d"

    built = [c["ag"] for c in patch_from_mda]
    assert built == [ligand_ag, protein_ag, water_ag]
    assert fp.calls["plot_3d"]["kw"]["frame"] == 0


def test_plot_prolif_functions_raise_without_ifp(make_fp, make_ag):
    """Plotting before the fingerprint is run raises a clear RuntimeError."""
    fp = make_fp(ifp={})
    ag = make_ag()
    with pytest.raises(RuntimeError, match="No ProLIF fingerprint data"):
        plot_prolif_lignetwork(fp, ag)
    with pytest.raises(RuntimeError, match="No ProLIF fingerprint data"):
        plot_prolif_3d(fp, ag, ag)
    with pytest.raises(RuntimeError, match="No ProLIF fingerprint data"):
        plot_prolif_barcode(fp)


def test_plot_prolif_lignetwork_invalid_frame_raises(make_fp, make_ag):
    """kind='frame' with a frame not in the results raises ValueError."""
    fp = make_fp(ifp={0: {"x": []}})
    ag = make_ag()
    with pytest.raises(ValueError, match="not present"):
        plot_prolif_lignetwork(fp, ag, frame=99, kind="frame")


def test_plot_prolif_lignetwork_auto_picks_first_frame(make_fp, make_ag, patch_from_mda):
    """With no frame given, the first available frame is used."""
    fp = make_fp(ifp={5: {"x": []}})
    ag = make_ag(frames=(5,))

    assert plot_prolif_lignetwork(fp, ag) == "fake-view"
    assert fp.calls["lignetwork"]["kw"]["frame"] == 5


def test_plot_prolif_barcode_delegates(make_fp):
    """plot_prolif_barcode delegates to the fingerprint's plot_barcode."""
    fp = make_fp()
    assert plot_prolif_barcode(fp) == "fake-barcode"
    assert fp.calls["barcode"]["kw"]["xlabel"] == "Frame"


def test_plot_prolif_3d_invalid_frame_raises(make_fp, make_ag):
    """plot_prolif_3d with a frame not in the results raises ValueError."""
    fp = make_fp(ifp={0: {"x": []}})
    ag = make_ag()
    with pytest.raises(ValueError, match="not present"):
        plot_prolif_3d(fp, ag, ag, frame=99)


def test_plot_prolif_3d_skips_water_when_absent(make_fp, make_ag, patch_from_mda):
    """With water_ag=None, no water molecule is built and water_mol stays None."""
    fp = make_fp()
    assert plot_prolif_3d(fp, make_ag(), make_ag(), water_ag=None, frame=0) == "fake-3d"
    assert len(patch_from_mda) == 2  # ligand + protein only
    assert fp.calls["plot_3d"]["kw"]["water_mol"] is None
