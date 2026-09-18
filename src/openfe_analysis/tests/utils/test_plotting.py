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


def test_plot_prolif_lignetwork_builds_ligand_mol_and_delegates(monkeypatch):
    """
    plot_prolif_lignetwork builds a ligand molecule when one is not provided
    and delegates to the fingerprint's plot_lignetwork.
    """
    calls = {}

    class DummyTrajectory:
        def __init__(self):
            self.last_frame = None

        def __getitem__(self, frame):
            self.last_frame = frame
            return None

    traj = DummyTrajectory()
    ligand_ag = type("AG", (), {"universe": type("U", (), {"trajectory": traj})()})()

    class DummyFP:
        ifp = {0: {"dummy": []}}
        use_segid = False

        def plot_lignetwork(self, ligand_mol, **kwargs):
            calls["plot_lignetwork"] = (ligand_mol, kwargs)
            return "fake-view"

    fp = DummyFP()
    fake_ligand_mol = object()

    def fake_from_mda(atomgroup, **kwargs):
        calls["from_mda"] = (atomgroup, kwargs)
        return fake_ligand_mol

    monkeypatch.setattr(
        "openfe_analysis.utils.plotting.plf.Molecule.from_mda",
        fake_from_mda,
    )

    view = plot_prolif_lignetwork(fp, ligand_ag, frame=0, kind="frame")

    assert view == "fake-view"
    assert calls["from_mda"][0] is ligand_ag
    assert calls["from_mda"][1]["inferrer"] is None
    assert calls["from_mda"][1]["implicit_hydrogens"] is False
    assert calls["from_mda"][1]["use_segid"] == fp.use_segid
    assert calls["plot_lignetwork"][0] is fake_ligand_mol
    assert calls["plot_lignetwork"][1]["frame"] == 0
    assert calls["plot_lignetwork"][1]["kind"] == "frame"
    assert traj.last_frame == 0


def test_plot_prolif_3d_builds_mols_and_delegates(monkeypatch):
    """plot_prolif_3d builds ligand/protein/water mols and delegates to fp.plot_3d."""
    calls = {}
    traj = {0: None}

    def ag(n):
        return type(
            "AG",
            (),
            {"n_atoms": n, "universe": type("U", (), {"trajectory": traj})()},
        )()

    ligand_ag, protein_ag, water_ag = ag(10), ag(100), ag(3)

    class DummyFP:
        ifp = {0: {"x": []}}
        use_segid = False

        def plot_3d(self, lig, prot, **kw):
            calls.update(args=(lig, prot), kw=kw)
            return "fake-3d"

    fp = DummyFP()

    made = []
    monkeypatch.setattr(
        "openfe_analysis.utils.plotting.plf.Molecule.from_mda",
        lambda ag, **kw: made.append(ag) or object(),
    )

    assert plot_prolif_3d(fp, ligand_ag, protein_ag, water_ag, frame=0) == "fake-3d"
    assert made == [ligand_ag, protein_ag, water_ag]
    assert calls["kw"]["frame"] == 0


def test_plot_prolif_functions_raise_without_ifp():
    """Plotting before the fingerprint is run raises a clear RuntimeError."""
    fp = type("FP", (), {"ifp": {}})()
    ag = object()
    with pytest.raises(RuntimeError, match="No ProLIF fingerprint data"):
        plot_prolif_lignetwork(fp, ag)
    with pytest.raises(RuntimeError, match="No ProLIF fingerprint data"):
        plot_prolif_3d(fp, ag, ag)
    with pytest.raises(RuntimeError, match="No ProLIF fingerprint data"):
        plot_prolif_barcode(fp)


def test_plot_prolif_lignetwork_invalid_frame_raises():
    """kind='frame' with a frame not in the results raises ValueError."""
    fp = type("FP", (), {"ifp": {0: {"x": []}}, "use_segid": False})()
    ag = type("AG", (), {"universe": type("U", (), {"trajectory": {0: None}})()})()
    with pytest.raises(ValueError, match="not present"):
        plot_prolif_lignetwork(fp, ag, frame=99, kind="frame")


def test_plot_prolif_lignetwork_auto_picks_first_frame(monkeypatch):
    """With no frame given, the first available frame is used."""
    calls = {}
    ag = type("AG", (), {"universe": type("U", (), {"trajectory": {5: None}})()})()

    class DummyFP:
        ifp = {5: {"x": []}}
        use_segid = False

        def plot_lignetwork(self, ligand_mol, **kwargs):
            calls["kw"] = kwargs
            return "fake-view"

    monkeypatch.setattr(
        "openfe_analysis.utils.plotting.plf.Molecule.from_mda",
        lambda ag, **kw: object(),
    )

    assert plot_prolif_lignetwork(DummyFP(), ag) == "fake-view"
    assert calls["kw"]["frame"] == 5


def test_plot_prolif_barcode_delegates():
    """plot_prolif_barcode delegates to the fingerprint's plot_barcode."""
    calls = {}

    class DummyFP:
        ifp = {0: {"x": []}}

        def plot_barcode(self, **kwargs):
            calls["kw"] = kwargs
            return "fake-barcode"

    assert plot_prolif_barcode(DummyFP()) == "fake-barcode"
    assert calls["kw"]["xlabel"] == "Frame"


def test_plot_prolif_3d_invalid_frame_raises():
    """plot_prolif_3d with a frame not in the results raises ValueError."""
    fp = type("FP", (), {"ifp": {0: {"x": []}}, "use_segid": False})()
    ag = type("AG", (), {"universe": type("U", (), {"trajectory": {0: None}})()})()
    with pytest.raises(ValueError, match="not present"):
        plot_prolif_3d(fp, ag, ag, frame=99)


def test_plot_prolif_3d_skips_water_when_absent(monkeypatch):
    """With water_ag=None, no water molecule is built and water_mol stays None."""
    calls = {}

    def ag():
        return type("AG", (), {"universe": type("U", (), {"trajectory": {0: None}})()})()

    class DummyFP:
        ifp = {0: {"x": []}}
        use_segid = False

        def plot_3d(self, lig, prot, **kw):
            calls["kw"] = kw
            return "fake-3d"

    made = []
    monkeypatch.setattr(
        "openfe_analysis.utils.plotting.plf.Molecule.from_mda",
        lambda a, **kw: made.append(a) or object(),
    )

    assert plot_prolif_3d(DummyFP(), ag(), ag(), water_ag=None, frame=0) == "fake-3d"
    assert len(made) == 2  # ligand + protein only
    assert calls["kw"]["water_mol"] is None
