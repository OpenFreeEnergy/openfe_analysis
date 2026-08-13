import MDAnalysis as mda
import numpy as np
import pytest
from rdkit.Chem import Lipinski

from openfe_analysis.prolif import ProLIFAnalysis
from openfe_analysis.reader import FEReader


def test_prolifanalysis_runs_vdwcontact(simulation_skipped_nc, hybrid_system_skipped_pdb):
    """
    Test for identification of interactions
    """
    u = mda.Universe(hybrid_system_skipped_pdb, simulation_skipped_nc, format=FEReader, index=0)
    ligand_ag = u.select_atoms("resname UNK")

    analysis = ProLIFAnalysis(u, ligand_ag, interactions=["VdWContact"], guess_bonds=True)
    analysis.run(stop=5, step=1, n_jobs=1, progress=False)

    df = analysis.to_dataframe(dtype=np.uint8)
    assert df.shape[0] == 5
    assert hasattr(analysis.fp, "ifp")
    assert len(analysis.fp.ifp) == 5

    assert analysis.ifp is analysis.fp.ifp
    assert len(analysis.ifp) == 5

    # Ensure there is at least one detected interaction across all processed frames
    assert sum(len(v) for v in analysis.fp.ifp.values()) > 0


def test_guess_bonds_enables_protein_chemistry(simulation_skipped_nc, hybrid_system_skipped_pdb):
    """
    Test for protein connectivity
    """
    u = mda.Universe(hybrid_system_skipped_pdb, simulation_skipped_nc, format=FEReader, index=0)
    ligand_ag = u.select_atoms("resname UNK")

    analysis = ProLIFAnalysis(u, ligand_ag, interactions=["VdWContact"], guess_bonds=True)

    # pick a residue from the pocket and check it has connectivity in RDKit
    u.trajectory[0]
    res_atoms = analysis.protein_ag.residues[0].atoms
    res_mol = res_atoms.convert_to("RDKIT", implicit_hydrogens=False)
    assert res_mol.GetNumBonds() > 0

    # ensure the protein donors/acceptors exist
    prot_mol = analysis.protein_ag.convert_to("RDKIT", implicit_hydrogens=False)
    assert Lipinski.NumHDonors(prot_mol) + Lipinski.NumHAcceptors(prot_mol) > 0


def test_prolifanalysis_accepts_all_keyword(simulation_skipped_nc, hybrid_system_skipped_pdb):
    """
    The string "all" should be accepted as the special keyword for
    all available ProLIF interactions.
    """
    u = mda.Universe(hybrid_system_skipped_pdb, simulation_skipped_nc, format=FEReader, index=0)
    ligand_ag = u.select_atoms("resname UNK")

    analysis = ProLIFAnalysis(u, ligand_ag, interactions="all", guess_bonds=True)

    assert analysis.fp is not None


def test_default_interactions_are_prolif_defaults(simulation_skipped_nc, hybrid_system_skipped_pdb):
    """interactions=None should track ProLIF's DEFAULT_INTERACTIONS."""
    u = mda.Universe(hybrid_system_skipped_pdb, simulation_skipped_nc, format=FEReader, index=0)
    ligand_ag = u.select_atoms("resname UNK")

    analysis = ProLIFAnalysis(u, ligand_ag, interactions=None)

    expected = {
        "Hydrophobic",
        "HBDonor",
        "HBAcceptor",
        "PiStacking",
        "Anionic",
        "Cationic",
        "CationPi",
        "PiCation",
        "VdWContact",
    }
    assert set(analysis.fp.interactions) == expected


def test_waterbridge_empty_selection_warns_and_skips_parameters(
    simulation_skipped_nc, hybrid_system_skipped_pdb, monkeypatch
):
    """
    Requesting WaterBridge with an empty water selection should warn
    instead of raising, and should not configure WaterBridge parameters.
    """
    u = mda.Universe(hybrid_system_skipped_pdb, simulation_skipped_nc, format=FEReader, index=0)
    ligand_ag = u.select_atoms("resname UNK")

    original_select_atoms = u.select_atoms

    def patched_select_atoms(selection, *args, **kwargs):
        if selection == "water and byres around 8 (group ligand or group pocket)":
            return u.atoms[[]]
        return original_select_atoms(selection, *args, **kwargs)

    monkeypatch.setattr(u, "select_atoms", patched_select_atoms)

    with pytest.warns(UserWarning, match="WaterBridge selected"):
        analysis = ProLIFAnalysis(
            u,
            ligand_ag,
            interactions=["WaterBridge"],
            guess_bonds=True,
        )

    assert analysis._parameters is None


def test_waterbridge_with_water_sets_parameters(simulation_skipped_nc, hybrid_system_skipped_pdb):
    """
    Requesting WaterBridge with waters present should configure the
    WaterBridge parameters.
    """
    u = mda.Universe(hybrid_system_skipped_pdb, simulation_skipped_nc, format=FEReader, index=0)
    ligand_ag = u.select_atoms("resname UNK")

    analysis = ProLIFAnalysis(u, ligand_ag, interactions=["WaterBridge"])

    assert analysis._parameters is not None
    assert "WaterBridge" in analysis._parameters


def test_guess_bonds_false_skips_guessing(simulation_skipped_nc, hybrid_system_skipped_pdb):
    """guess_bonds=False should skip bond guessing and still build a fingerprint."""
    u = mda.Universe(hybrid_system_skipped_pdb, simulation_skipped_nc, format=FEReader, index=0)
    ligand_ag = u.select_atoms("resname UNK")

    analysis = ProLIFAnalysis(u, ligand_ag, interactions=["VdWContact"], guess_bonds=False)

    assert analysis.fp is not None


def test_plot_prolif_lignetwork_builds_ligand_mol_and_delegates(monkeypatch):
    """
    plot_prolif_lignetwork builds a ligand molecule when one is not provided
    and delegates to the fingerprint's plot_lignetwork.
    """
    from openfe_analysis.utils.plotting import plot_prolif_lignetwork

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
    from openfe_analysis.utils.plotting import plot_prolif_3d

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
    from openfe_analysis.utils.plotting import (
        plot_prolif_3d,
        plot_prolif_barcode,
        plot_prolif_lignetwork,
    )

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
    from openfe_analysis.utils.plotting import plot_prolif_lignetwork

    fp = type("FP", (), {"ifp": {0: {"x": []}}, "use_segid": False})()
    ag = type("AG", (), {"universe": type("U", (), {"trajectory": {0: None}})()})()
    with pytest.raises(ValueError, match="not present"):
        plot_prolif_lignetwork(fp, ag, frame=99, kind="frame")


def test_plot_prolif_lignetwork_auto_picks_first_frame(monkeypatch):
    """With no frame given, the first available frame is used."""
    from openfe_analysis.utils.plotting import plot_prolif_lignetwork

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
    from openfe_analysis.utils.plotting import plot_prolif_barcode

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
    from openfe_analysis.utils.plotting import plot_prolif_3d

    fp = type("FP", (), {"ifp": {0: {"x": []}}, "use_segid": False})()
    ag = type("AG", (), {"universe": type("U", (), {"trajectory": {0: None}})()})()
    with pytest.raises(ValueError, match="not present"):
        plot_prolif_3d(fp, ag, ag, frame=99)


def test_plot_prolif_3d_skips_water_when_absent(monkeypatch):
    """With water_ag=None, no water molecule is built and water_mol stays None."""
    from openfe_analysis.utils.plotting import plot_prolif_3d

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
