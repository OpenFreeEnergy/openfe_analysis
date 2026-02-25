from __future__ import annotations

import numpy as np
from typing import Any, Dict, Optional, Sequence, Tuple

import MDAnalysis as mda
from MDAnalysis.analysis.base import AnalysisBase

import prolif as plf


class ProLIFAnalysis(AnalysisBase):
    """
    ProLIF interaction fingerprint analysis for an OpenFEReader Universe.
    """

    def __init__(
        self,
        universe: mda.Universe,
        ligand_ag: mda.AtomGroup,
        water_order: int = 3,
        interactions: Optional[Sequence[str] | str] = None,
        guess_bonds: bool = True,
        vdwradii: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the ProLIF analysis.

        Parameters
        ----------
        universe
            MDAnalysis Universe containing topology and trajectory.
        ligand_ag
            mda.AtomGroup representing the ligand.
        water_order
            Maximum WaterBridge interaction order (water-water interaction).
            Only used if "WaterBridge" is tracked.
        interactions
            Which interactions to track:
              - None: ProLIF defaults
              - "all": all registered (non-bridged; depends on ProLIF version)
              - Sequence[str]: explicit list like ["VdWContact", "HBDonor"]
        guess_bonds
            If True, guess bonds for (protein, ligand, water) so ProLIF can
            recognize donors/acceptors and bonded hydrogens.
        vdwradii
            Optional dict of van der Waals radii used by MDAnalysis bond guesser.
            Useful when your topology contains types the guesser doesn't know
            (e.g. "Cl", "Na"). If None, uses coded defaults.
        """
        self.universe = universe
        self.ligand_ag = ligand_ag
        self.water_order = water_order

        super().__init__(universe.trajectory, **kwargs)

        # --- Guess bonds once on stable selections so RDKit/ProLIF can detect HBonds ---
        if guess_bonds:
            if vdwradii is None:
                # minimal overrides needed for your system (atom types include Cl/Na)
                vdwradii = {
                    "Cl": 1.75,
                    "CL": 1.75,
                    "Br": 1.85,
                    "BR": 1.85,
                    "Na": 2.27,
                    "NA": 2.27,
                }

            # Protein: guess on the full protein so any pocket residue later has bonds
            universe.select_atoms("protein").guess_bonds(vdwradii=vdwradii)

            # Ligand: stable group
            self.ligand_ag.guess_bonds(vdwradii=vdwradii)

            # Water: only if you care about water-mediated interactions
            if guess_bonds:
                wat_all = universe.select_atoms("water")
                if wat_all.n_atoms:
                    wat_all.guess_bonds(vdwradii=vdwradii)

        # Currently adding here but maybe as args?
        self.protein_ag = self.universe.select_atoms(
            "protein and byres around 12 group ligand",
            ligand=self.ligand_ag,
            updating=True,
        )
        self.water_ag = self.universe.select_atoms(
            "water and byres around 8 (group ligand or group pocket)",
            ligand=self.ligand_ag,
            pocket=self.protein_ag,
            updating=True,
        )

        available = plf.Fingerprint.list_available()

        if interactions is None:
            fp_interactions = None

        elif interactions == "all":
            fp_interactions = "all"

        else:
            # Cover case of false interaction
            missing = [i for i in interactions if i not in available]
            if missing:
                raise ValueError(
                    f"Unknown interaction(s): {missing}. " f"Available: {available}"
                )
            fp_interactions = list(interactions)

        self._parameters = None
        if (
            fp_interactions is not None
            and fp_interactions != "all"
            and "WaterBridge" in fp_interactions
        ):
            if self.water_ag.n_atoms == 0:
                raise ValueError("WaterBridge selected but water selection is empty.")
            self._parameters = {
                "WaterBridge": {"water": self.water_ag, "order": self.water_order}
            }

        if fp_interactions is None:
            self.fp = plf.Fingerprint()
        else:
            self.fp = plf.Fingerprint(interactions=fp_interactions)

    def _prepare(self):
        self.results.ifp = None
        self.results.ifp_df = None

    def _conclude(self):
        self.results.ifp = getattr(self.fp, "ifp", None)

    def run(
        self,
        *,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        step: Optional[int] = None,
        residues: Optional[bool] = None,
        progress: bool = True,
        n_jobs: Optional[int] = None,
        parallel_strategy: Optional[str] = None,
        converter_kwargs: Optional[Tuple[Dict[str, Any], Dict[str, Any]]] = None,
    ) -> "ProLIFAnalysis":
        """
        Run the fingerprint calculation over a slice of the trajectory.

        Parameters
        ----------
        start, stop, step
            Trajectory slicing parameters.
        residues
            Passed to ProLIF: whether to aggregate interactions with residues.
            If None, ProLIF's default is used and interactions with atoms are identified.
        progress
            Show progress bar.
        n_jobs
            Number of workers for parallel execution.).
        parallel_strategy
            ProLIF parallel strategy. If None, this wrapper sets:
              - "chunk" for n_jobs None/1
              - "queue" for n_jobs > 1
        converter_kwargs
            Two dicts: (ligand_kwargs, protein_kwargs) forwarded to the MDAnalysis→RDKit
            converter. If None, we default to:
              - ligand: {"inferrer": None, "implicit_hydrogens": False}  (avoid valence issues)
              - protein: {"implicit_hydrogens": False}                  (use topology bonds)

        Returns
        -------
        self
            Returned for fluent chaining.
        """
        # Due to FEReader trajectory only certain strategies work with the format
        if parallel_strategy is None:
            # avoid ProLIF trying to pickle FEReader/netCDF trajectory to auto-pick strategy
            parallel_strategy = "chunk" if (n_jobs is None or n_jobs == 1) else "queue"

        _slice = slice(start, stop, step)
        traj = self.universe.trajectory[_slice]

        try:
            n_total = len(self.universe.trajectory)
            s0, s1, s2 = _slice.indices(n_total)
            self.frames = np.arange(s0, s1, s2, dtype=int)
            self.n_frames = len(traj)

            if (
                hasattr(self.universe.trajectory, "times")
                and self.universe.trajectory.times is not None
            ):
                self.times = np.asarray(self.universe.trajectory.times)[self.frames]
            elif getattr(self.universe.trajectory, "dt", None) is not None:
                self.times = self.frames * self.universe.trajectory.dt
            else:
                self.times = None
        except Exception:
            self.frames = None
            self.times = None
            self.n_frames = None

        if converter_kwargs is None:
            # Avoid Valence errors
            converter_kwargs = (
                {"inferrer": None, "implicit_hydrogens": False},  # ligand
                {"implicit_hydrogens": False},  # protein
            )

        self.fp.run(
            traj,
            self.ligand_ag,
            self.protein_ag,
            residues=residues,
            converter_kwargs=converter_kwargs,
            progress=progress,
            n_jobs=n_jobs,
            parallel_strategy=parallel_strategy,
        )

        self._conclude()
        return self

    # For now, depending on what we do withe the data
    def to_dataframe(self, **kwargs):
        """
        Transform fingerprint results to pd.DataFrame.
        """
        df = self.fp.to_dataframe(**kwargs)
        self.results.ifp_df = df
        return df
