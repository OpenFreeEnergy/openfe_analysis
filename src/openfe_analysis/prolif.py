from __future__ import annotations

import warnings
from typing import Any, Dict, Literal, Optional, Sequence, Tuple

import MDAnalysis as mda
import numpy as np
import prolif as plf

from .utils.universe_utils import guess_ligand_bonds


class ProLIFAnalysis:
    """
    ProLIF interaction fingerprint analysis for an OpenFEReader Universe.

    Parameters
    ----------
    universe
        MDAnalysis Universe containing topology and trajectory.
    ligand_ag
        mda.AtomGroup representing the ligand.
    water_order
        Maximum WaterBridge interaction order (water-water interaction).
        Only used if "WaterBridge" is tracked.
    protein_cutoff
        Distance cutoff in angstrom used to define the protein pocket
        around the ligand.
    water_cutoff
        Distance cutoff in angstrom used to define waters considered
        around the ligand/protein pocket.
    interactions
        Which interactions to track:
          - None: ProLIF defaults (Hydrophobic, HBDonor, HBAcceptor,
            PiStacking, Anionic, Cationic, CationPi, PiCation, VdWContact);
            see ProLIF's ``DEFAULT_INTERACTIONS``:
            https://github.com/chemosim-lab/ProLIF/blob/6d993eb1b54cd20cc160461dba1ee5e775cb4037/prolif/fingerprint.py#L97
          - "all": every available interaction, including bridged ones
            (e.g. WaterBridge)
          - Sequence[str]: explicit list like ["VdWContact", "HBDonor"]
    guess_bonds
        If True, guess bonds for (protein, ligand, water) so ProLIF can
        recognize donors/acceptors and bonded hydrogens.
    """

    def __init__(
        self,
        universe: mda.Universe,
        ligand_ag: mda.AtomGroup,
        water_order: int = 3,
        protein_cutoff: float = 12.0,
        water_cutoff: float = 8.0,
        interactions: Optional[Sequence[str] | str] = None,
        guess_bonds: bool = True,
    ) -> None:
        self.universe = universe
        self.ligand_ag = ligand_ag
        self.water_order = water_order

        self.frames: Optional[np.ndarray] = None
        self.times: Optional[np.ndarray] = None
        self.n_frames: Optional[int] = None

        if guess_bonds:
            self._guess_bonds()

        self._setup_selections(protein_cutoff, water_cutoff)

        self.fp = self._build_fingerprint(interactions)

    def _guess_bonds(self) -> None:
        """
        Guess bonds for the protein, ligand and water selections in-place so
        RDKit/ProLIF can detect donors/acceptors and bonded hydrogens.
        """
        # Protein: guess on the full protein so any pocket residue later has bonds
        guess_ligand_bonds(self.universe.select_atoms("protein"))

        # Ligand: stable group
        guess_ligand_bonds(self.ligand_ag)

        # Water: only if water-mediated interactions are of interest
        water = self.universe.select_atoms("water")
        if water.n_atoms:
            guess_ligand_bonds(water)

    def _setup_selections(self, protein_cutoff: float, water_cutoff: float) -> None:
        """
        Build the updating pocket and water selections around the ligand.
        """
        self.protein_ag = self.universe.select_atoms(
            f"protein and byres around {protein_cutoff} group ligand",
            ligand=self.ligand_ag,
            updating=True,
        )
        self.water_ag = self.universe.select_atoms(
            f"water and byres around {water_cutoff} (group ligand or group pocket)",
            ligand=self.ligand_ag,
            pocket=self.protein_ag,
            updating=True,
        )

    def _build_fingerprint(self, interactions: Optional[Sequence[str] | str]) -> plf.Fingerprint:
        """
        Resolve the requested interactions and construct the ProLIF Fingerprint.

        Configures WaterBridge parameters when it is requested and waters are
        present, and drops WaterBridge (with a warning) when they are not.
        """
        available = plf.Fingerprint.list_available(show_bridged=True)

        fp_interactions: Optional[list[str]]
        if interactions is None:
            fp_interactions = None

        elif interactions == "all":
            # ProLIF's "all" excludes bridged interactions (e.g. WaterBridge)
            fp_interactions = list(available)

        else:
            fp_interactions = list(interactions)

        self._parameters: Optional[dict] = None
        if fp_interactions is not None and "WaterBridge" in fp_interactions:
            if self.water_ag.n_atoms == 0:
                warnings.warn(
                    "WaterBridge selected but water selection is empty at the initial "
                    "frame; removing WaterBridge from the requested interactions.",
                    UserWarning,
                    stacklevel=3,
                )
                fp_interactions = [
                    interaction for interaction in fp_interactions if interaction != "WaterBridge"
                ]
            else:
                self._parameters = {
                    "WaterBridge": {"water": self.water_ag, "order": self.water_order}
                }

        if not fp_interactions:
            return plf.Fingerprint(parameters=self._parameters)
        return plf.Fingerprint(
            interactions=fp_interactions,
            parameters=self._parameters,
        )

    def run(
        self,
        *,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        step: Optional[int] = None,
        residues: Optional[Literal["all"] | Sequence[str | int]] = None,
        progress: bool = True,
        n_jobs: Optional[int] = None,
        parallel_strategy: Optional[Literal["chunk", "queue"]] = None,
        converter_kwargs: Optional[Tuple[Dict[str, Any], Dict[str, Any]]] = None,
    ) -> "ProLIFAnalysis":
        """
        Run the fingerprint calculation over a slice of the trajectory.

        Parameters
        ----------
        start, stop, step
            Trajectory slicing parameters.
        residues
            Passed to ProLIF: ``"all"`` to track every residue, or an explicit
            sequence of residue identifiers. If None, ProLIF's default is used
            and interactions with atoms are identified.
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

        return self

    @property
    def ifp(self):
        """
        Convenience accessor for underlying ProLIF fingerprint results.
        """
        return getattr(self.fp, "ifp", None)

    def to_dataframe(self, **kwargs):
        """
        Transform fingerprint results to pd.DataFrame.
        """
        return self.fp.to_dataframe(**kwargs)
