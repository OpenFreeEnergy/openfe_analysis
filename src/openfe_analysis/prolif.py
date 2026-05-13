from __future__ import annotations

import numpy as np
from typing import Any, Dict, Optional, Sequence, Tuple, Literal
import warnings

import MDAnalysis as mda
from MDAnalysis.guesser.tables import vdwradii as MDA_VDWRADII

import prolif as plf


class ProLIFAnalysis:
    """
    ProLIF interaction fingerprint analysis for an OpenFEReader Universe.
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
        vdwradii: Optional[Dict[str, float]] = None,
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
        protein_cutoff
            Distance cutoff in angstrom used to define the protein pocket
            around the ligand.
        water_cutoff
            Distance cutoff in angstrom used to define waters considered
            around the ligand/protein pocket.
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

        self.frames = None
        self.times = None
        self.n_frames = None
        self.ifp_df = None

        # --- Guess bonds once on stable selections so RDKit/ProLIF can detect HBonds ---
        if guess_bonds:
            if vdwradii is None:
                vdwradii = dict(MDA_VDWRADII)
                vdwradii.update(
                    {
                        "Cl": vdwradii["CL"],
                        "Br": vdwradii["BR"],
                        "Na": vdwradii["NA"],
                    }
                )

            # Protein: guess on the full protein so any pocket residue later has bonds
            universe.select_atoms("protein").guess_bonds(vdwradii=vdwradii)

            # Ligand: stable group
            self.ligand_ag.guess_bonds(vdwradii=vdwradii)

            # Water: only if you care about water-mediated interactions
            wat_all = universe.select_atoms("water")
            if wat_all.n_atoms:
                wat_all.guess_bonds(vdwradii=vdwradii)

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

        available = plf.Fingerprint.list_available(show_bridged=True)

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
                warnings.warn(
                    "WaterBridge selected but water selection is empty at the initial "
                    "frame; removing WaterBridge from the requested interactions.",
                    UserWarning,
                    stacklevel=2,
                )
                fp_interactions = [
                    interaction
                    for interaction in fp_interactions
                    if interaction != "WaterBridge"
                ]
            else:
                self._parameters = {
                    "WaterBridge": {"water": self.water_ag, "order": self.water_order}
                }

        if not fp_interactions:
            self.fp = plf.Fingerprint(parameters=self._parameters)
        else:
            self.fp = plf.Fingerprint(
                interactions=fp_interactions,
                parameters=self._parameters,
            )

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

        return self

    @property
    def ifp(self):
        """
        Convenience accessor for underlying ProLIF fingerprint results.
        """
        return getattr(self.fp, "ifp", None)

    # For now, depending on what we do withe the data
    def to_dataframe(self, **kwargs):
        """
        Transform fingerprint results to pd.DataFrame.
        """
        df = self.fp.to_dataframe(**kwargs)
        self.ifp_df = df
        return df

    def plot_lignetwork(
        self,
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
        if not self.ifp:
            raise RuntimeError(
                "No ProLIF fingerprint data found. Run `analysis.run(...)` first."
            )

        available_frames = list(self.fp.ifp.keys())

        if frame is None:
            frame = available_frames[0]

        if kind == "frame" and frame not in self.fp.ifp:
            preview = available_frames[:10]
            suffix = " ..." if len(available_frames) > 10 else ""
            raise ValueError(
                f"frame={frame} not present in fingerprint results. "
                f"Available frames: {preview}{suffix}"
            )

        if frame is not None:
            self.universe.trajectory[frame]

        if ligand_mol is None:
            ligand_mol = plf.Molecule.from_mda(
                self.ligand_ag,
                inferrer=None,
                implicit_hydrogens=False,
                use_segid=self.fp.use_segid,
            )

        return self.fp.plot_lignetwork(
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

    plot_2d = plot_lignetwork

    def plot_barcode(
        self,
        *,
        figsize: tuple[int, int] = (8, 10),
        dpi: int = 100,
        interactive: bool = True,
        n_frame_ticks: int = 10,
        residues_tick_location: Literal["top", "bottom"] = "top",
        xlabel: str = "Frame",
        subplots_kwargs: Optional[dict] = None,
        tight_layout_kwargs: Optional[dict] = None,
    ):
        """
        Barcode plot of interactions across frames.
        """
        if not self.ifp:
            raise RuntimeError(
                "No ProLIF fingerprint data found. Run `analysis.run(...)` first."
            )

        return self.fp.plot_barcode(
            figsize=figsize,
            dpi=dpi,
            interactive=interactive,
            n_frame_ticks=n_frame_ticks,
            residues_tick_location=residues_tick_location,
            xlabel=xlabel,
            subplots_kwargs=subplots_kwargs,
            tight_layout_kwargs=tight_layout_kwargs,
        )

    def plot_3d(
        self,
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
        if not self.ifp:
            raise RuntimeError(
                "No ProLIF fingerprint data found. Run `analysis.run(...)` first."
            )

        if frame not in self.fp.ifp:
            raise ValueError(f"frame={frame} not present in fingerprint results.")

        self.universe.trajectory[frame]

        if ligand_mol is None:
            ligand_mol = plf.Molecule.from_mda(
                self.ligand_ag,
                inferrer=None,
                implicit_hydrogens=False,
                use_segid=self.fp.use_segid,
            )

        if protein_mol is None:
            protein_mol = plf.Molecule.from_mda(
                self.protein_ag,
                implicit_hydrogens=False,
                use_segid=self.fp.use_segid,
            )

        if water_mol is None and self.water_ag.n_atoms:
            water_mol = plf.Molecule.from_mda(
                self.water_ag,
                implicit_hydrogens=False,
                use_segid=self.fp.use_segid,
            )

        return self.fp.plot_3d(
            ligand_mol,
            protein_mol,
            water_mol=water_mol,
            frame=frame,
            size=size,
            display_all=display_all,
            only_interacting=only_interacting,
            remove_hydrogens=remove_hydrogens,
        )
