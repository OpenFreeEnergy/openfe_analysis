import netCDF4 as nc
import numpy as np
from numpy.typing import NDArray
from pathlib import Path
import warnings
from openff.units import unit
from typing import Optional, Tuple

from openfe_analysis import __version__


def _determine_position_indices(dataset: nc.Dataset) -> NDArray:
    """
    Determine which iteration indices hold positions.

    Parameters
    ----------
    dataset : nc.Dataset
      Dataset holding the MultiStateReporter generated NetCDF file.

    Returns
    -------
    indices : NDArray[int]
      An ordered array of iteration indices which hold positions.

    Note
    ----
    This assumes that the indices are equally spaced by a given
    value.
    """
    if hasattr(dataset, 'PositionInterval'):
        indices = [
            i for i in
            range(0, dataset.dimensions['iteration'].size, dataset.PositionInterval)
        ]
    else:
        wmsg = ('This is an older NetCDF file that does not yet contain '
                   'information about the write frequency of positions and '
                   'velocities. We will assume that positions and velocities '
                   'were written out at every iteration. ')
        warnings.warn(wmsg)
        indices = [i for i in range(0, dataset.dimensions['iteration'].size)]

    indices = np.array(indices)

    if not all(np.diff(indices) == np.diff(indices)[0]):
        errmsg = (
            "Positions are not written at a consistent frame rate, "
            "this is not currently supported"
        )
        raise ValueError(errmsg)

    return indices


def _state_to_replica(dataset: nc.Dataset, state_num: int,
                      frame_num: int) -> int:
    """Convert a state index to replica index at a given Dataset frame

    Parameters
    ----------
    dataset : netCDF4.Dataset
        Dataset containing the MultiState reporter generated NetCDF file
        with information about all the frames and replica in the system.
    state_num : int
        Index of the state to get the effective replica for.
    frame_num : int
        Index of the frame to get the effective replica for.

    Returns
    -------
    int
        Index of the replica which represents that thermodynamic state
        for that frame.
    """
    state_distribution = dataset.variables['states'][frame_num].data
    return np.where(state_distribution == state_num)[0][0]


def _replica_positions_at_frame(
    dataset: nc.Dataset,
    replica_index: int,
    frame_num: int
) -> Optional[unit.Quantity]:
    """
    Helper method to extract atom positions of a state at a given
    Dataset frame.

    Parameters
    ----------
    dataset : netCDF4.Dataset
        Dataset containing the MultiState information.
    replica_index : int
        Replica index to extract positions for.
    frame_num : int
        Frame number to extract positions for.

    Returns
    -------
    Optional[unit.Quantity]
        A n_atoms * 3 position Quantity array. Returns ``None``
        if all the values are masked (i.e. no positions were stored
        for that frame).
    """
    # If all the positions are masked (i.e. not present)
    if dataset.variables['positions'][frame_num][replica_index].mask.all():
        return None

    pos = dataset.variables['positions'][frame_num][replica_index].data
    pos_units = dataset.variables['positions'].units
    return pos * unit(pos_units)


def _create_new_dataset(
    filename: Path, n_atoms: int,
    title: str
) -> nc.Dataset:
    """
    Helper method to create a new NetCDF dataset which follows the
    AMBER convention (see: https://ambermd.org/netcdf/nctraj.xhtml)

    Parameters
    ----------
    filename : path.Pathlib
        Name of the new netcdf trajectory to write.
    n_atoms : int
        Number of atoms to store in trajectory.
    title : str
        Title of trajectory.

    Returns
    -------
    netCDF4.Dataset
        AMBER Conventions compliant NetCDF dataset to store information
        contained in MultiState reporter generated NetCDF file.
    """
    ncfile = nc.Dataset(filename, 'w', format='NETCDF3_64BIT_OFFSET')
    ncfile.Conventions = 'AMBER'
    ncfile.ConventionVersion = "1.0"
    ncfile.application = "openfe_analysis"
    ncfile.program = f"openfe_analysis {__version__}"
    ncfile.programVersion = f"{__version__}"
    ncfile.title = title

    # Set the dimensions
    ncfile.createDimension('frame', None)
    ncfile.createDimension('spatial', 3)
    ncfile.createDimension('atom', n_atoms)
    ncfile.createDimension('cell_spatial', 3)
    ncfile.createDimension('cell_angular', 3)
    ncfile.createDimension('label', 5)

    # Set the variables
    # positions
    pos = ncfile.createVariable('coordinates', 'f4', ('frame', 'atom', 'spatial'))
    pos.units = 'angstrom'
    # we could also set this to 0.1 and do no nm to angstrom scaling on write
    pos.scale_factor = 1.0 

    # Note: OpenMMTools NetCDF files store velocities
    # but honestly it's rather useless, so we don't populate them
    # Note 2: NetCDF file doesn't contain any time information... 
    # so we can't populate that either, this might trip up some readers..
    # Note 3: We'll need to convert box vectors (in nm) to
    # unitcell (in angstrom & degrees)
    cell_lengths = ncfile.createVariable(
        'cell_lengths', 'f8', ('frame', 'cell_spatial')
    )
    cell_lengths.units = 'angstrom'
    cell_angles = ncfile.createVariable(
        'cell_angles', 'f8', ('frame', 'cell_angular')
    )
    cell_angles.units = 'degree'

    return ncfile


def _get_unitcell(
    dataset: nc.Dataset, replica_index: int, frame_num: int
) -> Optional[Tuple[unit.Quantity]]:
    """
    Helper method to extract a unit cell from the stored
    box vectors in a MultiState reporter generated NetCDF file
    at a given state and Dataset frame.

    Parameters
    ----------
    dataset : netCDF4.Dataset
        Dataset of MultiState reporter generated NetCDF file.
    replica_index : int
        Replica for which to get the unit cell for.
    frame_num : int
        Frame for which to get the unit cell for.

    Returns
    -------
    Optional[Tuple[lx, ly, lz, alpha, beta, gamma]]
        Unit cell lengths and angles in angstroms and degrees.
        If box_vectors are masked (i.e. they were not stored at this frame),
        will return ``None``.
    """
    # Case: no box_vectors were stored at this frame
    if dataset.variables['box_vectors'][frame_num][replica_index].mask.all():
        return None

    vecs = dataset.variables['box_vectors'][frame_num][replica_index].data
    vecs_units = dataset.variables['box_vectors'].units
    x, y, z = (vecs * unit(vecs_units)).to('angstrom').m
    lx = np.linalg.norm(x)
    ly = np.linalg.norm(y)
    lz = np.linalg.norm(z)
    # angle between y and z
    alpha = np.arccos(np.dot(y, z) / (ly * lz))
    # angle between x and z
    beta = np.arccos(np.dot(x, z) / (lx * lz))
    # angle between x and y
    gamma = np.arccos(np.dot(x, y) / (lx * ly))

    return lx, ly, lz, np.rad2deg(alpha), np.rad2deg(beta), np.rad2deg(gamma)


def trajectory_from_multistate(input_file: Path, output_file: Path,
                               state_number: Optional[int] = None,
                               replica_number: Optional[int] = None) -> None:
    """
    Extract a state's trajectory (in an AMBER compliant format)
    from a MultiState sampler generated NetCDF file.

    Either a state or replica index must be supplied, but not both!

    Parameters
    ----------
    input_file : path.Pathlib
        Path to the input MultiState sampler generated NetCDF file.
    output_file : path.Pathlib
        Path to the AMBER-style NetCDF trajectory to be written.
    state_number : int, optional
        Index of the state to write out to the trajectory.
    replica_number : int, optional
        Index of the replica to write out
    """
    if not ((state_number is None) ^ (replica_number is None)):
        raise ValueError("Supply either state or replica number, "
                         f"got state_number={state_number} "
                         f"and replica_number={replica_number}")

    # Open MultiState NC file and get number of atoms and frames
    multistate = nc.Dataset(input_file, 'r')
    n_atoms = len(multistate.variables['positions'][0][0])
    n_replicas = len(multistate.variables['positions'][0])
    frame_list = _determine_position_indices(multistate)
    n_frames = len(frame_list)

    # Sanity check
    if state_number is not None and (state_number + 1 > n_replicas):
        # Note this works for now, but when we have more states
        # than replicas (e.g. SAMS) this won't really work
        errmsg = "State does not exist"
        raise ValueError(errmsg)

    # Create output AMBER NetCDF convention file
    traj = _create_new_dataset(
        output_file, n_atoms,
        title=f"state {state_number} trajectory from {input_file}"
    )

    replica_id: int = -1
    if replica_number is not None:
        replica_id = replica_number

    # Loopy de loop over n_frames so that the new Dataset
    # is just 0 -> n_frames
    for frame in range(n_frames):
        if state_number is not None:
            replica_id = _state_to_replica(
                multistate, state_number, frame_list[frame]
            )

        traj.variables['coordinates'][frame] = _replica_positions_at_frame(
            multistate, replica_id, frame_list[frame]
        ).to('angstrom').m
        unitcell = _get_unitcell(multistate, replica_id, frame_list[frame])
        traj.variables['cell_lengths'][frame] = unitcell[:3]
        traj.variables['cell_angles'][frame] = unitcell[3:]

    # Make sure to clean up when you are done
    multistate.close()
    traj.close()
