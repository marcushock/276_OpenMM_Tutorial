import openmm
import openmm.app as openmm_app
import openmm.unit as unit
import md_min_equil_gentle

# Read the PSF
psf = openmm_app.CharmmPsfFile("charmm-gui-4183445964/step3_pbcsetup.psf")
psf.setBox(64.0*unit.angstroms, 64.0*unit.angstroms, 64.0*unit.angstroms)

# Get the coordinates from the PDB
unsolvated_pdb_filename = "charmm-gui-4183445964/step1_pdbreader.pdb"
solvated_pdb_filename = "charmm-gui-4183445964/step3_pbcsetup.pdb"
pdb = openmm_app.PDBFile(solvated_pdb_filename)

# Load the parameter set.
params = openmm_app.CharmmParameterSet(
    'charmm-gui-4183445964/toppar/par_all36m_prot.prm', 
    'charmm-gui-4183445964/toppar/top_all36_prot.rtf', 
    'charmm-gui-4183445964/toppar/toppar_water_ions.str')

# Create the system, defining periodic boundary conditions, a nonbonded cutoff, and constrain bonds with hydrogen to remain constant-length.
system = psf.createSystem(params, nonbondedMethod=openmm_app.PME, nonbondedCutoff=1.2*unit.nanometer, constraints=openmm_app.HBonds)
topology = psf.topology
temperature = 277.15 * unit.kelvin
target_pressure = 1.0 * unit.bar

simulation = md_min_equil_gentle.main(
    unsolvated_pdb_filename,
    solvated_pdb_filename,
    system,
    topology,
    temperature,
    target_pressure,
    is_membrane_system=False,
    stage_8_total_steps=125000,  # 0.5 ns at 4 fs timestep
    stage_8_timestep=0.002 * unit.picoseconds,
    eq_traj_interval=1000,
)

# Save a state file to maintain positions and velocities into the production stage.
state = simulation.context.getState(getPositions=True, getVelocities=True, getParameters=False, getIntegratorParameters=False)
xml = openmm.XmlSerializer.serialize(state)
with open('equil_done.xml', 'w') as f:
    f.write(xml)

