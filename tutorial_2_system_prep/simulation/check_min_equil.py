"""Check the resulting minimizations and equilibrations by plotting energies
and analyzing the final structure for correct chirality, etc.

This is done according to Carlos Simmerling's "Guide to setting up
a new system and doing equilibrations".
"""

import argparse
import numpy as np
import mdtraj
import matplotlib.pyplot as plt
import pandas as pd

from Burbuja import burbuja

# Parse command line arguments
parser = argparse.ArgumentParser(description='Check equilibration quality and analyze structure')
parser.add_argument('-m', '--membrane', action='store_true', 
                    help='Enable membrane mode (plots area and thickness)')
args = parser.parse_args()

eq_traj_filename = "eq_traj.pdb"
eq_state_filename = "eq_state.dat"
output_pdb_file = "equilibrated.pdb"

# Generate plots

print("Generating plots from energy data...")

# Read the energy data from both equilibration phases
# Read eq1 data (no header in current files)
eq_data = pd.read_csv(eq_state_filename, header=None)
eq_data.columns = ['Step', 'Potential_Energy_kcal_mol', 'Kinetic_Energy_kcal_mol', 
                    'Total_Energy_kcal_mol', 'Temperature_K', 'Box_Volume_nm3',
                    'Box_XY_Area_nm2', 'Bilayer_Thickness_nm',
                    'Bond_Energy_kcal_mol', 'Angle_Energy_kcal_mol', 
                    'Dihedral_Energy_kcal_mol', 'Nonbonded_Energy_kcal_mol']

# Combine the data - adjust eq4 timesteps to be continuous from eq1
combined_data = pd.concat([eq_data], ignore_index=True)

# Convert timesteps to time in nanoseconds (assuming 2 fs timestep)
timestep_fs = 2.0  # femtoseconds
combined_data['Time_ns'] = combined_data['Step'] * timestep_fs / 1000000.0  # convert fs to ns

# Plot 1: Total Energy over Time
plt.figure(figsize=(10, 6))
plt.plot(combined_data['Time_ns'], combined_data['Total_Energy_kcal_mol'], 'b-', linewidth=0.8)
plt.xlabel('Time (ns)')
plt.ylabel('Total Energy (kcal/mol)')
plt.title('Total Energy vs Time During Equilibration')
plt.grid(True, alpha=0.3)

# Add caption as a text box
caption_text = "Check the total energy at every step of your equilibration.\nDoes it get very high? Why?"
plt.figtext(0.5, 0.02, caption_text, ha='center', va='bottom', 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8),
            fontsize=9, style='italic')

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)  # Make room for the caption
plt.savefig('total_energies_over_time.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"Successfully created total_energies_over_time.png")
print(f"Data spans {combined_data['Time_ns'].min():.3f} to {combined_data['Time_ns'].max():.3f} ns")
print(f"Total energy range: {combined_data['Total_Energy_kcal_mol'].min():.1f} to {combined_data['Total_Energy_kcal_mol'].max():.1f} kcal/mol")

# Plot 2: Potential Energy, Kinetic Energy, and Volume over Time (multiline)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Top subplot: Potential and Kinetic Energy
ax1.plot(combined_data['Time_ns'], combined_data['Potential_Energy_kcal_mol'], 'r-', 
            linewidth=0.8, label='Potential Energy')
ax1.plot(combined_data['Time_ns'], combined_data['Kinetic_Energy_kcal_mol'], 'b-', 
            linewidth=0.8, label='Kinetic Energy')
ax1.set_ylabel('Energy (kcal/mol)')
ax1.set_title('Energy Components and Volume vs Time During Equilibration')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right')

# Bottom subplot: Volume
ax2.plot(combined_data['Time_ns'], combined_data['Box_Volume_nm3'], 'g-', 
            linewidth=0.8, label='Box Volume')
ax2.set_xlabel('Time (ns)')
ax2.set_ylabel('Volume (nm³)')
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper right')

# Add caption as a text box
caption_text = "Check that these quantities reach plateau values and have no unusual peaks."
plt.figtext(0.5, 0.02, caption_text, ha='center', va='bottom', 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8),
            fontsize=9, style='italic')

plt.tight_layout()
plt.subplots_adjust(bottom=0.12)  # Make room for the caption
plt.savefig('energy_components_and_volume_over_time.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"Successfully created energy_components_and_volume_over_time.png")

# Plot 3: Individual Energy Components (Bond, Angle, Dihedral, Nonbonded) - Split for better visibility
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Top subplot: Bond, Angle, and Dihedral energies (smaller scale)
ax1.plot(combined_data['Time_ns'], combined_data['Bond_Energy_kcal_mol'], 'orange', 
            linewidth=0.8, label='Bond Energy')
ax1.plot(combined_data['Time_ns'], combined_data['Angle_Energy_kcal_mol'], 'red', 
            linewidth=0.8, label='Angle Energy')
ax1.plot(combined_data['Time_ns'], combined_data['Dihedral_Energy_kcal_mol'], 'purple', 
            linewidth=0.8, label='Dihedral Energy')
ax1.set_ylabel('Energy (kcal/mol)')
ax1.set_title('Individual Energy Components vs Time During Equilibration')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right')

# Bottom subplot: Nonbonded energy (larger scale)
ax2.plot(combined_data['Time_ns'], combined_data['Nonbonded_Energy_kcal_mol'], 'blue', 
            linewidth=0.8, label='Nonbonded Energy')
ax2.set_xlabel('Time (ns)')
ax2.set_ylabel('Energy (kcal/mol)')
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper right')

# Add caption as a text box
caption_text = "Pay special attention to bond and angle energies. They should end up low and never spike.\nIncreasing values usually indicate some structural strain that the equilibration won't fix, such as steric overlap."
plt.figtext(0.5, 0.02, caption_text, ha='center', va='bottom', 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8),
            fontsize=9, style='italic')

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)  # Make room for the caption
plt.savefig('individual_energy_components_over_time.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"Successfully created individual_energy_components_over_time.png")
print(f"Bond energy range: {combined_data['Bond_Energy_kcal_mol'].min():.1f} to {combined_data['Bond_Energy_kcal_mol'].max():.1f} kcal/mol")
print(f"Angle energy range: {combined_data['Angle_Energy_kcal_mol'].min():.1f} to {combined_data['Angle_Energy_kcal_mol'].max():.1f} kcal/mol")
print(f"Dihedral energy range: {combined_data['Dihedral_Energy_kcal_mol'].min():.1f} to {combined_data['Dihedral_Energy_kcal_mol'].max():.1f} kcal/mol")
print(f"Nonbonded energy range: {combined_data['Nonbonded_Energy_kcal_mol'].min():.1f} to {combined_data['Nonbonded_Energy_kcal_mol'].max():.1f} kcal/mol")

# Plot 4: Membrane properties (XY area and bilayer thickness) - only if membrane mode
if args.membrane:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Top subplot: XY membrane area
    ax1.plot(combined_data['Time_ns'], combined_data['Box_XY_Area_nm2'], 'purple', 
                linewidth=0.8, label='Membrane XY Area')
    ax1.set_ylabel('XY Area (nm²)')
    ax1.set_title('Membrane Properties vs Time During Equilibration')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    
    # Bottom subplot: Bilayer thickness
    ax2.plot(combined_data['Time_ns'], combined_data['Bilayer_Thickness_nm'], 'teal', 
                linewidth=0.8, label='Bilayer Thickness')
    ax2.set_xlabel('Time (ns)')
    ax2.set_ylabel('Thickness (nm)')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    
    # Add caption as a text box
    caption_text = "Check that membrane area and thickness reach stable plateau values.\nArea should equilibrate to match lipid composition. Thickness indicates proper bilayer structure."
    plt.figtext(0.5, 0.02, caption_text, ha='center', va='bottom', 
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lavender', alpha=0.8),
                fontsize=9, style='italic')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for the caption
    plt.savefig('membrane_properties_over_time.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Successfully created membrane_properties_over_time.png")
    print(f"XY Area range: {combined_data['Box_XY_Area_nm2'].min():.2f} to {combined_data['Box_XY_Area_nm2'].max():.2f} nm²")
    print(f"Bilayer thickness range: {combined_data['Bilayer_Thickness_nm'].min():.2f} to {combined_data['Bilayer_Thickness_nm'].max():.2f} nm")

# Structural Analysis of Final Structure
print("\n" + "="*60)
print("STRUCTURAL ANALYSIS OF EQUILIBRATED STRUCTURE")
print("="*60)

# Load the final equilibrated structure
final_structure = mdtraj.load(output_pdb_file)
#print("Testing chirality code to see if mirror image produces D-amino acids...")
#final_structure.xyz *= -1  # Invert coordinates to create mirror image

print(f"Loaded final structure: {output_pdb_file}")
print(f"Number of frames: {final_structure.n_frames}")
print(f"Number of atoms: {final_structure.n_atoms}")
print(f"Number of residues: {final_structure.n_residues}")

# Analyze peptide bond conformations (cis/trans)
print("\nAnalyzing peptide bond conformations...")

# Find all peptide bonds (omega dihedrals: CA(i-1) - C(i-1) - N(i) - CA(i))
peptide_bonds = []
cis_bonds = []
trans_bonds = []

# Get protein residues (exclude water, ions, etc.)
# TODO: use a more general protein selection method - mdtraj select "protein"
protein_indices = final_structure.topology.select('protein')
protein_residues = [res for res in final_structure.topology.residues \
                    if any(atom.index in protein_indices for atom in res.atoms)]
#protein_residues = [res for res in final_structure.topology.residues 
#                    if res.name in ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLU', 'GLN', 'GLY', 
#                                    'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 
#                                    'THR', 'TRP', 'TYR', 'VAL', 'HIE', 'HID', 'HIP', 'CYX']]

print(f"Found {len(protein_residues)} protein residues")

# Loop through consecutive protein residues to find peptide bonds
for i in range(len(protein_residues) - 1):
    res_i = protein_residues[i]
    res_j = protein_residues[i + 1]
    
    # Check if residues are consecutive in sequence
    if res_j.resSeq == res_i.resSeq + 1:
        try:
            # Find atoms for omega dihedral: CA(i) - C(i) - N(i+1) - CA(i+1)
            ca_i = [atom for atom in res_i.atoms if atom.name == 'CA'][0]
            c_i = [atom for atom in res_i.atoms if atom.name == 'C'][0]
            n_j = [atom for atom in res_j.atoms if atom.name == 'N'][0]
            ca_j = [atom for atom in res_j.atoms if atom.name == 'CA'][0]
            
            # Calculate omega dihedral angle
            atom_indices = [ca_i.index, c_i.index, n_j.index, ca_j.index]
            omega_angle = mdtraj.compute_dihedrals(final_structure, [atom_indices])[0][0]
            omega_degrees = np.degrees(omega_angle)
            omega_degrees %= 360.0  # Normalize to [0, 360]
            
            peptide_bonds.append({
                'residue_i': res_i.resSeq,
                'residue_j': res_j.resSeq,
                'res_name_i': res_i.name,
                'res_name_j': res_j.name,
                'omega_angle': omega_degrees,
                'atom_indices': atom_indices
            })
            
            # Classify as cis or trans
            # Cis: omega angle between -90 and 90 degrees
            # Trans: omega angle between 90 and 270 degrees (or -90 and -270)
            if omega_degrees < 90 or omega_degrees > 270:
                cis_bonds.append((res_i.resSeq, res_j.resSeq, omega_degrees))
            else:
                trans_bonds.append((res_i.resSeq, res_j.resSeq, omega_degrees))
                
        except (IndexError, KeyError) as e:
            print(f"Warning: Could not analyze peptide bond between {res_i.name}{res_i.resSeq} and {res_j.name}{res_j.resSeq}: {e}")

# Report results
total_peptide_bonds = len(peptide_bonds)
n_cis = len(cis_bonds)
n_trans = len(trans_bonds)

print(f"\nPeptide Bond Analysis Results:")
print(f"Total peptide bonds analyzed: {total_peptide_bonds}")
print(f"Cis peptide bonds: {n_cis} ({100*n_cis/total_peptide_bonds:.1f}%)")
print(f"Trans peptide bonds: {n_trans} ({100*n_trans/total_peptide_bonds:.1f}%)")

# Print details of cis bonds (these are unusual and worth noting)
if n_cis > 0:
    print(f"\nDetailed analysis of CIS peptide bonds:")
    print("Res1-Res2    Names       Omega Angle")
    print("-" * 40)
    for res_i, res_j, omega in cis_bonds:
        res_name_i = next(r.name for r in protein_residues if r.resSeq == res_i)
        res_name_j = next(r.name for r in protein_residues if r.resSeq == res_j)
        print(f"{res_i:3d}-{res_j:3d}      {res_name_i}-{res_name_j}     {omega:8.1f}°")

# Check if there are any proline residues (which have higher cis propensity)
proline_bonds = [bond for bond in peptide_bonds if bond['res_name_j'] == 'PRO']
if proline_bonds:
    print(f"\nProline peptide bonds (higher cis propensity expected):")
    n_pro_cis = sum(1 for bond in proline_bonds if abs(bond['omega_angle']) < 90)
    n_pro_trans = len(proline_bonds) - n_pro_cis
    print(f"Proline bonds: {len(proline_bonds)} total, {n_pro_cis} cis ({100*n_pro_cis/len(proline_bonds):.1f}%), {n_pro_trans} trans")

# Save detailed results to file
with open('peptide_bond_analysis.txt', 'w') as f:
    f.write("Peptide Bond Conformation Analysis\n")
    f.write("="*50 + "\n")
    f.write(f"Structure: {output_pdb_file}\n")
    f.write(f"Total peptide bonds: {total_peptide_bonds}\n")
    f.write(f"Cis bonds: {n_cis} ({100*n_cis/total_peptide_bonds:.1f}%)\n")
    f.write(f"Trans bonds: {n_trans} ({100*n_trans/total_peptide_bonds:.1f}%)\n\n")
    
    f.write("Detailed bond information:\n")
    f.write("Residue1\tResidue2\tName1\tName2\tOmega_Angle\tConformation\n")
    for bond in peptide_bonds:
        conformation = "Cis" if abs(bond['omega_angle']) < 90 else "Trans"
        f.write(f"{bond['residue_i']}\t{bond['residue_j']}\t{bond['res_name_i']}\t{bond['res_name_j']}\t{bond['omega_angle']:.1f}\t{conformation}\n")

print(f"\nDetailed results saved to: peptide_bond_analysis.txt")

# Analyze alpha carbon chiralities
print("\n" + "-"*50)
print("ALPHA CARBON CHIRALITY ANALYSIS")
print("-"*50)

def calculate_tetrahedral_chirality(ca_pos, n_pos, c_pos, cb_pos, h_pos):
    """
    Calculate chirality using signed volume of tetrahedron.
    Returns positive for L-chirality, negative for D-chirality.
    Uses the standard L-amino acid reference: N, CA, C, CB in that priority order.
    """
    # Vectors from CA to each substituent
    v1 = n_pos - ca_pos    # CA -> N
    v2 = c_pos - ca_pos    # CA -> C  
    v3 = cb_pos - ca_pos   # CA -> CB
    
    # Calculate signed volume using scalar triple product: v1 · (v2 × v3)
    cross_product = np.cross(v2, v3)
    signed_volume = np.dot(v1, cross_product)
    
    return signed_volume

chirality_results = []
l_chirality_count = 0
d_chirality_count = 0
invalid_count = 0

# Analyze each non-glycine residue
for res in protein_residues:
    if res.name != 'GLY':  # Skip glycine (no chirality)
        try:
            # Find the required atoms
            ca_atom = [atom for atom in res.atoms if atom.name == 'CA'][0]
            n_atom = [atom for atom in res.atoms if atom.name == 'N'][0]
            c_atom = [atom for atom in res.atoms if atom.name == 'C'][0]
            cb_atom = [atom for atom in res.atoms if atom.name == 'CB'][0]
            
            # Find hydrogen bonded to CA
            h_atom = None
            for atom in res.atoms:
                if atom.element.symbol == 'H':
                    # Check if this hydrogen is bonded to CA by distance
                    ca_pos = final_structure.xyz[0, ca_atom.index]
                    h_pos = final_structure.xyz[0, atom.index]
                    distance = np.linalg.norm(ca_pos - h_pos)
                    if distance < 0.15:  # 1.5 Angstroms in nm
                        h_atom = atom
                        break
            
            if h_atom is None:
                raise KeyError(f"No hydrogen found bonded to CA in {res.name}{res.resSeq}")
            
            # Get positions
            ca_pos = final_structure.xyz[0, ca_atom.index]
            n_pos = final_structure.xyz[0, n_atom.index]
            c_pos = final_structure.xyz[0, c_atom.index]
            cb_pos = final_structure.xyz[0, cb_atom.index]
            h_pos = final_structure.xyz[0, h_atom.index]
            
            # Calculate chirality
            signed_vol = calculate_tetrahedral_chirality(ca_pos, n_pos, c_pos, cb_pos, h_pos)
            
            # L-amino acids should have positive signed volume with this arrangement
            if signed_vol > 0:
                chirality = 'L'
                l_chirality_count += 1
            else:
                chirality = 'D'
                d_chirality_count += 1
            
            chirality_results.append({
                'residue': res.resSeq,
                'res_name': res.name,
                'signed_volume': signed_vol,
                'chirality': chirality
            })
            
        except (IndexError, KeyError) as e:
            print(f"Warning: Could not analyze chirality for {res.name}{res.resSeq}: {e}")
            invalid_count += 1
            chirality_results.append({
                'residue': res.resSeq,
                'res_name': res.name,
                'signed_volume': None,
                'chirality': 'Unknown'
            })

# Report chirality results
total_analyzed = len([r for r in chirality_results if r['chirality'] != 'Unknown'])
print(f"\nChirality Analysis Results:")
print(f"Total non-glycine residues analyzed: {total_analyzed}")
if total_analyzed > 0:
    print(f"L-chirality (natural): {l_chirality_count} ({100*l_chirality_count/total_analyzed:.1f}%)")
    print(f"D-chirality (unnatural): {d_chirality_count} ({100*d_chirality_count/total_analyzed:.1f}%)")
print(f"Unknown/invalid: {invalid_count}")

# Calculate statistics for valid chiralities
if total_analyzed > 0:
    valid_volumes = [r['signed_volume'] for r in chirality_results if r['signed_volume'] is not None]
    if valid_volumes:
        mean_vol = np.mean(valid_volumes)
        std_vol = np.std(valid_volumes)
        print(f"\nSigned volume statistics (nm³):")
        print(f"Mean: {mean_vol:.6f}")
        print(f"Standard deviation: {std_vol:.6f}")
        print(f"Range: {min(valid_volumes):.6f} to {max(valid_volumes):.6f}")

# Report any D-chirality residues (these are problematic)
d_residues = [r for r in chirality_results if r['chirality'] == 'D']
if d_residues:
    print(f"\nWARNING: Found {len(d_residues)} residues with D-chirality (unnatural):")
    print("Residue  Name    Signed Volume")
    print("-" * 30)
    for r in d_residues:
        print(f"{r['residue']:3d}      {r['res_name']}     {r['signed_volume']:10.6f}")
    print("\nD-chirality residues indicate serious structural problems!")
else:
    print("\nEXCELLENT: All residues have natural L-chirality!")

# Save chirality results to file
with open('chirality_analysis.txt', 'w') as f:
    f.write("Alpha Carbon Chirality Analysis\n")
    f.write("="*40 + "\n")
    f.write(f"Structure: {output_pdb_file}\n")
    f.write(f"Total residues analyzed: {total_analyzed}\n")
    if total_analyzed > 0:
        f.write(f"L-chirality: {l_chirality_count} ({100*l_chirality_count/total_analyzed:.1f}%)\n")
        f.write(f"D-chirality: {d_chirality_count} ({100*d_chirality_count/total_analyzed:.1f}%)\n")
    f.write(f"Unknown: {invalid_count}\n\n")
    
    f.write("Method: Signed tetrahedral volume around alpha carbon\n")
    f.write("Atoms: N, CA, C, CB (and H for validation)\n")
    f.write("L-chirality: positive signed volume\n")
    f.write("D-chirality: negative signed volume\n\n")
    
    f.write("Detailed chirality information:\n")
    f.write("Residue\tName\tSigned_Volume\tChirality\n")
    for r in chirality_results:
        vol_str = f"{r['signed_volume']:.6f}" if r['signed_volume'] is not None else "N/A"
        f.write(f"{r['residue']}\t{r['res_name']}\t{vol_str}\t{r['chirality']}\n")

print(f"\nDetailed chirality results saved to: chirality_analysis.txt")

# Analyze side chain chiralities for Isoleucine and Threonine
print("\n" + "-"*50)
print("SIDE CHAIN CHIRALITY ANALYSIS (ILE & THR)")
print("-"*50)

# Find Ile and Thr residues
ile_residues = [res for res in protein_residues if res.name == 'ILE']
thr_residues = [res for res in protein_residues if res.name == 'THR']

print(f"Found {len(ile_residues)} Isoleucine residues")
print(f"Found {len(thr_residues)} Threonine residues")

side_chain_results = []
ile_l_count = 0
ile_d_count = 0
thr_l_count = 0
thr_d_count = 0
ile_invalid = 0
thr_invalid = 0

# Analyze Isoleucine side chains
print("\nAnalyzing Isoleucine side chain chiralities...")
for res in ile_residues:
    try:
        # For Isoleucine: chiral center at CB bonded to CA, CG1, CG2, and H
        # Need to find all four substituents of the CB chiral center
        ca_atom = [atom for atom in res.atoms if atom.name == 'CA'][0]
        cb_atom = [atom for atom in res.atoms if atom.name == 'CB'][0]
        cg1_atom = [atom for atom in res.atoms if atom.name == 'CG1'][0]
        cg2_atom = [atom for atom in res.atoms if atom.name == 'CG2'][0]
        
        # Find hydrogen bonded to CB
        h_atom = None
        for atom in res.atoms:
            if atom.element.symbol == 'H':
                # Check if this hydrogen is bonded to CB by distance
                cb_pos = final_structure.xyz[0, cb_atom.index]
                h_pos = final_structure.xyz[0, atom.index]
                distance = np.linalg.norm(cb_pos - h_pos)
                if distance < 0.15:  # 1.5 Angstroms in nm
                    h_atom = atom
                    break
        
        if h_atom is None:
            raise KeyError(f"No hydrogen found bonded to CB in ILE{res.resSeq}")
        
        # Get positions
        cb_pos = final_structure.xyz[0, cb_atom.index]
        ca_pos = final_structure.xyz[0, ca_atom.index]
        cg1_pos = final_structure.xyz[0, cg1_atom.index]
        cg2_pos = final_structure.xyz[0, cg2_atom.index]
        h_pos = final_structure.xyz[0, h_atom.index]
        
        # Calculate chirality using signed volume around CB
        # Order: CA, CB, CG1, CG2 (with H for validation)
        signed_vol = calculate_tetrahedral_chirality(cb_pos, ca_pos, cg1_pos, cg2_pos, h_pos)
        
        # For L-Isoleucine, the signed volume should be positive with this arrangement
        if signed_vol > 0:
            chirality = 'L'
            ile_l_count += 1
        else:
            chirality = 'D'
            ile_d_count += 1
        
        side_chain_results.append({
            'residue': res.resSeq,
            'res_name': 'ILE',
            'signed_volume': signed_vol,
            'chirality': chirality
        })
        
    except (IndexError, KeyError) as e:
        print(f"Warning: Could not analyze ILE{res.resSeq}: {e}")
        ile_invalid += 1
        side_chain_results.append({
            'residue': res.resSeq,
            'res_name': 'ILE',
            'signed_volume': None,
            'chirality': 'Unknown'
        })

# Analyze Threonine side chains
print("Analyzing Threonine side chain chiralities...")
for res in thr_residues:
    try:
        # For Threonine: chiral center at CB bonded to CA, CG2, OG1, and H
        # Need to find all four substituents of the CB chiral center
        ca_atom = [atom for atom in res.atoms if atom.name == 'CA'][0]
        cb_atom = [atom for atom in res.atoms if atom.name == 'CB'][0]
        og1_atom = [atom for atom in res.atoms if atom.name == 'OG1'][0]
        cg2_atom = [atom for atom in res.atoms if atom.name == 'CG2'][0]
        
        # Find hydrogen bonded to CB
        h_atom = None
        for atom in res.atoms:
            if atom.element.symbol == 'H':
                # Check if this hydrogen is bonded to CB by distance
                cb_pos = final_structure.xyz[0, cb_atom.index]
                h_pos = final_structure.xyz[0, atom.index]
                distance = np.linalg.norm(cb_pos - h_pos)
                if distance < 0.15:  # 1.5 Angstroms in nm
                    h_atom = atom
                    break
        
        if h_atom is None:
            raise KeyError(f"No hydrogen found bonded to CB in THR{res.resSeq}")
        
        # Get positions
        cb_pos = final_structure.xyz[0, cb_atom.index]
        ca_pos = final_structure.xyz[0, ca_atom.index]
        og1_pos = final_structure.xyz[0, og1_atom.index]
        cg2_pos = final_structure.xyz[0, cg2_atom.index]
        h_pos = final_structure.xyz[0, h_atom.index]
        
        # Calculate chirality using signed volume around CB
        # Order: CA, CB, OG1, CG2 (with H for validation)
        signed_vol = calculate_tetrahedral_chirality(cb_pos, ca_pos, og1_pos, cg2_pos, h_pos)
        
        # For L-Threonine, the signed volume should be positive with this arrangement
        if signed_vol > 0:
            chirality = 'L'
            thr_l_count += 1
        else:
            chirality = 'D'
            thr_d_count += 1
        
        side_chain_results.append({
            'residue': res.resSeq,
            'res_name': 'THR',
            'signed_volume': signed_vol,
            'chirality': chirality
        })
        
    except (IndexError, KeyError) as e:
        print(f"Warning: Could not analyze THR{res.resSeq}: {e}")
        thr_invalid += 1
        side_chain_results.append({
            'residue': res.resSeq,
            'res_name': 'THR',
            'signed_volume': None,
            'chirality': 'Unknown'
        })

# Report side chain chirality results
print(f"\nSide Chain Chirality Analysis Results:")

if len(ile_residues) > 0:
    total_ile = ile_l_count + ile_d_count
    if total_ile > 0:
        print(f"\nIsoleucine residues:")
        print(f"  Total analyzed: {total_ile}")
        print(f"  L-chirality: {ile_l_count} ({100*ile_l_count/total_ile:.1f}%)")
        print(f"  D-chirality: {ile_d_count} ({100*ile_d_count/total_ile:.1f}%)")
        print(f"  Invalid: {ile_invalid}")
    else:
        print(f"\nIsoleucine: No valid residues analyzed ({ile_invalid} invalid)")

if len(thr_residues) > 0:
    total_thr = thr_l_count + thr_d_count
    if total_thr > 0:
        print(f"\nThreonine residues:")
        print(f"  Total analyzed: {total_thr}")
        print(f"  L-chirality: {thr_l_count} ({100*thr_l_count/total_thr:.1f}%)")
        print(f"  D-chirality: {thr_d_count} ({100*thr_d_count/total_thr:.1f}%)")
        print(f"  Invalid: {thr_invalid}")
    else:
        print(f"\nThreonine: No valid residues analyzed ({thr_invalid} invalid)")

# Report any D-chirality side chains
d_ile = [r for r in side_chain_results if r['res_name'] == 'ILE' and r['chirality'] == 'D']
d_thr = [r for r in side_chain_results if r['res_name'] == 'THR' and r['chirality'] == 'D']

if d_ile or d_thr:
    print(f"\nWARNING: Found residues with D-chirality side chains:")
    if d_ile:
        print("D-Isoleucine residues:")
        for r in d_ile:
            print(f"  ILE{r['residue']}: signed volume = {r['signed_volume']:.6f}")
    if d_thr:
        print("D-Threonine residues:")
        for r in d_thr:
            print(f"  THR{r['residue']}: signed volume = {r['signed_volume']:.6f}")
else:
    print(f"\nEXCELLENT: All Ile/Thr residues have natural L-chirality side chains!")

# Save side chain chirality results to file
with open('sidechain_chirality_analysis.txt', 'w') as f:
    f.write("Side Chain Chirality Analysis (Ile & Thr)\n")
    f.write("="*45 + "\n")
    f.write(f"Structure: {output_pdb_file}\n")
    f.write(f"Isoleucine residues: {len(ile_residues)} found, {ile_l_count + ile_d_count} analyzed\n")
    f.write(f"Threonine residues: {len(thr_residues)} found, {thr_l_count + thr_d_count} analyzed\n\n")
    
    f.write("Method: Signed tetrahedral volume around beta carbon (CB)\n")
    f.write("Ile: CB bonded to CA, CG1, CG2, H\n")
    f.write("Thr: CB bonded to CA, OG1, CG2, H\n")
    f.write("L-chirality: positive signed volume\n")
    f.write("D-chirality: negative signed volume\n\n")
    
    f.write("Detailed side chain chirality information:\n")
    f.write("Residue\tType\tSigned_Volume\tChirality\n")
    for r in side_chain_results:
        vol_str = f"{r['signed_volume']:.6f}" if r['signed_volume'] is not None else "N/A"
        f.write(f"{r['residue']}\t{r['res_name']}\t{vol_str}\t{r['chirality']}\n")

print(f"\nDetailed side chain chirality results saved to: sidechain_chirality_analysis.txt")

# Check for bubbles:

has_bubbles = burbuja.has_bubble(output_pdb_file)
if has_bubbles:
    print(f"WARNING: bubble detected in {output_pdb_file}.")
    bubble_detected = "YES"
else:
    print("No bubble detected.")
    bubble_detected = "no"

print("\n" + "="*60)
print("Next steps:")
print("1. Load the plots (.png files) individually to examine the equilibration quality.")
if args.membrane:
    print("   - Membrane mode: Also check membrane_properties_over_time.png")
print("2. Load the following files into VMD to visualize the trajectory:")
print(f"  - {eq_traj_filename}")
print(f"  - {output_pdb_file}")
print(f"  Pay special attention to any ligands or other non-standard residues, ")
print(f"   does the geometry (bonds, angles, planarity) look correct?")
print("3. Review structural analysis files:")
print(f"   - peptide_bond_analysis.txt (cis/trans peptide bonds: non-proline: {n_cis}/{n_trans}, proline: {n_pro_cis}/{n_pro_trans})")
print(f"   - chirality_analysis.txt (alpha carbon chiralities: L: {l_chirality_count}, D: {d_chirality_count})")
print("   - sidechain_chirality_analysis.txt (Ile & Thr side chain chiralities)")
print(f"    Thr L: {thr_l_count}, Ile L: {ile_l_count}, Thr D: {thr_d_count}, Ile D: {ile_d_count}")
print("4. Pay attention to any D-chirality residues - these may indicate local structural issues.")
print("   Natural proteins should have ~100% L-chirality for both backbone and side chains.")
print(f"5. Bubble detected? {bubble_detected} - simulations should not have any bubbles")
