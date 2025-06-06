import os
from ase import Atoms
from ase.calculators.gaussian import Gaussian

h_atom = Atoms('H', positions=[(0, 0, 0)])

def calc_dft_energy(atoms, label, chg):
    calc_opt = Gaussian(mem='8GB',
                        nprocshared=24,
                        label=os.path.join('/home/ppdeshmu/scratch/', f"{label}"),
                        xc='PBE1PBE',
                        basis='def2TZVP',
                        command='g16 < PREFIX.com> PREFIX.log',
                        charge=chg,
                        extra='EmpiricalDispersion=GD3BJ'
                        )

    atoms.calc = calc_opt
    energy = atoms.get_potential_energy()

    return energy

atoms_dict = {
    'hydride': [-1],
    'proton': [1]
}

h2kcal = 627.509

with open('/home/ppdeshmu/Generative-FLP/data/proton_hydride_energies.txt', 'w') as f:
    for a_type, charge in atoms_dict.items():
        a_energy = calc_dft_energy(h_atom, label=a_type, chg=charge)
        a_energy_kcal = a_energy*h2kcal
        f.write(f"{a_type}: {a_energy_kcal} Kcal/mol")
    f.close()