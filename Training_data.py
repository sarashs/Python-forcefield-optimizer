import re
def lammps_input_creator(Input_structure_file="Inputstructurefile.txt",Input_forcefield='ffield.reax'):
    """
    This file creates the lammps inputs
    :param Input_structure_file:
    :param Input_forcefield:
    :return:
    """
    f=open(Input_structure_file,'r')
    l=f.readlines()
    for item in l:
           atom_type=0
           if '#structure ' in item:
              LAMMPS_Data_file=l[l.index(item)].replace('#structure ','').replace('\n','').replace(' ','')+".data"
              LAMMPS_Input_file=l[l.index(item)].replace('#structure ','').replace('\n','').replace(' ','')+".dat"
              s=open(LAMMPS_Input_file,'w')
              s.close()
              s=open(LAMMPS_Input_file,'a')
              #for n in lists:
              s.write('# 1.- Inizialization #######################\n')
              s.write('units real\n')
              s.write('  #mass = grams/mole\n')
              s.write('  #distance = Angstroms\n')
              s.write('  #time = femtoseconds\n')
              s.write('  #energy = kcal/mol\n')
              s.write('  #velocity = Angstroms/femtosecond\n')
              s.write('  #force = kcal/mol.Angstrom\n')
              s.write('  #torque = kcal/mole\n')
              s.write('  #temperature = degrees K\n')
              s.write('  #pressure = atmospheres (0.1013 GPa)\n')
              s.write('  #dynamic viscosity = Poise\n')
              s.write('  #charge = multiple of electron charge (+1.0 is a proton)\n')
              s.write('  #dipole = charge*Angstroms\n')
              s.write('  #electric field = volts/Angstrom\n')
              s.write('dimension 3\n')
              s.write('processors * * *\n')
              s.write('##\n')
              s.write('boundary p p p\n')
              s.write('atom_style charge\n\n# 2.- Atom definition ######################\n\n')
              s.write('atom_modify map hash\n')
              s.write('read_data   '+'scripts/'+LAMMPS_Data_file+'\n')
              s.write('\n# 3.- Force-Field ##########################\n\n')
              number_of_atoms=int(l[l.index(item)+1])
              #Forcefield params
              s.write('pair_style reax/c NULL\n')
              s.write('pair_coeff * * '+'scripts/'+Input_forcefield)
              #calculate number of atom types
              for item2 in l[(l.index(item)+3):]:
                 if not ('#dimensions' in item2):
                    atom_type=atom_type+1
                 else:
                    break
              #Add elements to the forcefield line
              for i in range(1,atom_type+1):
                 s.write(' '+l[l.index(item)+2+i][0:2])
              s.write('\n'+'fix 99 all qeq/reax 1 0.0 10.0 1.0e-6 reax/c\n')
              s.write('neighbor        2.0 bin\n')
              s.write('neigh_modify    every 10 check yes\n\n')
              s.write('## 4.- MD & relax parameters ################\n\n')
              ######
              s.write('dump DUMP2 all custom 1000000 '+ 'logs/lmp_logs/'+LAMMPS_Data_file.replace('.data','.out')+'.lammpstrj'+'id type x y z q #this size \n')
              s.write('thermo_style custom step etotal ke pe temp press pxx pyy pzz \n')
              s.write('thermo 1000000\n')
              #####fix restraints
              i=l.index(item)+atom_type+number_of_atoms+6
              if '#restrain' in l[i-1]:
                     s.write('fix holdem all restrain')
                     while i<len(l) and ('#structure ' not in l[i]):
                            s.write(' '+l[i].replace('\n',''))
                            i=i+1
              s.write('\n')
              s.write('fix_modify holdem energy yes\n')
              s.write('min_style cg\n')
              s.write('minimize 1.0e-7 1.0e-9 20000 20000\n')
              s.write('undump DUMP2\n')
           s.close()
    f.close()
def list_of_structures(Input_structure_file_path):
    Input_data_file_list=[]
    try:
        f=open(Input_structure_file_path,'U')
        l=f.readlines()
        for item in l:
            if '#structure ' in item:
                Input_data_file_list.append(item.replace('#structure ','').replace('\n','').replace(' ',''))
        f.close()
    except IOError:
        print('An error occured trying to read the training data file.')
        return Input_data_file_list
class Training_data(object):
# This would be a list like: self.training_energy=[[Weight_number,Coefficient_number,name_string,Coefficient_number,name_string,Energy_number],[],...]
    def __init__(self, train_data_filepath):
        self.training_charge= []
        self.training_energy= []
        self.training_charge_weight= 0
        self.training_energy_weight= 0
        try:
            temp_file=open(train_data_filepath,"r")
            temp_list=temp_file.readlines()
            Energy_pattern=re.compile(r'ENERGY')
            charge_pattern=re.compile(r'CHARGE')
            Energy_flag=False
            charge_flag=False
            for line_item in temp_list:
                if Energy_pattern.search(line_item):
                    Energy_flag=True
                    charge_flag=False
                    self.training_energy_weight=float(re.findall(r'[-+]?\d*\.\d+|\d+',line_item)[0])
                    continue
                if charge_pattern.search(line_item):
                    Energy_flag=False
                    charge_flag=True
                    self.training_charge_weight=float(re.findall(r'[-+]?\d*\.\d+|\d+',line_item)[0])
                    continue
                if (Energy_flag and not(charge_flag)):
                    if re.findall(r'[-+]?\d*\.\d+|\d+',line_item):
                        begining_of_first_string=re.search('\*',line_item).span()[0]+1
                        end_of_first_string=re.search('\s',line_item[begining_of_first_string:]).span()[0]+begining_of_first_string
                        begining_of_second_string=re.search('\*',line_item[end_of_first_string:]).span()[0]+1+end_of_first_string
                        end_of_second_string=re.search('\s',line_item[begining_of_second_string:]).span()[0]+begining_of_second_string
                        self.training_energy.append([float(re.findall(r'[-+]?\d*\.\d+|\d+',line_item)[0]),float(re.findall(r'[-+]?\d*\.\d+|\d+',line_item)[1]),line_item[begining_of_first_string:end_of_first_string],float(re.findall(r'[-+]?\d*\.\d+|\d+',line_item[end_of_first_string+1:])[0]),line_item[begining_of_second_string:end_of_second_string],float(re.findall(r'[-+]?\d*\.\d+|\d+',line_item[end_of_second_string+1:])[0])])
        except IOError:
            print('An error occured trying to read the training data file.')
