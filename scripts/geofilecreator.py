import re
LAMMPS_Input_file="LAMMPS_Input_file.dat"
Input_structure_file="Inputstructurefile.txt"
f=open(Input_structure_file,'r')
l=f.readlines()
for item in l:
       atom_type=0
       if '#structure ' in item:
          LAMMPS_Data_file=l[l.index(item)].replace('#structure ','').replace('\n','').replace(' ','')+".data"
          s=open(LAMMPS_Data_file,'w')
          s.close()
          s=open(LAMMPS_Data_file,'a')
          #for n in lists:
          s.write('# System description #######################\n')
          s.write('#\n')
          s.write('\n')
          #line for number of atoms
          s.write(l[l.index(item)+1].replace('\n','  atoms\n')) 
          number_of_atoms=int(l[l.index(item)+1])
          #line for atom types
          for item2 in l[(l.index(item)+3):]:
             if not ('#dimensions' in item2):
                atom_type=atom_type+1
             else:
                break   
          s.write('%d atom types\n' % atom_type)
          #line for box dimentions
          dimensions=re.findall(r"[-+]?\d*\.\d+|\d+",l[(l.index(item)+4+atom_type)])
          #s.write(l[(l.index(item)+3+atom_type)])
          s.write('0 %f xlo xhi\n' % float(dimensions[0]) )
          s.write('0 %f ylo yhi\n' % float(dimensions[1]) )
          s.write('0 %f zlo zhi\n' % float(dimensions[2]) )
          s.write('#\n')
          s.write('# for a crystal:\n')
          s.write('# lx=a;  ly2+xy2=b2;  lz2+xz2+yz2=c2\n')
          s.write('# xz=c*cos(beta);  xy=b*cos(gamma)\n')
          s.write('# xy*xz+ly*yz=b*c*cos(alpha)\n')
          s.write('#\n\n')
          s.write('# Elements #################################\n\n')
          s.write('Masses\n\n')
          # line for atomic masses
          for i in range(1,atom_type+1):
             s.write(l[l.index(item)+2+i].replace(l[l.index(item)+2+i][0:2],'%d ' % i))
          s.write('\nAtoms\n')
          for item2 in l[(l.index(item)+5+atom_type):(l.index(item)+5+atom_type+number_of_atoms)]:
             for i in range(1,atom_type+1):
                    item2=item2.replace(l[l.index(item)+2+i][0:2],'%d ' % i)
             s.write('\n'+item2.replace('\n',''))
       s.close()
f.close()
