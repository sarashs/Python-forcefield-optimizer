import os

for file_name in os.listdir("./"):
    if file_name.endswith(".xyz"):
        s = open(file_name, 'r')
        f = open(file_name[:-4] + ".sh", 'w')
        f.write("#!/bin/bash\n" + "#SBATCH --account=def-ivanov\n" + "#SBATCH --mem=16G \n" + "#SBATCH --time=0-10:00\n" + "#SBATCH --output=Bash_"+file_name[:-4]+".log \n" + "#SBATCH --cpus-per-task=16\n\n" + "module load gaussian/g16.b01\n")
        s.close()
        f.write("g16 < " + file_name[:-4] + ".com >& " + file_name[:-4] + ".log\n")
        f.close()