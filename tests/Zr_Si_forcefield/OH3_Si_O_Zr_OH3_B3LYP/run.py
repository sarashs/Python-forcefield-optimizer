import os

for file_name in os.listdir("./"):
    if file_name.endswith(".sh"):
        os.system("sbatch " + file_name)
