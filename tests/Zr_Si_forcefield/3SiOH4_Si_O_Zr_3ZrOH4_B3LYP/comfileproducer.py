# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 13:38:03 2020

@author: User
"""

import os

for file_name in os.listdir("./"):
    if file_name.endswith(".xyz"):
        s = open(file_name, 'r')
        f = open(file_name[:-4] + ".com", 'w')
        f.write("%NProcShared=16\n" + "#n B3LYP/GEN PSEUDO=READ Opt(modredundant)\n\n" + " Title\n\n" + "0 1\n")
        l = s.readlines()
        for item in l[2:]:
            f.write(item)
        s.close()
        f.write("\n"+"H O Si 0\n"+"6-31G(d,p)\n"+"****\n"+"Zr 0\n"+"LanL2DZ\n"+"****\n\n"+"Zr 0\n"+"LanL2DZ\n\n")
        f.close()