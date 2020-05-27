from SA import *

c=SA("ffield1.reax","params","Trainingfile.txt","Inputstructurefile.txt")
c.anneal()
print(c.costs)
print([c.sol_,c.cost_])
