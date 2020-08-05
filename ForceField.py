import REAXConstants 
import re

class ForceField(object):
    """General forcefield class, other forcefield classes are a subclass of this one
    Attributes:
    ------------
   param_min_max_delta: dictionary {(tuple): {min: val, max: val, delta: val}}
   
   """
    def __init__(self, ff_filepath_input ,ParamSelect_filePath_input):
        """ff_filepath_input and ParamSelect_filePath_input are strings."""
        self.params = {}
        self.param_selection = []
        self.param_min_max_delta = {}
        # dictionary {(tuple): {min: val, max: val, delta: val}}
        self._param_selected = 0
        self.ff_filePath = ff_filepath_input
        self.ParamSelect_filePath = ParamSelect_filePath_input
        #self.item_number=[]
    def write_forcefield(self):
        pass
    def parseParamSelectionFile(self):
        pass
    
def list_to_dict(input_list):
    length = len(input_list)
    return {(i+1):input_list[i] for i in range(length)}

class REAX_FF(ForceField):
    """The reactive forcefield (REAX_FF) class. This is a subclass of the ForceField class"""
    def __init__(self,ff_filePath,ParamSelect_filePath):
        super().__init__(ff_filePath,ParamSelect_filePath)
        """This fucntion will parse the input forcefield file and write the parameters into a [#][#][#] dictionary
        this is corresponds to the input param file where you have # # # min max

    Attributes
    --------------
        param_selection: list of tuples [(a, b, c), ]
            n*3 is the list of reaxff parameters to be optimized

        params: dictionary
            list of parameters in the reaxff file format shape dict[section][entry][item].
            for general parameters it's always dict[section][item][1]

        selected_parameters_value: float array
            array of the values of the selected parameters

        """
        self.Param_read=None
        self.reaxFile = None
        self.Num_Of_Atoms=0
        self.Num_Of_BONDS=0
        self.Num_Of_OFF_DIAG=0
        self.Num_Of_ANGLES=0
        self.Num_Of_TORSIONS=0
        self.Num_Of_H_BONDS=0
        self.Num_Of_GENERAL=0
        self.removed_parts_of_FField=[]
        self.params[REAXConstants.GENERAL_NUM] = {} 
        self.params[REAXConstants.ANGLES_NUM] = {} 
        self.params[REAXConstants.ATOMS_NUM] = {} 
        self.params[REAXConstants.BONDS_NUM] = {}
        self.params[REAXConstants.H_BONDS_NUM] = {} 
        self.params[REAXConstants.OFF_DIAG_NUM] = {} 
        self.params[REAXConstants.TORSIONS_NUM] = {} 
        #self.ParamSelect_filePath=ParamSelect_filePath
        #self.ff_filePath=ff_filePath
        try:
            temp_file=open(self.ff_filePath,"r")
            self.reaxFile=temp_file.readlines()
        except IOError:
            print('An error occured trying to read the forcefield file.')
        GENERAL_pattern=re.compile(r'! Number of general')
        ATOMS_pattern=re.compile(r'! Nr of atoms')
        BONDS_pattern=re.compile(r'! Nr of bonds')
        OFF_DIAG_pattern=re.compile(r'! Nr of off')
        ANGLES_pattern=re.compile(r'! Nr of angles')
        TORSIONS_pattern=re.compile(r'! Nr of torsions')
        H_BONDS_pattern=re.compile(r'! Nr of hydrogen')
        for line_item in self.reaxFile:
            GENERAL_flag=GENERAL_pattern.search(line_item)
            ATOMS_flag=ATOMS_pattern.search(line_item)
            BONDS_flag=BONDS_pattern.search(line_item)
            OFF_DIAG_flag=OFF_DIAG_pattern.search(line_item)
            ANGLES_flag=ANGLES_pattern.search(line_item)
            TORSIONS_flag=TORSIONS_pattern.search(line_item)
            H_BONDS_flag=H_BONDS_pattern.search(line_item)
            if GENERAL_flag:
                self.Num_Of_GENERAL=int("".join(re.findall("\d",line_item[0:GENERAL_flag.span()[0]])))
                for j in range(self.Num_Of_GENERAL):
                    templine_index=j+self.reaxFile.index(line_item)+1
                    bogus=self.reaxFile[templine_index]
                    self.params[REAXConstants.GENERAL_NUM][j+1] = {}
                    self.params[REAXConstants.GENERAL_NUM][j+1][1]=float(bogus[0:re.search("!",bogus).span()[0]-1].replace(" ",""))
                    self.removed_parts_of_FField.append(["  ",bogus[re.search("!",bogus).span()[0]-1:].replace("\n","")])
            if ATOMS_flag:
                self.Num_Of_Atoms=int("".join(re.findall("\d",line_item[0:ATOMS_flag.span()[0]])))
                for j in range(self.Num_Of_Atoms):
                    temp_line=[]
                    for k in range(4):
                         templine_index=j*4+self.reaxFile.index(line_item)+4+k
                         if re.findall("[A-Z]"+"[a-z]",self.reaxFile[templine_index]):
                             self.removed_parts_of_FField.append(" "+re.findall("[A-Z]"+"[a-z]",self.reaxFile[templine_index])[0]+"   ")
                         elif re.findall("[A-Z]",self.reaxFile[templine_index]):
                             self.removed_parts_of_FField.append(" "+re.findall("[A-Z]",self.reaxFile[templine_index])[0]+"    ")
                         else: self.removed_parts_of_FField.append("      ")
                         bogus=re.sub("[a-z]","",re.sub("[A-Z]","",self.reaxFile[templine_index])).replace('\n','').split(" ")
                         while '' in bogus: bogus.remove('')
                         bogus=[float(i) for i in bogus]
                         temp_line=temp_line+bogus
                    self.params[REAXConstants.ATOMS_NUM][j+1] = list_to_dict(temp_line)
            if BONDS_flag:
                self.Num_Of_BONDS=int("".join(re.findall("\d",line_item[0:BONDS_flag.span()[0]])))
                for j in range(self.Num_Of_BONDS):
                    temp_line=[]
                    for k in range(2):
                         templine_index=j*2+self.reaxFile.index(line_item)+2+k
                         bogus=re.sub("[a-z]","",re.sub("[A-Z]","",self.reaxFile[templine_index])).replace('\n','').split(" ")
                         while '' in bogus: bogus.remove('')
                         if k==0:
                            self.removed_parts_of_FField.append("  "+bogus[0]+"  "+bogus[1]+" ")
                         else: self.removed_parts_of_FField.append("       ")
                         bogus=[float(i) for i in bogus]
                         if k==0:
                            temp_line=temp_line+bogus[2:]
                         else: temp_line=temp_line+bogus
                    self.params[REAXConstants.BONDS_NUM][j+1] = list_to_dict(temp_line)
            if OFF_DIAG_flag:
                self.Num_Of_OFF_DIAG=int("".join(re.findall("\d",line_item[0:OFF_DIAG_flag.span()[0]])))
                for j in range(self.Num_Of_OFF_DIAG):
                    temp_line=[]
                    for k in range(1):
                         templine_index=j*1+self.reaxFile.index(line_item)+1+k
                         bogus=re.sub("[a-z]","",re.sub("[A-Z]","",self.reaxFile[templine_index])).replace('\n','').split(" ")
                         while '' in bogus: bogus.remove('')
                         if k==0:
                            self.removed_parts_of_FField.append("  "+bogus[0]+"  "+bogus[1]+" ")
                         else: self.removed_parts_of_FField.append("       ")
                         bogus=[float(i) for i in bogus]
                         if k==0:
                            temp_line=temp_line+bogus[2:]
                         else: temp_line=temp_line+bogus
                    self.params[REAXConstants.OFF_DIAG_NUM][j+1] = list_to_dict(temp_line)
            if ANGLES_flag:
                self.Num_Of_ANGLES = int("".join(re.findall("\d",line_item[0:ANGLES_flag.span()[0]])))
                for j in range(self.Num_Of_ANGLES):
                    temp_line=[]
                    for k in range(1):
                         templine_index=j*1+self.reaxFile.index(line_item)+1+k
                         bogus=re.sub("[a-z]","",re.sub("[A-Z]","",self.reaxFile[templine_index])).replace('\n','').split(" ")
                         while '' in bogus: bogus.remove('')
                         self.removed_parts_of_FField.append("  "+bogus[0]+"  "+bogus[1]+"  "+bogus[2]+" ")
                         bogus=[float(i) for i in bogus]
                         temp_line=temp_line+bogus[3:]
                    self.params[REAXConstants.ANGLES_NUM][j+1] = list_to_dict(temp_line)
            if TORSIONS_flag:
                self.Num_Of_TORSIONS = int("".join(re.findall("\d",line_item[0:TORSIONS_flag.span()[0]])))
                for j in range(self.Num_Of_TORSIONS):
                    temp_line=[]
                    for k in range(1):
                         templine_index=j*1+self.reaxFile.index(line_item)+1+k
                         bogus=re.sub("[a-z]","",re.sub("[A-Z]","",self.reaxFile[templine_index])).replace('\n','').split(" ")
                         while '' in bogus: bogus.remove('')
                         self.removed_parts_of_FField.append("  "+bogus[0]+"  "+bogus[1]+"  "+bogus[2]+"  "+bogus[3]+" ")
                         bogus=[float(i) for i in bogus]
                         temp_line=temp_line+bogus[4:]
                    self.params[REAXConstants.TORSIONS_NUM][j+1] = list_to_dict(temp_line)
            if H_BONDS_flag:
                self.Num_Of_H_BONDS = int("".join(re.findall("\d",line_item[0:H_BONDS_flag.span()[0]])))
                for j in range(self.Num_Of_H_BONDS):
                    temp_line=[]
                    for k in range(1):
                         templine_index=j*1+self.reaxFile.index(line_item)+1+k
                         bogus=re.sub("[a-z]","",re.sub("[A-Z]","",self.reaxFile[templine_index])).replace('\n','').split(" ")
                         while '' in bogus: bogus.remove('')
                         self.removed_parts_of_FField.append("  "+bogus[0]+"  "+bogus[1]+"  "+bogus[2]+" ")
                         bogus=[float(i) for i in bogus]
                         temp_line=temp_line+bogus[3:]
                    self.params[REAXConstants.H_BONDS_NUM][j+1] = list_to_dict(temp_line)
        temp_file.close()
    def write_forcefield(self,Out_ff_filePath,params=None):
        """
        This function writes a forcefield based on params in to a file
        It can be called everytime the forcefield needs to be updated
        :param Out_ff_filePath:
        :param params:
        :return: null
        """
        #By default I use self.params for params
        if params is None:
            params = self.params
        #
        try:
            temp_file = open(Out_ff_filePath,"w")
        except IOError:
            print('An error occured trying to write the forcefield file.')
        temp_file.write("Reactive MD-force field optimized by GRACE optimization tool\n")
        temp_file.write(" "+str(self.Num_Of_GENERAL)+"       ! Number of general parameters\n")
        for i in range(self.Num_Of_GENERAL):
            temp_file.write(self.removed_parts_of_FField[i][0]+str(params[REAXConstants.GENERAL_NUM][i+1][1]).ljust(8)+self.removed_parts_of_FField[i][1]+"\n")
        temp_file.write("  "+str(self.Num_Of_Atoms).ljust(6)+"! Nr of atoms; cov.r; valency;a.m;Rvdw;Evdw;gammaEEM;cov.r2;#\n")
        temp_file.write("      alfa;gammavdW;valency;Eunder;Eover;chiEEM;etaEEM;n.u.\n")
        temp_file.write("      cov r3;Elp;Heat inc.;n.u.;n.u.;n.u.;n.u.\n")
        temp_file.write("      Rov/un;val1;n.u.;val3,vval4\n")
        for i in range(self.Num_Of_Atoms):
            for k in range(4):
                temp_file.write(self.removed_parts_of_FField[i*4+self.Num_Of_GENERAL+k]+ "".join(str(params[REAXConstants.ATOMS_NUM][i+1][j+1]).ljust(9) for j in range(k*8, (k+1)*8))+"\n")
        ####
        temp_file.write("  "+str(self.Num_Of_BONDS)+"     ! Nr of bonds; Edis1;LPpen;n.u.;pbe1;pbo5;13corr;pbo6 \n")
        temp_file.write("         pbe2;pbo3;pbo4;n.u.;pbo1;pbo2;ovcorr\n")
        for i in range(self.Num_Of_BONDS):
            for k in range(2):
                temp_file.write(self.removed_parts_of_FField[i*2+self.Num_Of_GENERAL+self.Num_Of_Atoms*4+k]+ "".join(str(params[REAXConstants.BONDS_NUM][i+1][j+1]).ljust(10) for j in range(k*8, (k+1)*8))+"\n")
        ####
        temp_file.write("  " + str(self.Num_Of_OFF_DIAG) + "     ! Nr of off-diagonal terms; Ediss;Ro;gamma;rsigma;rpi;rpi2\n")
        for i in range(self.Num_Of_OFF_DIAG):
            temp_file.write(self.removed_parts_of_FField[i+self.Num_Of_BONDS*2+self.Num_Of_GENERAL+self.Num_Of_Atoms*4]+ "".join(str(j).ljust(10) for j in list(params[REAXConstants.OFF_DIAG_NUM][i+1].values()))+"\n")
        ####
        temp_file.write("  "+str(self.Num_Of_ANGLES)+"     ! Nr of angles;at1;at2;at3;Thetao,o;ka;kb;pv1;pv2\n")
        for i in range(self.Num_Of_ANGLES):
            temp_file.write(self.removed_parts_of_FField[i+self.Num_Of_OFF_DIAG+self.Num_Of_BONDS*2+self.Num_Of_GENERAL+self.Num_Of_Atoms*4]+ "".join(str(j).ljust(10) for j in list(params[REAXConstants.ANGLES_NUM][i+1].values()))+"\n")
        ####
        temp_file.write("  "+str(self.Num_Of_TORSIONS)+"     ! Nr of torsions;at1;at2;at3;at4;;V1;V2;V3;V2(BO);vconj;n.u;n\n")
        for i in range(self.Num_Of_TORSIONS):
            temp_file.write(self.removed_parts_of_FField[i+self.Num_Of_ANGLES+self.Num_Of_OFF_DIAG+self.Num_Of_BONDS*2+self.Num_Of_GENERAL+self.Num_Of_Atoms*4]+ "".join(str(j).ljust(10) for j in list(params[REAXConstants.TORSIONS_NUM][i+1].values()))+"\n")
        ####
        temp_file.write("  "+str(self.Num_Of_H_BONDS)+"     ! Nr of hydrogen bonds;at1;at2;at3;Rhb;Dehb;vhb1\n")
        if self.Num_Of_H_BONDS>0:
            for i in range(self.Num_Of_H_BONDS):
                temp_file.write(self.removed_parts_of_FField[i+self.Num_Of_TORSIONS+self.Num_Of_ANGLES+self.Num_Of_OFF_DIAG+self.Num_Of_BONDS*2+self.Num_Of_GENERAL+self.Num_Of_Atoms*4]+ "".join(str(j).ljust(10) for j in list(params[REAXConstants.H_BONDS_NUM][i+1].values()))+"\n")
        temp_file.close()
    def parseParamSelectionFile(self):
        """This file reads the parameter file in
        builds the param_selection list and selected_parameters_value array
        """
        try:
            temp_file = open(self.ParamSelect_filePath,"r")
            self.Param_read=temp_file.readlines()
            self._param_selected=len(self.Param_read)
        except IOError:
            print('An error occured trying to read the param file.')
        self.Param_read=[i.replace("\n","").split(" ") for i in self.Param_read]
        for j in range(self._param_selected):
            while '' in self.Param_read[j]: self.Param_read[j].remove('')
        for i in range(self._param_selected):
            temp_tuple = (int(self.Param_read[i][0]), int(self.Param_read[i][1]), int(self.Param_read[i][2]))
            self.param_selection.append(temp_tuple)
            self.param_min_max_delta[temp_tuple] = {'delta' : float(self.Param_read[i][3]), 'min': float(self.Param_read[i][4]), 'max': float(self.Param_read[i][5])}
        temp_file.close()