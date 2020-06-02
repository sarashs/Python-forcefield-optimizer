class ForceField(object):
    """General forcefield class, other forcefield classes are a subclass of this one"""
    def __init__(self, ff_filepath_input ,ParamSelect_filePath_input):
        """ff_filepath_input and ParamSelect_filePath_input are strings."""
        self.params={}
        self.param_selection=[]
        self.param_range=[]
        self.param_selected=0
        self.ff_filePath=ff_filepath_input
        self.ParamSelect_filePath=ParamSelect_filePath_input
        #self.item_number=[]
    def parseFile(self):
        pass
    def updateFile(self):
        pass
    def parseParamSelectionFile(self):
        pass
