from ..utilities import utils

class ParamContext():

    def __init__(self, desc):
        self.desc = desc
        self.slots = utils.gen_memory_slots(self.desc)
        self.size = sum(self.desc)