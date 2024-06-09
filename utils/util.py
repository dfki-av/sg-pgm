import os,json
import math
from collections import deque


color_palette = [
        (221, 34, 0),
       (174, 199, 232),		
       (152, 223, 138),		
       (31, 119, 180), 		
       (255, 187, 120),		
       (188, 189, 34), 		
       (140, 86, 75),  		
       (255, 152, 150),		
       (214, 39, 40),  		
       (197, 176, 213),		
       (148, 103, 189),		
       (196, 156, 148),		
       (23, 190, 207), 		
       (178, 76, 76),  
       (247, 182, 210),		
       (66, 188, 102), 
       (219, 219, 141),		
       (140, 57, 197), 
       (202, 185, 52), 
       (51, 176, 203), 
       (200, 54, 131), 
       (92, 193, 61),  
       (78, 71, 183),  
       (172, 114, 82), 
       (255, 127, 14), 		
       (91, 163, 138), 
       (153, 98, 156), 
       (140, 153, 101),
       (158, 218, 229),		
       (100, 125, 154),
       (178, 127, 135),
       (120, 185, 128),
       (146, 111, 194),
       (44, 160, 44),  		
       (112, 128, 144),		
       (96, 207, 209), 
       (227, 119, 194),		
       (213, 92, 176), 
       (94, 106, 211), 
       (82, 84, 163),  		
       (100, 85, 144)
    ]




def set_random_seed(seed):
    import random,torch
    import numpy as np
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def check_file_exist(path):
    if not os.path.exists(path):
            raise RuntimeError('Cannot open file. (',path,')')

def read_txt_to_list(file):
    output = [] 
    with open(file, 'r') as f: 
        for line in f: 
            entry = line.rstrip().lower() 
            output.append(entry) 
    return output

def check_file_exist(path):
    if not os.path.exists(path):
            raise RuntimeError('Cannot open file. (',path,')')
            

def read_classes(read_file):
    obj_classes = [] 
    with open(read_file, 'r') as f: 
        for line in f: 
            obj_class = line.rstrip().lower() 
            obj_classes.append(obj_class) 
    return obj_classes 


def read_relationships(read_file):
    relationships = [] 
    with open(read_file, 'r') as f: 
        for line in f: 
            relationship = line.rstrip().lower() 
            relationships.append(relationship) 
    return relationships 



def load_semseg(json_file, name_mapping_dict=None, mapping = True):    
    '''
    Create a dict that maps instance id to label name.
    If name_mapping_dict is given, the label name will be mapped to a corresponding name.
    If there is no such a key exist in name_mapping_dict, the label name will be set to '-'

    Parameters
    ----------
    json_file : str
        The path to semseg.json file
    name_mapping_dict : dict, optional
        Map label name to its corresponding name. The default is None.
    mapping : bool, optional
        Use name_mapping_dict as name_mapping or name filtering.
        if false, the query name not in the name_mapping_dict will be set to '-'
    Returns
    -------
    instance2labelName : dict
        Map instance id to label name.

    '''
    instance2labelName = {}
    with open(json_file, "r") as read_file:
        data = json.load(read_file)
        for segGroups in data['segGroups']:
            # print('id:',segGroups["id"],'label', segGroups["label"])
            # if segGroups["label"] == "remove":continue
            labelName = segGroups["label"]
            if name_mapping_dict is not None:
                if mapping:
                    if not labelName in name_mapping_dict:
                        labelName = 'none'
                    else:
                        labelName = name_mapping_dict[labelName]
                else:
                    if not labelName in name_mapping_dict.values():
                        labelName = 'none'

            instance2labelName[segGroups["id"]] = labelName.lower()#segGroups["label"].lower()
    return instance2labelName


class MovingAverage:
    """ Keeps an average window of the specified number of items. """

    def __init__(self, max_window_size=1000):
        self.max_window_size = max_window_size
        self.reset()

    def add(self, elem):
        """ Adds an element to the window, removing the earliest element if necessary. """
        if not math.isfinite(elem):
            print("Warning: Moving average ignored a value of %f" % elem)
            return

        self.window.append(elem)
        self.sum += elem

        if len(self.window) > self.max_window_size:
            self.sum -= self.window.popleft()

    def append(self, elem):
        """ Same as add just more pythonic. """
        self.add(elem)

    def reset(self):
        """ Resets the MovingAverage to its initial state. """
        self.window = deque()
        self.sum = 0

    def get_avg(self):
        """ Returns the average of the elements in the window. """
        return self.sum / max(len(self.window), 1)

    def __str__(self):
        return str(self.get_avg())

    def __repr__(self):
        return repr(self.get_avg())

    def __len__(self):
        return len(self.window)


class ProgressBar:
    """ A simple progress bar that just outputs a string. """

    def __init__(self, length, max_val):
        self.max_val = max_val
        self.length = length
        self.cur_val = 0

        self.cur_num_bars = -1
        self._update_str()

    def set_val(self, new_val):
        self.cur_val = new_val

        if self.cur_val > self.max_val:
            self.cur_val = self.max_val
        if self.cur_val < 0:
            self.cur_val = 0

        self._update_str()

    def is_finished(self):
        return self.cur_val == self.max_val

    def _update_str(self):
        num_bars = int(self.length * (self.cur_val / self.max_val))

        if num_bars != self.cur_num_bars:
            self.cur_num_bars = num_bars
            self.string = "█" * num_bars + "░" * (self.length - num_bars)

    def __repr__(self):
        return self.string

    def __str__(self):
        return self.string