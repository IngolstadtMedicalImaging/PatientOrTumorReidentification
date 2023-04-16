import numpy as np
import tables as tb
import os
from typing import Tuple
import openslide

from numpy import random

class WSIObject():
    
    def __init__(self,db_entry,path_to_active_maps):
        # open slide object
        #self.slide_object = openslide.open_slide(db_entry['SLIDE_PATH'])
        self.slide_path = db_entry['SLIDE_PATH']
        # store general information
        self.patient_id = db_entry['PATIENT_ID']
        if 'TUMOR_ID' in db_entry:
            self.tumor_id = db_entry['TUMOR_ID']
        if 'TUMOR_CUT' in db_entry:
            self.tumor_cut = db_entry['TUMOR_CUT']
        if 'YEAR' in db_entry:
            self.year = db_entry['YEAR']
        if 'CLASSIFICATION_LABEL' in db_entry:
            self.label = db_entry['CLASSIFICATION_LABEL']
        # load path to activemap
        self.path_to_active_map = self.get_path_to_active_map(db_entry['FILENAME'],path_to_active_maps)
        
        # store slide information
        slide = self.open_slidefile()
        self.level_downsamples = slide.level_downsamples
        self.dimensions = slide.dimensions
        slide.close()

    def get_path_to_active_map(self,filename,path_to_active_maps):
        # find right activeMap
        files = os.listdir(path_to_active_maps)
        active_map_file = [file for file in files if file[:-3] == filename][0]
        return path_to_active_maps + active_map_file
    
    def load_active_map(self):
        '''loads the pre computed active map from disk'''
        table = tb.open_file(self.path_to_active_map, 'r')
        ds = table.root.ds[0]
        return table, ds
        
    
    def needs_calculation(self,center:Tuple[int,int],crop_size:int,level:int = 0, th:float = 0.7):
        '''checks wether a gives pair of coordinates has sufficient tissue coverage'''
        # load active map from disk
        table, ds = self.load_active_map()
        x,y = center

        factor = self.level_downsamples[level]
        # the coordinates always refer to level 0 when we sample from a slide. So we do not need to multiply them with the factor.
        x_ds = int(np.floor((float(x)) /ds))
        y_ds = int(np.floor((float(y)) /ds))
        step_ds = int(np.ceil((float(crop_size) * factor) /ds))

        if (np.sum(table.root.map[y_ds:y_ds+step_ds,x_ds:x_ds+step_ds]) / step_ds**2)>th:
            table.close()
            return True
        else:
            table.close()
            return False
        
    def sample_patches(self, bag_size, crop_size, random_level:bool = False, level: int = 0):
        '''samples a bag of patches from a wsi'''
        slide_object = self.open_slidefile()
        width, height = self.dimensions
        
        imgs = []
        for i in range(bag_size):
            is_matching = False
            while not is_matching:
                if random_level:
                    level = random.randint(low = 0, high = 3)
                else:
                    level = level

                center = random.randint(low = [0,0], high = [width - crop_size,height - crop_size], size = [1,2]).flatten()
                is_matching = self.needs_calculation(center, crop_size, level = level)
            imgs.append(slide_object.read_region(location = center,size = (crop_size,crop_size),level = level).convert('RGB'))        
        slide_object.close()
        return imgs

    def get_patch(self,location:Tuple[int,int], level:int, size:Tuple[int,int]):
        slide_object = self.open_slidefile()
        img = np.array(slide_object.read_region(location=location, level=level, size=size).convert('RGB'))
        slide_object.close()
        return img
    
    def open_slidefile(self):
        return openslide.open_slide(self.slide_path)