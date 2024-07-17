import os
import torch
import torch.nn.functional as F
import numpy as np

import utils.general as utils
from sdfusion.utils.demo_util import preprocess_image


from utils import rend_util
from glob import glob
import cv2
import random

import h5py

import torchvision.transforms as transforms

class RefinementDataset(torch.utils.data.Dataset):

    def __init__(self,
                 data_dir,
                 scan_id=0,
                 ):
        self.instance_dir = os.path.join('../data', data_dir, 'scan{0}'.format(scan_id))
        
        
        def glob_data(data_dir):
            data_paths = []
            data_paths.extend(glob(data_dir))
            data_paths = sorted(data_paths)
            return data_paths

        obj_paths = glob_data(os.path.join('{0}'.format(self.instance_dir), "obj", "*.h5"))

      
        self.n_objs = len(obj_paths)

        self.objs = []
        self.bboxes = []
        self.obj_ids = []
        
        for path in obj_paths :
            h5_f = h5py.File(path, 'r')
            sdf = h5_f['pc_sdf_sample'][:]
            sdf = torch.Tensor(sdf)
            bbox =  h5_f['obj_area'][:]
            obj_id =  h5_f['obj_id'][:][0]
            
            self.objs.append(sdf)
            self.bboxes.append(bbox)
            self.obj_ids.append(obj_id)



    def __len__(self):
        return len(self.objs)

    def __getitem__(self, idx):        
        object_id = self.obj_ids[idx]
        object = self.objs[idx]
        bbox = self.bboxes[idx]
        x0, x1, z0, z1, y0, y1 = bbox
        # object = object.view(x1-x0, z1-z0, y1-y0)
        
        return object_id, object, bbox

 