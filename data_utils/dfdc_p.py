from glob import glob
from os.path import join
from .faceforensics import FaceForensics
import json
import numpy as np

class DFDCP(FaceForensics):

    def __init__(self, args, mode):
        # super().__init__(config, mode)
        self.args = args
        self.mode = mode
        self.image_size = args.image_size
        self.transforms = self.get_transforms()
        self.dataset_dir = join(self.args.dataset_root, 'DFDC-P')

        real_num  = 0
        fake_num = 0
        
        indices = join(self.dataset_dir, "dataset.json")
        with open(indices, 'r', encoding='utf-8') as f:
            folder_indices = json.load(f)

        label_dict = {'real': 0, 'fake': 1}
        self.image_paths = []
        self.labels = []
        for k, v in folder_indices.items():
            if v['set'] == self.mode:
                k = k.replace('.mp4', '')
                k = k.split('/')
                k = join(k[0], k[-1])
                path_list = glob(join(self.dataset_dir, k, '*'))
                self.image_paths.extend(path_list)
                self.labels.extend([label_dict[v['label']]] * len(path_list))
                
                if label_dict[v['label']] == 1:
                    fake_num = fake_num + len(path_list)
                    
                else:
                    real_num = real_num + len(path_list)
                
                    

        num_images = len(self.image_paths)
        num_labels = len(self.labels)

        assert num_images == num_labels , "The number of images and targets not consistent."
        print(f"{mode} Data from 'DFDC-P' loaded. Real: {real_num}, Fake: {fake_num} \n")
        print("Dataset contains {} images.\n".format(num_images))
  