from glob import glob
from os.path import join
from .faceforensics import FaceForensics
import json


class DFD(FaceForensics):

    def __init__(self, args, mode='test'):
        self.args = args
        self.mode = mode
        self.image_size = args.image_size
        self.transforms = self.get_transforms()

        self.dataset_dir = join(self.args.dataset_root, 'DFD')

        self.image_paths = []
        self.labels = []


        real_images = glob(join(self.dataset_dir, 'real', 'c23', 'frames', '*', '*.png'))
        self.image_paths += real_images
        print('real', len(real_images))
        self.labels += [0] * len(real_images)
        fake_images = glob(join(self.dataset_dir, 'fake', 'c23', 'frames', '*', '*.png'))

        self.image_paths += fake_images
        self.labels += [1] * len(fake_images)
        print('fake', len(fake_images))
        num_images = len(self.image_paths)
        num_labels = len(self.labels)

        assert num_images == num_labels, "The number of images and targets not consistent."
        real_num = len(real_images)
        fake_num = len(fake_images)
        

        print(f"{mode} Data from 'DFD' loaded. Real: {real_num}, Fake {fake_num}. \n")
        print("Dataset contains {} images.\n".format(num_images))
