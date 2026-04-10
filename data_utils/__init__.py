from torch.utils.data import DataLoader
import os
import json
import glob
from .faceforensics import FaceForensics
from .celeb_df import CelebDF
from .ffiw import FFIW
from .dfdc_p import DFDCP
from .dfd import DFD


dataset_dict = {
    "FF++": FaceForensics,
    "Celeb-DF": CelebDF,
    "FFIW": FFIW,
    "DFDC-P": DFDCP,
    "DFD": DFD,
}

class MergeDataset(FaceForensics):

    def __init__(self, args, dataset_list, mode='train'):
        # super().__init__(config, mode)
        self.args = args
        self.mode = mode
        self.image_size = args.image_size
        self.transforms = self.get_transforms()

        self.image_paths = []
        self.labels = []

        for dataset in dataset_list:
            self.image_paths += dataset.image_paths
            self.labels += dataset.labels

        self.transforms = self.get_transforms()

        print("{} Data from 'Merge' loaded.\n".format(mode))
        print("Dataset contains {} images.\n".format(len(self.labels)))



class NewDataset(FaceForensics):


    def __init__(self, args, dataset_name, mode='test'):

        self.args = args
        self.mode = mode
        self.image_size = args.image_size
        self.root = args.dataset_root

        self.image_paths = []
        self.labels = []
        
        
        real_path = []
        
        
        with (open('./datasets/FaceForensics++/test.json', 'r', encoding='utf-8') as f):
            data_index = json.load(f)
            
        real_dir = os.path.join('./datasets/FaceForensics++', 'original_sequences', 'youtube', 'c23', 'frames')

        for i in data_index:
            real_path += glob.glob(real_dir + '/' + i[0] + '/*.png')
            real_path += glob.glob(real_dir + '/' + i[1] + '/*.png')

        if dataset_name in ['VQGAN', 'SiT', 'StyleGAN3', 'StyleGAN2', 'DiT', 'ddim', 'RDDM', 'StyleGANXL']:
        
            fake_path = glob.glob(os.path.join(self.root, 'FaceSwapping', dataset_name, 'ff', '*', '*'))#[:100]
        
        else:
            fake_path = glob.glob(os.path.join(self.root, 'FaceSwapping', dataset_name, 'ff', 'frames', '*', '*'))#[:100]


        print(dataset_name, len(real_path), len(fake_path))
        # print(real_path[0],fake_path[0])

        self.image_paths.extend(real_path)
        self.labels.extend([0] * len(real_path))
   

        self.image_paths.extend(fake_path)
        self.labels.extend([1] * len(fake_path))
        
 
        self.transforms = self.get_transforms()

        print("{} Data from {} loaded.\n".format(dataset_name, mode))
        print("Dataset contains {} images.\n".format(len(self.labels)))


def data_loader(args):
    if args.scenario == 'cross_dataset':
        train_datasets = []

        for train_set in args.training_sets:
            train_dataset = dataset_dict[train_set](args, mode='train')
            train_datasets.append(train_dataset)

        training_data = MergeDataset(args=args, dataset_list=train_datasets, mode='train')
        training_data_loader = DataLoader(training_data, batch_size=args.batch_size,
                                shuffle=True, num_workers=args.num_workers)
        test_data_loaders = {}

        for test_set in args.test_sets:

            test_dataset = dataset_dict[test_set](args, mode='test')

            test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                                              shuffle=False, num_workers=args.num_workers)
            test_data_loaders[test_set] = test_data_loader

    else:

        training_data = FaceForensics(args, mode='train', forgery_types=None)
        training_data_loader = DataLoader(training_data, batch_size=args.batch_size,
                                shuffle=True, num_workers=args.num_workers)

        test_data_loaders = {}
        
        
        test_sets = ['BlendFace', 'FaceDancer', 'FOMM', 'UniFace'] 
        
        
        for test_set in test_sets:
            test_dataset = NewDataset(args, dataset_name=test_set,  mode='test')
            
            test_data_loader = DataLoader(
                dataset=test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers
            )
            
            test_data_loaders[test_set] = test_data_loader



    return training_data_loader, test_data_loaders
