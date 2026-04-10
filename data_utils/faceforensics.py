import json
import glob
import cv2
import os
from os.path import join
import albumentations as A

from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset


class FaceForensics(Dataset):

    def __init__(self, args, mode,
                 forgery_types=None):

        if forgery_types is None:
            forgery_types = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
        self.args = args
        self.mode = mode
        self.image_size = args.image_size
        self.transforms = self.get_transforms()
        self.dataset_dir = join(self.args.dataset_root, 'FaceForensics++')
        
        
        real_num  = 0
        fake_num = 0
        


        indices = join(self.dataset_dir, self.mode + ".json")
        real_path = os.path.join(self.dataset_dir, 'original_sequences', 'youtube', 'c23', 'frames')

        fake_paths = [os.path.join(self.dataset_dir, 'manipulated_sequences', ff, 'c23', 'frames')
                     for ff in forgery_types]

        with open(indices, 'r', encoding='utf-8') as f:
            folder_indices = json.load(f)

        self.image_paths, self.labels = [], []
        for i in folder_indices:
            real = glob.glob(real_path + '/' + i[0] + '/*.png')
            real += glob.glob(real_path + '/' + i[1] + '/*.png')

            self.image_paths.extend(real)
            self.labels.extend([0] * len(real))
            real_num = real_num + len(real)
            
    
            for fake_path in fake_paths:
                fake_0 = glob.glob(fake_path + '/' + i[0] + '_' + i[1] + '/*.png')
                fake_1 = glob.glob(fake_path + '/' + i[1] + '_' + i[0] + '/*.png')
                fake = fake_0 + fake_1
                self.image_paths.extend(fake)
                self.labels.extend([1] * len(fake))
                
                fake_num = fake_num + len(fake)
                
                
                
        num_images = len(self.image_paths)
        num_labels = len(self.labels)

        assert num_images == num_labels, "The number of images and targets not consistent."
        print(f"{mode} Data from 'FaceForensics++' loaded. Real: {real_num}, Fake {fake_num}. \n")
        print("Dataset contains {} images.\n".format(num_images))

    def get_transforms(self):

        if self.mode == 'train':
            train_aug = A.Compose([
                A.Resize(height=self.image_size , width=self.image_size),
                A.HorizontalFlip(p=0.5),
                A.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711]
                ),
                ToTensorV2()
            ])
            return train_aug
        else:
            test_aug = A.Compose([
                A.Resize(height=self.image_size, width=self.image_size),
                A.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711]
                ),
                ToTensorV2()
            ])
            return test_aug


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path, label = self.image_paths[index], self.labels[index]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)['image']

        return image, label





