from glob import glob
from os.path import join
from os import listdir
from .faceforensics import FaceForensics


class CelebDF(FaceForensics):

    def __init__(self, args, mode):
        # super().__init__(config, mode)
        self.args = args
        self.mode = mode
        self.image_size = args.image_size
        self.transforms = self.get_transforms()
        self.dataset_dir = join(self.args.dataset_root, 'Celeb-DF')

        images_ids = self.__get_images_ids()
        test_ids = self.__get_test_ids()
        train_ids = [images_ids[0] - test_ids[0],
                     images_ids[1] - test_ids[1],
                     images_ids[2] - test_ids[2]]

        self.image_paths, self.labels = self.__get_images(
            test_ids if self.mode == "test" else train_ids)

        num_images = len(self.image_paths)
        num_labels = len(self.labels)

        assert num_images == num_labels, "The number of images and targets not consistent."
        print(f"{mode} Data from 'Celeb-DF' loaded.\n")
        print("Dataset contains {} images.\n".format(num_images))

    def __get_images_ids(self):
        youtube_real = listdir(join(self.dataset_dir, 'YouTube-real', 'frames'))
        celeb_real = listdir(join(self.dataset_dir, 'Celeb-real', 'frames'))
        celeb_fake = listdir(join(self.dataset_dir, 'Celeb-synthesis', 'frames'))
        return set(youtube_real), set(celeb_real), set(celeb_fake)

    def __get_test_ids(self):
        youtube_real = set()
        celeb_real = set()
        celeb_fake = set()
        with open(join(self.dataset_dir, "List_of_testing_videos.txt"), "r", encoding="utf-8") as f:
            contents = f.readlines()
            for line in contents:
                name = line.split(" ")[-1]
                number = name.split("/")[-1].split(".")[0]
                if "YouTube-real" in name:
                    youtube_real.add(number)
                elif "Celeb-real" in name:
                    celeb_real.add(number)
                elif "Celeb-synthesis" in name:
                    celeb_fake.add(number)
                else:
                    raise ValueError("'List_of_testing_videos.txt' file corrupted.")
        return youtube_real, celeb_real, celeb_fake

    def __get_images(self, ids):
        real = list()
        fake = list()
        # YouTube-real
        for _ in ids[0]:
            real.extend(glob(join(self.dataset_dir, 'YouTube-real', 'frames', _, '*.png')))
        # Celeb-real
        for _ in ids[1]:
            real.extend(glob(join(self.dataset_dir, 'Celeb-real', 'frames', _, '*.png')))
        # Celeb-synthesis
        for _ in ids[2]:
            fake.extend(glob(join(self.dataset_dir, 'Celeb-synthesis', 'frames', _, '*.png')))
        print("Real: {}, Fake: {}".format(len(real), len(fake)))

        real_tgt = [0] * len(real)
        fake_tgt = [1] * len(fake)
        return [*real, *fake], [*real_tgt, *fake_tgt]
