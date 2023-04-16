import torch.utils.data as data_utils
import argparse
import pandas as pd
import numpy as np
from wsi_object import *
from augmentations import *
from sklearn.utils import shuffle

class SinglePatch(data_utils.Dataset):
    def __init__(self, data_csv, slide_level, patch_imgsize, transform, mode='train', pseudo_epoch_length:int = 1000, patches_valid:int = 100):
        self.pseudo_epoch_length = pseudo_epoch_length
        self.mode = mode
        self.slide_level = slide_level
        self.patch_imgsize = patch_imgsize
        self.csv = data_csv
        self.path_to_active_maps = 'data/active_maps/'
        self.transform = transform
        self.patches_valid = patches_valid

        self.df = pd.read_csv(self.csv, delimiter=",")
        self.df = shuffle(self.df, random_state=1)
        self.data_split()
        self.sampling_patients = self.__sample_patients()
        self.load_slids()


    def __sample_patients(self):
        # sample a list of patients for training
        patients = self.df.NUMERIC_ID.unique()
        if self.mode == 'train':
            indice = np.random.choice(len(patients), self.pseudo_epoch_length, replace=True)
        elif self.mode == 'valid':
            indice = sorted(self.patches_valid * list(range(len(patients))))

        return patients[indice]
    
    def resample_patients(self):
        self.sampling_patients = self.__sample_patients()
        print('training patients resampled\n')
    
    def __len__(self):
        if self.mode == 'train':
            epoch_len = self.pseudo_epoch_length
        elif self.mode == 'valid':
            epoch_len = len(self.sampling_patients)
        return epoch_len

    def __getitem__(self, index):
        patient_id = self.sampling_patients[index]
        patch = self.__getpatch__(self.df[self.df['NUMERIC_ID'] == patient_id].sample(n=1))

        return patch, patient_id

    def __getpatch__(self, row):
        slide = self.slides_loaded[row.iloc[0].SLIDE_PATH]
        patch = slide.sample_patches(bag_size=1, crop_size=self.patch_imgsize, level=self.slide_level)[0]

        if self.transform:
            patch = self.transform(patch)

        return patch

    def load_slids(self):
        self.slides_loaded = {}
        for ix, row in self.df.iterrows():
            self.slides_loaded[row.SLIDE_PATH] = WSIObject(row, self.path_to_active_maps)
    
    def data_split(self):
        train_data = []
        valid_data = []
        test_data = []
        valid_or_test = True

        for id in sorted(self.df.PATIENT_ID.unique()):
            patient_slides = self.df[self.df.PATIENT_ID == id]

            count_slides = len(patient_slides)

            if count_slides == 1:
                continue

            if count_slides == 2:
                train_slides = patient_slides.iloc[0]["ID"].tolist()
                if valid_or_test:
                    valid_slides = patient_slides.iloc[1]["ID"].tolist()
                    valid_data.append(valid_slides)
                    valid_or_test = False
                else:
                    test_slides = patient_slides.iloc[1]["ID"].tolist()
                    test_data.append(test_slides)
                    valid_or_test = True

                train_data.append(train_slides)

            if count_slides == 3:
                train_slides = patient_slides.iloc[0]["ID"].tolist()
                valid_slides = patient_slides.iloc[1]["ID"].tolist()
                test_slides = patient_slides.iloc[2]["ID"].tolist()
                    
                train_data.append(train_slides)
                valid_data.append(valid_slides)
                test_data.append(test_slides)
                
            if count_slides > 3:
                train_slides = patient_slides.iloc[0:count_slides-2]["ID"].tolist()
                valid_slides = patient_slides.iloc[-2]["ID"].tolist()
                test_slides = patient_slides.iloc[-1]["ID"].tolist()

                for train_slide in train_slides:
                    train_data.append(train_slide)
                valid_data.append(valid_slides)
                test_data.append(test_slides)


        # Patienten Pseudonym -> Numerische ID (0 bis #Patienten)
        self.id_mapping = {}
        classes = self.df[self.df['ID'].isin(train_data)].NUMERIC_ID.unique()
        for ix, id in enumerate(classes):
            self.id_mapping[id] = ix

        if self.mode == 'train':
            self.df = self.df[self.df['ID'].isin(train_data)]
        elif self.mode == 'valid':
            self.df = self.df[self.df['ID'].isin(valid_data)]
        elif self.mode == 'test':
            self.df = self.df[self.df['ID'].isin(test_data)]

        self.df = self.df.replace({"NUMERIC_ID": self.id_mapping})

    def get_num_classes(self) -> int:
        return self.df.PATIENT_ID.unique().size


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bag_size', type=int, default=10)
    parser.add_argument('--data_dir', type=str, default='data/Meningeome_1000_anon')
    parser.add_argument('--csv', type=str, default="data/final/reid_patches.csv")
    parser.add_argument('--slide_level', type=int, default=1)
    parser.add_argument('--patch_imgsize', type=int, default=512)
    args = parser.parse_args()

    dataset = SinglePatch(data_csv=args.csv, slide_level=args.slide_level, patch_imgsize=args.patch_imgsize, mode='valid', transform=CustomCompose([ToTensor(), ToPILImage()]))
    print(dataset.get_num_classes())
    print(dataset.__len__())

    # images = []
    # for ix in range(10):
    #     image, target = dataset.__getitem__(ix)
    #     images.append(image)

    # for ix, image in enumerate(images):
    #     image.save(f'single-resnet/example_patches/patch{ix}.png', format='PNG')
