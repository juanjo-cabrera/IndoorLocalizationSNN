import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from PIL import Image
import torch
import PIL.ImageOps
from scipy.spatial import distance
from config import CONFIG

class TrainingDataset(Dataset):
    def __init__(self, imageFolderDataset, same=0.5, transform=None,should_invert=True):
        self.same_probability = same
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.should_invert = should_invert
    def __getitem__(self,index):
        def coordenadas(ruta):
           x_index = ruta.index('_x')
           y_index = ruta.index('_y')
           a_index = ruta.index('_a')
           x = ruta[x_index+2:y_index]
           y = ruta[y_index+2:a_index]
           coor_list = [x, y]
           coor = torch.from_numpy(np.array(coor_list, dtype=np.float32))
           return coor

        img0_tuple = random.choice(self.imageFolderDataset.imgs)
        coor0=coordenadas(img0_tuple[0])
        #we need to make sure approx 60% of images are different
        numberList = [1, 0]
        should_get_same_class = np.random.choice(numberList, 1, p=[self.same_probability, 1 - self.same_probability])
        if should_get_same_class:#Si sale 1 coge dos imágenes de la misma carpeta
            while True:
                #keep looping till the same class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1]==img1_tuple[1]: #compara los directorios para que sean iguales
                    break
        else:
            while True:
                #keep looping till a different class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] !=img1_tuple[1]:
                    break

        coor1 = coordenadas(img1_tuple[0])
        if img0_tuple[1]==img1_tuple[1]:
            dst = distance.euclidean(coor0,coor1)
            dst = dst/18.99                      #normalizamos los valores entre 0 y 1, siendo 16.98 la distancia máx en el pasillo
        else:
            dst = 1                  #normalizamos los valores entre 0 y 1, siendo 16.98 la distancia máx en el pasillo

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        img0 = img0.convert("RGB")
        img1 = img1.convert("RGB")

        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0) #transform to tensor
            img1 = self.transform(img1)

        return img0, img1, torch.from_numpy(np.array([dst],dtype=np.float32))

    def __len__(self):
        return len(self.imageFolderDataset.imgs)


class TestDataset(Dataset):
    def __init__(self, imageFolderDataset, transform=None, should_invert=True):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.should_invert = should_invert

    def __getitem__(self, index):
        def coordenadas(ruta):
            x_index = ruta.index('_x')
            y_index = ruta.index('_y')
            a_index = ruta.index('_a')
            x = ruta[x_index + 2:y_index]
            y = ruta[y_index + 2:a_index]
            coor_list = [x, y]
            coor = torch.from_numpy(np.array(coor_list, dtype=np.float32))
            return coor

        img_tuple = self.imageFolderDataset.imgs[index]
        img = self.imageFolderDataset[index][0]
        coor = coordenadas(img_tuple[0])
        return img, coor

    def __len__(self):
        return len(self.imageFolderDataset.imgs)





baseline_dataset = dset.ImageFolder(root=CONFIG.baseline_training_dir)

baseline_dataset = TrainingDataset(imageFolderDataset=baseline_dataset, same=0.5,
                                        transform=transforms.Compose([transforms.Resize((128, 512)),
                                                                      transforms.ToTensor()]),
                                        should_invert=False)
baseline_dataloader = DataLoader(baseline_dataset,
                            shuffle=True,
                            num_workers=0,
                            batch_size=16)

baseline_dataset_s20 = dset.ImageFolder(root=CONFIG.baseline_training_dir)
baseline_dataset_s20 = TrainingDataset(imageFolderDataset=baseline_dataset_s20, same=0.2,
                                        transform=transforms.Compose([transforms.Resize((128, 512)),
                                                                      transforms.ToTensor()]),
                                        should_invert=False)
baseline_dataloader_s20 = DataLoader(baseline_dataset_s20,
                            shuffle=True,
                            num_workers=0,
                            batch_size=16)
baseline_dataset_s30 = dset.ImageFolder(root=CONFIG.baseline_training_dir)
baseline_dataset_s30 = TrainingDataset(imageFolderDataset=baseline_dataset_s30, same=0.3,
                                        transform=transforms.Compose([transforms.Resize((128, 512)),
                                                                      transforms.ToTensor()]),
                                        should_invert=False)
baseline_dataloader_s30 = DataLoader(baseline_dataset_s30,
                            shuffle=True,
                            num_workers=0,
                            batch_size=16)
baseline_dataset_s40 = dset.ImageFolder(root=CONFIG.baseline_training_dir)
baseline_dataset_s40 = TrainingDataset(imageFolderDataset=baseline_dataset_s40, same=0.4,
                                        transform=transforms.Compose([transforms.Resize((128, 512)),
                                                                      transforms.ToTensor()]),
                                        should_invert=False)
baseline_dataloader_s40 = DataLoader(baseline_dataset_s40,
                            shuffle=True,
                            num_workers=0,
                            batch_size=16)

baseline_dataset_s60 = dset.ImageFolder(root=CONFIG.baseline_training_dir)
baseline_dataset_s60 = TrainingDataset(imageFolderDataset=baseline_dataset_s60, same=0.6,
                                        transform=transforms.Compose([transforms.Resize((128, 512)),
                                                                      transforms.ToTensor()]),
                                        should_invert=False)
baseline_dataloader_s60 = DataLoader(baseline_dataset_s60,
                            shuffle=True,
                            num_workers=0,
                            batch_size=16)
baseline_dataset_s70 = dset.ImageFolder(root=CONFIG.baseline_training_dir)
baseline_dataset_s70 = TrainingDataset(imageFolderDataset=baseline_dataset_s70, same=0.7,
                                        transform=transforms.Compose([transforms.Resize((128, 512)),
                                                                      transforms.ToTensor()]),
                                        should_invert=False)
baseline_dataloader_s70 = DataLoader(baseline_dataset_s70,
                            shuffle=True,
                            num_workers=0,
                            batch_size=16)
baseline_dataset_s80 = dset.ImageFolder(root=CONFIG.baseline_training_dir)
baseline_dataset_s80 = TrainingDataset(imageFolderDataset=baseline_dataset_s80, same=0.8,
                                        transform=transforms.Compose([transforms.Resize((128, 512)),
                                                                      transforms.ToTensor()]),
                                        should_invert=False)
baseline_dataloader_s80 = DataLoader(baseline_dataset_s80,
                            shuffle=True,
                            num_workers=0,
                            batch_size=16)



DA2_dataset_s50 = dset.ImageFolder(root=CONFIG.DA2_training_dir)

DA2_dataset_s50 = TrainingDataset(imageFolderDataset=DA2_dataset_s50, same=0.5,
                                        transform=transforms.Compose([transforms.Resize((128,512)),
                                                                      transforms.ToTensor()]),
                                        should_invert=False)
DA2_dataloader_s50 = DataLoader(DA2_dataset_s50,
                            shuffle=True,
                            num_workers=0,
                            batch_size=16)



DA2_dataset_s60 = dset.ImageFolder(root=CONFIG.DA2_training_dir)
DA2_dataset_s60 = TrainingDataset(imageFolderDataset=DA2_dataset_s60, same=0.6,
                                        transform=transforms.Compose([transforms.Resize((128,512)),
                                                                        transforms.ToTensor()]), 
                                         should_invert=False)                            
DA2_dataloader_s60 = DataLoader(DA2_dataset_s60,
                            shuffle=True,
                            num_workers=0,
                            batch_size=16)


DA3rot_dataset_s50 = dset.ImageFolder(root=CONFIG.DA3rot_training_dir)

DA3rot_dataset_s50 = TrainingDataset(imageFolderDataset=DA3rot_dataset_s50, same=0.5,
                                        transform=transforms.Compose([transforms.Resize((128,512)),
                                                                      transforms.ToTensor()]),
                                        should_invert=False)
DA3rot_dataloader_s50 = DataLoader(DA3rot_dataset_s50,
                            shuffle=True,
                            num_workers=0,
                            batch_size=16)



DA3rot_dataset_s60 = dset.ImageFolder(root=CONFIG.DA3rot_training_dir)
DA3rot_dataset_s60 = TrainingDataset(imageFolderDataset=DA3rot_dataset_s60, same=0.6,
                                        transform=transforms.Compose([transforms.Resize((128,512)),
                                                                        transforms.ToTensor()]),
                                         should_invert=False)
DA3rot_dataloader_s60 = DataLoader(DA3rot_dataset_s60,
                            shuffle=True,
                            num_workers=0,
                            batch_size=16)


test_data_cloudy = dset.ImageFolder(root=CONFIG.cloudy_test_dir, transform=transforms.Compose([transforms.Resize((128,512)),
                                                                      transforms.ToTensor()]))
test_dataset_cloudy = TestDataset(test_data_cloudy)
test_dataloader_cloudy = DataLoader(test_dataset_cloudy,
                        shuffle=False,
                        num_workers=0,
                        batch_size=1)

test_data_night = dset.ImageFolder(root=CONFIG.night_test_dir, transform=transforms.Compose([transforms.Resize((128,512)),
                                                                      transforms.ToTensor()]))
test_dataset_night = TestDataset(test_data_night)
test_dataloader_night = DataLoader(test_dataset_night,
                        shuffle=False,
                        num_workers=0,
                        batch_size=1)

test_data_sunny = dset.ImageFolder(root=CONFIG.sunny_tes_dir, transform=transforms.Compose([transforms.Resize((128,512)),
                                                                      transforms.ToTensor()]))
test_dataset_sunny = TestDataset(test_data_sunny)
test_dataloader_sunny = DataLoader(test_dataset_sunny,
                        shuffle=False,
                        num_workers=0,
                        batch_size=1)

test_data_global = dset.ImageFolder(root=CONFIG.global_test_dir, transform=transforms.Compose([transforms.Resize((128,512)),
                                                                      transforms.ToTensor()]))
test_dataset_global = TestDataset(test_data_global)
test_dataloader_global = DataLoader(test_dataset_global,
                        shuffle=False,
                        num_workers=0,
                        batch_size=1)


map_data = dset.ImageFolder(root=CONFIG.map_dir, transform=transforms.Compose([transforms.Resize((128,512)),
                                                                      transforms.ToTensor()]))
