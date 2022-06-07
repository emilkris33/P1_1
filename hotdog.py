import torch
import os
import glob
import PIL.Image as Image


import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class Hotdog_NotHotdog(torch.utils.data.Dataset):
    def __init__(self, train, transform, data_path='/dtu/datasets1/02514/hotdog_nothotdog'):
        'Initialization'
        self.transform = transform
        data_path = os.path.join(data_path, 'train' if train else 'test')
        image_classes = [os.path.split(d)[1] for d in glob.glob(data_path +'/*') if os.path.isdir(d)]
        image_classes.sort()
        self.name_to_label = {c: id for id, c in enumerate(image_classes)}
        self.image_paths = glob.glob(data_path + '/*/*.jpg')
        
    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_paths)

    def __getitem__(self, idx):
        'Generates one sample of data'
        image_path = self.image_paths[idx]
        
        image = Image.open(image_path)
        c = os.path.split(os.path.split(image_path)[0])[1]
        y = self.name_to_label[c]
        X = self.transform(image)
        return X, y

class dataset(torch.utils.data.Dataset):
    def __init__(self):
        size = 128
        transform_base = transforms.Compose([transforms.Resize((size, size)), 
                                            transforms.ToTensor()])
        transform_2 = transforms.Compose([transform_base, 
                                          transforms.RandomRotation(40)])
        transform_3 = transforms.Compose([transform_base, 
                                          transforms.RandomHorizontalFlip(1),
                                          transforms.RandomRotation(40)])
        batch_size = 64
        #self.trainset = Hotdog_NotHotdog(train=True, transform=transform_base)
        self.trainset = Hotdog_NotHotdog(train=True, transform=transform_2)
        self.trainset += Hotdog_NotHotdog(train=True, transform=transform_3)
        self.train_loader = DataLoader(self.trainset, batch_size=batch_size, shuffle=True, num_workers=3)


        self.testset = Hotdog_NotHotdog(train=False, transform=transform_base)
        self.test_loader = DataLoader(self.testset, batch_size=batch_size, shuffle=False, num_workers=3)