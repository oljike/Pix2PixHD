import os
import glob
from PIL import Image

import torchvision.transforms as transforms
from torch.utils.data import Dataset

class DatasetFromFolder(Dataset):
    def __init__(self, img_path, edges_path, transform=None):
        super(DatasetFromFolder, self).__init__()

        self.edge_path = glob.glob(os.path.join(edges_path, '*.jpg'))
        self.img_path = [os.path.join(img_path, os.path.basename(fname)) for fname in self.edge_path
                         if os.path.isfile(os.path.join(img_path, os.path.basename(fname)))]

        transform_list = [transforms.ToTensor()
                          ]

        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):

        img = Image.open(self.img_path[index])
        img = img.resize((256, 128), Image.BICUBIC)
        img = self.transform(img)

        edge = Image.open(self.edge_path[index]).convert('L')
        edge = edge.resize((256, 128), Image.BICUBIC)
        edge = self.transform(edge)

        return img, edge

    def __len__(self):
        return len(self.img_path)

