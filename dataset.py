import os
import glob
from PIL import Image

import torchvision.transforms as transforms
from torch.utils.data import Dataset

class DatasetFromFolder(Dataset):
    def __init__(self, img_path, landmarks_path, transform=None):
        super(DatasetFromFolder, self).__init__()

        #         self.img_path = glob.glob(os.path.join(img_path), '*.jpg')
        self.landmarks_path = glob.glob(os.path.join(landmarks_path, '*.jpg'))
        self.img_path = [os.path.join(img_path, os.path.basename(fname)) for fname in self.landmarks_path
                         if os.path.isfile(os.path.join(img_path, os.path.basename(fname)))]

        transform_list = [transforms.ToTensor()
                          ]

        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        # Load Image}

        img = Image.open(self.img_path[index])  # .convert('RGB')
        img = img.resize((256, 128), Image.BICUBIC)
        img = self.transform(img)

        landmark = Image.open(self.landmarks_path[index]).convert('L')
        landmark = landmark.resize((256, 128), Image.BICUBIC)
        landmark = self.transform(landmark)

        return img, landmark

    def __len__(self):
        return len(self.img_path)

