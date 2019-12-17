from torch.utils.data import Dataset,DataLoader
from PIL import  Image
from data_division import load_Fashion_Minst
from torchvision import transforms
import numpy as np
import time
import os



class FashionDataset(Dataset):

    def __init__(self,root_dir,transform=None,Train=True):
        self.root_dir = root_dir
        self.transform = transform
        mode = "train" if Train else "t10k"
        self.image,self.label = load_Fashion_Minst(self.root_dir,mode)  #根据模式确定返回的是训练集还是测试集


    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        image = self.image[idx]
        image = image.repeat(3,2)
        label = self.label[idx]
        image = Image.fromarray(np.uint8(image)).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image,label

#
# transform = transforms.Compose(
#     [
#         transforms.Resize((32,32)),
#         # transforms.CenterCrop(128),
#         # transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#     ]
# )
#
# if __name__ == '__main__':
#
#     data_path = r"H:\Dataset\FashionMNIST\raw"
#     dataset = FashionDataset(data_path,transform)
#     loader = DataLoader(dataset, shuffle=True, batch_size=10, num_workers=4)
#     loader = iter(loader)
#     image,label = next(loader)
#
#     time1 = time.time()
#     for i in range(500):
#         image, label = next(loader)
#     time2 = time.time()
#     print(time2-time1)
#
#     print("")