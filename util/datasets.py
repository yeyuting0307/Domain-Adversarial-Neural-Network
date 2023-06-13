# =============== Torch Dataset ===============
## 繼承 torch.utils.data.Dataset
## 繼承後需要實作 __len__ 與 __getitem__ 兩個函數
## __len__ : 回傳資料集大小
## __getitem__ : 回傳一筆資料，通常是(data, label)
## 之後能夠用torch.utils.data.DataLoader來讀取
## 後續由DataLoader指定 batch_size, shuffle, num_workers 等參數

from torch.utils.data import Dataset
from PIL import Image
from glob import glob
import os

class MNIST_M(Dataset):
    def __init__(self, img_folder, label_file = None ,img_transform=None, label_transform=None):
        self.img_folder = img_folder
        self.img_paths = []
        self.img_transform = img_transform

        self.label_file = label_file
        # self.img_labels = []
        self.label_transform = label_transform

        # image_paths process
        self.img_paths = glob(os.path.join(img_folder, '*.png'))
        self.img_labels_map = {}

        # img_labels process
        if label_file:
            with open(label_file, 'r') as f:
                str_lines = f.readlines() # e.g. '00000021.png 0\n',
                self.img_labels_map = {line.split(' ')[0] : int(line.split(' ')[1].strip()) for line in str_lines}
                
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        img = Image.open(self.img_paths[index])
        img = img.convert('RGB')
        
        # 要用DataLoader讀取，就要將圖片轉成tensor, numpy arrays, numbers, dicts or lists
        if self.img_transform: 
            img = self.img_transform(img)

        label = []
        if self.img_labels_map != {}:
            path = self.img_paths[index].split('/')[-1]
            label = self.img_labels_map.get(path)
            if self.label_transform:
                label = self.label_transform(label)
            
        return img, label
