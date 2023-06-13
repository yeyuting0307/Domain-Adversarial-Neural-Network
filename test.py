#%%
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
# Datasets
from torchvision.datasets import MNIST
from util.datasets import MNIST_M

# Models
from util.models import FeatureExtractor, LabelClassifier, DomainClassifier


# -------- MNIST --------
mnist_root = "" # mnist_root
assert mnist_root != "", "Please set mnist_root in test.py"

# DataLoader Parameters 
BATCH = 1
WORKERS = 0
IS_SHUFFLE = False

source_transform = transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,)),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)) # 複製單通道轉為三通道
])

source_dataset_test = MNIST(
    root = mnist_root, 
    train=False, # Test set
    transform = source_transform, 
    download=False
)

source_dataloader_test = DataLoader(
    dataset = source_dataset_test, 
    batch_size = BATCH, 
    shuffle = IS_SHUFFLE, 
    num_workers = WORKERS
)



# -------- MNIST_M --------
test_img_folder = "" # mnist_m_test_root
test_label_file = "" # mnist_m_test_label_txt_file
assert test_img_folder != "", "Please set mnist_m_test_root in test.py"
assert test_label_file != "", "Please set mnist_m_test_label_txt_file in test.py"

target_transorm = transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor(),
    transforms.Normalize(mean=(.5,), std=(.5,))
])

target_dataset_test = MNIST_M(
    img_folder = test_img_folder, 
    label_file=test_label_file,
    img_transform = target_transorm
)

target_dataloader_test = DataLoader(
    dataset = target_dataset_test, 
    batch_size=BATCH, 
    shuffle=IS_SHUFFLE, 
    num_workers=WORKERS
)

#%%
ckpt = -1 # checkpoints/*_{ckpt}.pth
assert ckpt != -1, "Please set correct ckpt in test.py corresponding to the ckpt in checkpoints/"

device = torch.device('cpu')

fe = FeatureExtractor()
lc = LabelClassifier()
dc = DomainClassifier()

fe.load_state_dict(torch.load(f'./checkpoints/fe_{ckpt}.pth', map_location=device))
lc.load_state_dict(torch.load(f'./checkpoints/lc_{ckpt}.pth', map_location=device))
dc.load_state_dict(torch.load(f'./checkpoints/dc_{ckpt}.pth', map_location=device))

fe.to(device)
lc.to(device)
dc.to(device)

fe.eval()
lc.eval()
dc.eval()

#%%
src_acc = 0
src_count = 0
for i, source_data in enumerate(source_dataloader_test):
    # src-test
    src_x = source_data[0].to(device)
    src_y = source_data[1].to(device)
    src_feature = fe(src_x)
    src_label = lc(src_feature)

    if src_label.argmax(dim=1).cpu().item() == src_y.cpu().item():
        src_acc += 1
    src_count += 1

print(f"{src_acc} / {src_count} = {100*src_acc/src_count:.2f}%")

    
# %%
tar_acc = 0
tar_count = 0
for i, target_data in enumerate(target_dataloader_test):

    tar_x = target_data[0].to(device)
    tar_y = target_data[1] 

    # target-test
    tar_feature = fe(tar_x)
    tar_label = lc(tar_feature)

    if tar_label.argmax(dim=1).cpu().item() == tar_y.cpu().item():
        tar_acc += 1
    tar_count += 1

print(f"{tar_acc} / {tar_count} = {100*tar_acc/tar_count:.2f}%")

