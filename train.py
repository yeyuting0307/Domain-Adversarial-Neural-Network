#%%
import numpy as np
import time
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

# Datasets
from torchvision.datasets import MNIST
from util.datasets import MNIST_M

# Models
from util.models import FeatureExtractor, LabelClassifier, DomainClassifier


## 1. ================= Dataset Process =================
# DataLoader Parameters 
BATCH = 128
WORKERS = 0
IS_SHUFFLE = True

# -------- MNIST --------
mnist_root = "" # MNIST root
assert mnist_root != "", "Please set mnist_root in train.py"

source_transform = transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,)),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)) # 複製單通道轉為三通道
])

# train set
source_dataset = MNIST(
    root = mnist_root, 
    train=True, 
    transform = source_transform, 
    download=False
)

source_dataloader = DataLoader(
    dataset = source_dataset, 
    batch_size = BATCH, 
    shuffle = IS_SHUFFLE, 
    num_workers = WORKERS
)

# test set
source_dataset_test = MNIST(
    root = mnist_root, 
    train=False, # Test set
    transform = source_transform, 
    download=False
)

source_dataloader_test = DataLoader(
    dataset = source_dataset_test, 
    batch_size = 1, 
    shuffle = False, 
    num_workers = 0
)

# -------- MNIST_M --------
# train set
img_folder = "" # mnist_m_train_root
assert img_folder != "", "Please set img_folder in train.py"
target_transorm = transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor(),
    transforms.Normalize(mean=(.5,), std=(.5,))
])

target_dataset = MNIST_M(
    img_folder = img_folder, 
    img_transform = target_transorm
)

target_dataloader = DataLoader(
    dataset = target_dataset, 
    batch_size=BATCH, 
    shuffle=IS_SHUFFLE, 
    num_workers=WORKERS
)

# test set
test_img_folder = "" # mnist_m_test_root
test_label_file = "" # mnist_m_test_label_txt_file
assert test_img_folder != "", "Please set test_img_folder in train.py"
assert test_label_file != "", "Please set test_label_file in train.py"

target_dataset_test = MNIST_M(
    img_folder = test_img_folder, 
    label_file = test_label_file,
    img_transform = target_transorm
)

target_dataloader_test = DataLoader(
    dataset = target_dataset_test, 
    batch_size=1, 
    shuffle=False, 
    num_workers=0
)

## 2. ================= Model Define =================
fe = FeatureExtractor()
lc = LabelClassifier()
dc = DomainClassifier()


# Loss Function
label_criterion = torch.nn.CrossEntropyLoss()
domain_criterion = torch.nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.SGD(
    list(fe.parameters()) + 
    list(lc.parameters()) +
    list(dc.parameters())
    , lr = 0.01
)

#%%
## 3. ================= Model Train =================
EPOCH = 101
device = torch.device('cpu')

fe.to(device)
lc.to(device)
dc.to(device)

for epoch in range(EPOCH):
    S = time.time()
    for i, (source_data, target_data) in enumerate(zip(source_dataloader, target_dataloader)):
        # ------- init -------
        fe.train()
        lc.train()
        dc.train()
        optimizer.zero_grad()

        # ------- source process -------
        src_x = source_data[0].to(device)
        src_y = source_data[1].to(device)

        # FeatureExtractor
        src_feature = fe(src_x)
        
        # LabelClassifier
        src_label = lc(src_feature)
        src_label_loss = label_criterion(src_label, src_y)

        # DomainClassifier
        GAMMA = 10. # from paper
        P = (1 + epoch) / (1 + EPOCH)
        lambda_ = 2 / (1 + np.exp(-GAMMA * P)) - 1
        src_domain = dc(src_feature, lambda_)
        src_domain_y = torch.zeros(src_domain.size(0)).type(torch.long).to(device)

        # ------- target process -------
        tar_x = target_data[0].to(device)
        # tar_y = target_data[1] # [] 

        # FeatureExtractor
        tar_feature = fe(tar_x)

        # DomainClassifier
        tar_domain = dc(tar_feature, lambda_)
        tar_domain_y = torch.ones(tar_domain.size(0)).type(torch.long).to(device)

        domain_loss = domain_criterion(
            torch.concat((src_domain, tar_domain), dim=0),
            torch.concat((src_domain_y, tar_domain_y), dim=0)
        )

        # ------- loss and optimizer -------
        # from paper
        MU_0 = 0.01 
        ALPHA = 10 
        BETA = 0.75 
        MU_P = MU_0 / ((1 + ALPHA * P)**BETA) 
        optimizer.lr = MU_P

        loss = src_label_loss + domain_loss
        loss.backward()
        optimizer.step()

    # ------- print -------
    E = time.time()
    if epoch % 1 == 0:
        with torch.no_grad():
            print(f"Epoch: {epoch}, Time: {round(E-S)} sec, " + 
                f"loss: {loss.cpu().item() :.3f}, " +
                f"label_loss: {src_label_loss.cpu().item():.3f}, " + 
                f"domain_loss: {domain_loss.cpu().item():.3f}\n"
            )
    # ------- save  -------
    if epoch % 10 == 0:
        torch.save(fe.state_dict(), f"./checkpoints/fe_{epoch}.pth")
        torch.save(lc.state_dict(), f"./checkpoints/lc_{epoch}.pth")
        torch.save(dc.state_dict(), f"./checkpoints/dc_{epoch}.pth")

    # ------- evaluate -------
    if epoch % 5 == 0:
        fe.eval()
        lc.eval()
        dc.eval()

        src_label_acc = 0
        src_domain_acc = 0
        src_count = 0

        for i, test_source_data in enumerate(source_dataloader_test):
            # src-test
            src_x = test_source_data[0].to(device)
            src_y = test_source_data[1].to(device)
            with torch.no_grad():
                src_feature = fe(src_x)
                src_label = lc(src_feature)
                src_domain = dc(src_feature, 0)
                
                if src_label.argmax(dim=1).cpu().item() == src_y.cpu().item():
                    src_label_acc += 1
                if src_domain.argmax(dim=1).cpu().item() == 0:
                    src_domain_acc += 1
                src_count += 1

        print(f"[src label] {src_label_acc} / {src_count} = {100*src_label_acc/src_count:.2f}%")
        print(f"[src domain] {src_domain_acc} / {src_count} = {100*src_domain_acc/src_count:.2f}%")


        tar_label_acc = 0
        tar_domain_acc = 0
        tar_count = 0
        for i, test_target_data in enumerate(target_dataloader_test):
            tar_x = test_target_data[0].to(device)
            tar_y = test_target_data[1].to(device)
            with torch.no_grad():
                tar_feature = fe(tar_x)
                tar_label = lc(tar_feature)
                tar_domain = dc(tar_feature, 0)
                
                if tar_label.argmax(dim=1).cpu().item() == tar_y.cpu().item():
                    tar_label_acc += 1
                if tar_domain.argmax(dim=1).cpu().item() == 1:
                    tar_domain_acc += 1
                tar_count += 1

        print(f"[tar label] {tar_label_acc} / {tar_count} = {100*tar_label_acc/tar_count:.2f}%")
        print(f"[tar domain] {tar_domain_acc} / {tar_count} = {100*tar_domain_acc/tar_count:.2f}%")

