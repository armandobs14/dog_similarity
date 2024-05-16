from dataset_object_siamese import ImagePairDataset
from torch.utils.data import DataLoader
from model import SiameseNetwork
from loss import ContrastiveLoss
import torch.optim as optim
import torch

from train import train
from eval import eval
import os

# Loading .env
from dotenv import load_dotenv
load_dotenv()

# Train test Split
dataset = ImagePairDataset(os.getenv("BASE_PATH"))
print(os.getenv("BASE_PATH"))
print(len(dataset))
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, test_size]
)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


# Creating loss function
net = SiameseNetwork()
if os.getenv("CUDA", "DISABLED") == "ENABLED":
    net = net.cuda()

criterion = ContrastiveLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0005)


train(net, train_loader, optimizer, criterion)
eval(net, test_loader)
# show_plot(counter, loss_history)
