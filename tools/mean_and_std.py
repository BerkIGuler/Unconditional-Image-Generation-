from torchvision import transforms, datasets
import torch
from torch.utils.data import dataloader
from tqdm import tqdm


image_size = 256
preprocess = transforms.Compose([transforms.Resize((image_size, image_size)),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0, 0, 0], [1, 1, 1])])

train_dataset = datasets.ImageFolder("/auto/data2/bguler/DDAN/breast_cancer/train", preprocess)
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=8, num_workers=2)

psum = torch.tensor([0.0, 0.0, 0.0])
psum_sq = torch.tensor([0.0, 0.0, 0.0])

# loop through images
for inputs, labels in tqdm(train_dataloader):
    psum += inputs.sum(dim=[0, 2, 3])
    psum_sq += (inputs ** 2).sum(dim=[0, 2, 3])


count = len(train_dataset) * image_size* image_size

# mean and std
total_mean = psum / count
total_var  = (psum_sq / count) - (total_mean ** 2)
total_std  = torch.sqrt(total_var)

# output
print('mean: ' + str(total_mean))
print('std: ' + str(total_std))
