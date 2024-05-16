from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
import os


def eval(net, test_loader):

    # test_loader_one = DataLoader(test_dataset, batch_size=1, shuffle=False)
    dataiter = iter(test_loader)
    x0, _, _ = next(dataiter)

    for i in range(5):
        # Iterate over 5 images and test them with the first image (x0)
        _, x1, label2 = next(dataiter)

        # Concatenate the two images together
        concatenated = torch.cat((x0, x1), 0)

        if os.getenv("CUDA", "DISABED") == "ENABLED":
            x0, x1 = x0.cuda(), x1.cuda()

        output1, output2 = net(x0, x1)
        euclidean_distance = F.pairwise_distance(output1, output2)
        imshow(
            torchvision.utils.make_grid(concatenated),
            f"Dissimilarity: {euclidean_distance.item():.2f}",
        )
