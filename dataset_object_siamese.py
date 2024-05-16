from torchvision import transforms
from PIL import Image
import torch
import os


class ImagePairDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir

        self.transform = transforms.Compose(
            [
                # We first resize the input image to 256x256 and then we take center crop.
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ]
        )
        self.image_pairs = self.load_image_pairs()

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        image1_path, image2_path, label = self.image_pairs[idx]
        image1 = Image.open(image1_path).convert("RGB")
        image2 = Image.open(image2_path).convert("RGB")

        # Convert the tensor to a PIL image
        # image1 = functional.to_pil_image(image1)
        # image2 = functional.to_pil_image(image2)

        image1 = self.transform(image1)
        image2 = self.transform(image2)
        # image1 = torch.clamp(image1, 0, 1)
        # image2 = torch.clamp(image2, 0, 1)
        return image1, image2, label

    def load_image_pairs(self):
        image_pairs = []
        # Assume the directory structure is as follows:
        # root_dir
        # ├── similar
        # │   ├── similar_image1.jpg
        # │   ├── similar_image2.jpg
        # │   └── ...
        # └── dissimilar
        #     ├── dissimilar_image1.jpg
        #     ├── dissimilar_image2.jpg
        #     └── ...
        similar_dir = os.path.join(self.root_dir, "similar_all_images")
        dissimilar_dir = os.path.join(self.root_dir, "dissimilar_all_images")

        # Load similar image pairs with label 1
        similar_images = os.listdir(similar_dir)
        for i in range(len(similar_images) // 2):
            image1_path = os.path.join(similar_dir, f"similar_{i}_first.jpg")
            image2_path = os.path.join(similar_dir, f"similar_{i}_second.jpg")
            image_pairs.append((image1_path, image2_path, 0))

        # Load dissimilar image pairs with label 0
        dissimilar_images = os.listdir(dissimilar_dir)
        for i in range(len(dissimilar_images) // 2):
            image1_path = os.path.join(dissimilar_dir, f"dissimilar_{i}_first.jpg")
            image2_path = os.path.join(dissimilar_dir, f"dissimilar_{i}_second.jpg")
            image_pairs.append((image1_path, image2_path, 1))

        return image_pairs
