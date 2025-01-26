import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import torch.optim as optim
from torch import nn
import numpy as np
from sam_model import SAMSegmentation
from tqdm import tqdm

print("Starting the script...")


class CustomSegmentationDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.images = [img for img in os.listdir(image_dir) if img.endswith('.png')]
        print(f"Found {len(self.images)} images in the dataset.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        label_path = os.path.join(self.label_dir, self.images[index])
        image = Image.open(img_path).convert("RGB")
        label = Image.open(label_path)

        if self.transform is not None:
            image, label = self.transform(image, label)

        return image, label


def transform_pair(img, label):
    img = transforms.ToTensor()(img)
    img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
    label = torch.from_numpy(np.array(label)).long() - 1  # 将标签值减1
    label[label == -1] = 255  # 将原来的0值（现在是-1）映射到255
    return img, label


print("Initializing dataset...")
train_dataset = CustomSegmentationDataset('autodl-tmp/V-01/image', 'autodl-tmp/V-01/indexLabel',transform=transform_pair)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

print("Checking unique labels...")
all_labels = []
for i, (_, label) in enumerate(tqdm(train_dataset, desc="Processing images")):
    all_labels.append(label.unique())
unique_labels = torch.cat(all_labels).unique()

num_classes = 18  # 设置为18，因为原始标签从1到18

print(f"Number of unique classes: {num_classes}")
print(f"Unique class labels: {unique_labels}")

print("Creating data loader...")

model_path = 'best_model_weights.pth'


def get_model(num_classes):
    model = SAMSegmentation(num_classes=num_classes)
    return model


def load_weights(model, model_path):
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print("Loaded weights from:", model_path)
    else:
        print("No weights found, starting training from scratch.")
    return model


def calculate_iou(predictions, labels, num_classes, class_names):
    ious = {class_name: float('nan') for class_name in class_names}
    for cls in range(num_classes):
        pred = (predictions == cls).float()
        label = (labels == cls).float()
        intersection = (pred * label).sum().item()
        union = pred.sum().item() + label.sum().item() - intersection
        if union != 0:
            ious[class_names[cls]] = intersection / union
    return ious


def test_model_with_saved_weights(model_path, dataloader, num_classes, class_names):
    model = SAMSegmentation(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    total_ious = {class_name: 0.0 for class_name in class_names}
    num_samples = 0
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            ious = calculate_iou(predicted, labels, num_classes, class_names)

            for class_name, iou in ious.items():
                if not torch.isnan(torch.tensor(iou)):
                    total_ious[class_name] += iou

            num_samples += 1

    mean_ious = {class_name: total_iou / num_samples for class_name, total_iou in total_ious.items()}
    mean_iou = torch.nanmean(torch.tensor(list(mean_ious.values()))).item()
    return mean_ious, mean_iou


print(f"Initializing model with {num_classes} classes...")
model = get_model(num_classes=num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)

model = load_weights(model, model_path)

test_dataset = CustomSegmentationDataset('autodl-tmp/V-01/image', 'autodl-tmp/V-01/indexLabel',
                                         transform=transform_pair)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss(ignore_index=255)

num_epochs = 10

best_miou = 0.0
best_ious = None
best_epoch = 0
first_time = False
current_iou = 0.0
ious = None

print("Starting training...")

class_names = ["asphalt", "dirt", "mud", "water", "gravel", "other-terrain", "tree-trunk", "tree-foliage",
               "bush", "fence", "structure", "pole", "vehicle", "rock", "log", "other-object", "sky", "grass"]

# Add Model Update Verification
initial_params = [param.clone().detach() for param in model.parameters()]

# Add Data Loader Verification
print("Checking data loader...")
for check_epoch in range(2):  # Check for two epochs
    print(f"Check Epoch {check_epoch + 1}")
    for i, (images, labels) in enumerate(train_loader):
        if i == 0:  # Check only the first batch
            print(f"Batch shape: {images.shape}")
            print(f"Labels shape: {labels.shape}")
            print(f"Unique labels in batch: {torch.unique(labels)}")
        if i == len(train_loader) - 1:
            print(f"Total batches in epoch: {i + 1}")
    print("---")

for epoch in range(num_epochs):
    model = load_weights(model, model_path)

    model.train()
    running_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
    for i, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
    print(f"Epoch {epoch + 1}, Average Loss: {running_loss / len(train_loader)}")

    # Add Model Update Verification at the end of each epoch
    current_params = [param.clone().detach() for param in model.parameters()]
    param_change = sum(torch.sum(torch.abs(p1 - p2)) for p1, p2 in zip(initial_params, current_params))
    print(f"Total parameter change: {param_change.item()}")

    # first time in the loop
    if (epoch == 0):
        first_time = True
    else:
        # 测试当前 Epoch 的模型
        ious, current_iou = test_model_with_saved_weights(model_path, test_loader, num_classes, class_names)
        print(f"Epoch {epoch + 1}, Test mIoU: {current_iou}")

        # 打印每个类别的IoU
        for class_name, iou in ious.items():
            print(f"IOU for the class: {class_name} in this epoch is: {iou:.4f}")

    # 无论怎样都save这个新的model
    torch.save(model.state_dict(), 'best_model_weights.pth')
    # 如果当前模型的性能超过了之前的最佳性能，保存它
    if current_iou > best_miou or first_time:
        if (first_time):
            first_time = False
            best_epoch = epoch + 1
            # torch.save(model.state_dict(), 'best_model_weights.pth')
            ious, current_iou = test_model_with_saved_weights(model_path, test_loader, num_classes, class_names)
            best_miou = current_iou
            best_ious = ious
            print(f"Best model saved with mIoU {best_miou} at epoch {best_epoch}")
            # 打印每个类别的IoU
            for class_name, iou in ious.items():
                print(f"IOU for the class: {class_name} in this epoch is: {iou:.4f}")
        else:
            best_miou = current_iou
            best_ious = ious
            best_epoch = epoch + 1
            # torch.save(model.state_dict(), 'best_model_weights.pth')
            print(f"Best model saved with mIoU {best_miou} at epoch {best_epoch}")

print()
print("Testing model with saved weights...")
print()
print()

print(f"Best model was at epoch {best_epoch} with mIoU {best_miou}")

# 打印每个类别的IoU
for class_name, iou in best_ious.items():
    print(f"IoU for class: {class_name} in the best mIoU round: {iou:.4f}")

print("Script completed.")