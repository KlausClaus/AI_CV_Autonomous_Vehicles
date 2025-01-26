import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import torch.optim as optim
from torch import nn
import numpy as np
from tqdm import tqdm

print("Starting the script...")


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return x


class U_net_segmentation(nn.Module):
    def __init__(self, num_classes):
        super(U_net_segmentation, self).__init__()
        self.encoder1 = ConvBlock(3, 64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.encoder2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.encoder3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.encoder4 = ConvBlock(256, 512)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.middle = ConvBlock(512, 1024)

        self.decoder4 = ConvBlock(1024, 512)
        self.decoder3 = ConvBlock(512, 256)
        self.decoder2 = ConvBlock(256, 128)
        self.decoder1 = ConvBlock(128, 64)

        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        p1 = self.pool1(e1)
        e2 = self.encoder2(p1)
        p2 = self.pool2(e2)
        e3 = self.encoder3(p2)
        p3 = self.pool3(e3)
        e4 = self.encoder4(p3)
        p4 = self.pool4(e4)

        # Middle
        m = self.middle(p4)

        # Decoder
        d4 = self.decoder4(nn.functional.interpolate(m, size=e4.size()[2:], mode='bilinear', align_corners=True))
        d3 = self.decoder3(nn.functional.interpolate(d4 + e4, size=e3.size()[2:], mode='bilinear', align_corners=True))
        d2 = self.decoder2(nn.functional.interpolate(d3 + e3, size=e2.size()[2:], mode='bilinear', align_corners=True))
        d1 = self.decoder1(nn.functional.interpolate(d2 + e2, size=e1.size()[2:], mode='bilinear', align_corners=True))

        # Classifier
        output = self.classifier(d1)
        return output


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
    # Resize both image and label to 512x512
    img = transforms.Resize((512, 512), interpolation=Image.BILINEAR)(img)
    label = transforms.Resize((512, 512), interpolation=Image.NEAREST)(label)

    img = transforms.ToTensor()(img)
    img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
    label = torch.from_numpy(np.array(label)).long() - 1
    label[label == -1] = 255
    return img, label


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
    model = U_net_segmentation(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    total_ious = {class_name: [] for class_name in class_names}
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            ious = calculate_iou(predicted, labels, num_classes, class_names)

            for class_name, iou in ious.items():
                if not np.isnan(iou):
                    total_ious[class_name].append(iou)

    mean_ious = {class_name: np.mean(iou_list) if iou_list else float('nan')
                 for class_name, iou_list in total_ious.items()}
    valid_ious = [iou for iou in mean_ious.values() if not np.isnan(iou)]
    mean_iou = np.mean(valid_ious) if valid_ious else float('nan')

    return mean_ious, mean_iou


print("Initializing dataset...")
train_dataset = CustomSegmentationDataset('autodl-tmp/V-01/image', 'autodl-tmp/V-01/indexLabel', transform=transform_pair)
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)

print("Checking unique labels...")
all_labels = []
for i, (_, label) in enumerate(tqdm(train_dataset, desc="Processing images")):
    all_labels.append(label.unique())
unique_labels = torch.cat(all_labels).unique()

num_classes = 18
print(f"Number of unique classes: {num_classes}")
print(f"Unique class labels: {unique_labels}")

print("Creating data loader...")


##########################################################
##########################################################
# This model has been saved in google drive
# https://drive.google.com/file/d/1UYHN6j9Gs7N98is2Cbok3N5seUpUAonq/view?usp=sharing
##########################################################
##########################################################

model_path = 'best_model_weights.pth'


def get_model(num_classes):
    model = U_net_segmentation(num_classes=num_classes)
    return model


def load_weights(model, model_path):
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print("Loaded weights from:", model_path)
    else:
        print("No weights found, starting training from scratch.")
    return model


print(f"Initializing model with {num_classes} classes...")
model = get_model(num_classes=num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)

model = load_weights(model, model_path)

test_dataset = CustomSegmentationDataset('autodl-tmp/V-01/image', 'autodl-tmp/V-01/indexLabel', transform=transform_pair)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss(ignore_index=255)

num_epochs = 10

best_miou = 0.0
best_ious = None
best_epoch = 0
first_time = True

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

    if(epoch==0):
        torch.save(model.state_dict(), 'best_model_weights.pth')

    # Test the model
    ious, current_iou = test_model_with_saved_weights(model_path, test_loader, num_classes, class_names)
    print(f"Epoch {epoch + 1}, Test mIoU: {current_iou}")

    # Print IoU for each class
    for class_name, iou in ious.items():
        if np.isnan(iou):
            print(f"IOU for the class: {class_name} in this epoch is: N/A (no instances)")
        else:
            print(f"IOU for the class: {class_name} in this epoch is: {iou:.4f}")

    # Save the new model
    torch.save(model.state_dict(), 'best_model_weights.pth')

    # Update best model if performance improved
    if current_iou > best_miou or first_time:
        best_miou = current_iou
        best_ious = ious
        best_epoch = epoch + 1
        print(f"Best model saved with mIoU {best_miou} at epoch {best_epoch}")
        first_time = False

print()
print("Testing model with saved weights...")
print()
print()

print(f"Best model was at epoch {best_epoch} with mIoU {best_miou}")

# Print IoU for each class in the best model
for class_name, iou in best_ious.items():
    if np.isnan(iou):
        print(f"IoU for class: {class_name} in the best mIoU round: N/A (no instances)")
    else:
        print(f"IoU for class: {class_name} in the best mIoU round: {iou:.4f}")

print("Script completed.")