import pandas as pd
import torch
import torchvision as tv
import torchvision.models as models
import torchvision.transforms as transforms

# Load a pre-trained ResNet50 model
model = models.resnet50(weights=tv.models.ResNet50_Weights.IMAGENET1K_V1)

# Remove the classification layer to get the features
model = torch.nn.Sequential(*(list(model.children())[:-1]))
model.eval()

# Preprocess the image
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

n = 1000

dataset = tv.datasets.CIFAR10(root='ecifar10', transform=transform, download=True)

# for each image in the dataset, extract features
with torch.no_grad():
    features = []
    for i, (img, label) in enumerate(dataset):
        if len(features) >= n:
            break
        img = img.unsqueeze(0)  # Add batch dimension
        feature = model(img)
        feature = feature.view(feature.size(0), -1).squeeze().cpu().numpy()
        feature_dict = { 'image_number': i, 'label': label }
        print(feature_dict)
        feature_dict.update({f"f{j+1}": feature[j] for j in range(feature.shape[0])})
        features.append(feature_dict)
    df = pd.DataFrame.from_records(features)
    df.to_csv('cifar10_features.csv', float_format='%.6f', index=False)
    print(f"Extracted features for {len(features)} images and saved to cifar10_features.csv")
    