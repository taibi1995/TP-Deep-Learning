#!/usr/bin/env python
# coding: utf-8

# # Introduction à l'apprentissage profond - TP 7 (DM 2)

# ## Setup

# In[1]:


get_ipython().system('pip install torchinfo')
import torch as t
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from torch.nn import Sequential, Conv2d, BatchNorm2d, ReLU, MaxPool2d, Linear
from PIL import Image
from torchinfo import summary


# In[2]:


class AveragePool(nn.Module):
    def forward(self, x):
        return t.mean(x, dim=(2, 3))


# In[3]:


device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")
print(device)


# ## ResNet architecture (Q1-2)

# In[4]:


class ResidualBlock(nn.Module):
    def __init__(self, in_feats: int, out_feats: int, first_stride=1):
        super().__init__()

        self.left = nn.Sequential(
            nn.Conv2d(in_feats, out_feats, kernel_size=3, stride=first_stride, padding=1, bias=False),
            nn.BatchNorm2d(out_feats),
            nn.ReLU(),
            nn.Conv2d(out_feats, out_feats, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_feats)
        )

        if first_stride == 1:
            self.right = nn.Identity()
        else:
            self.right = nn.Sequential(
                nn.Conv2d(in_feats, out_feats, kernel_size=1, stride=first_stride, padding=0, bias=False),
                nn.BatchNorm2d(out_feats)
            )

        self.relu = nn.ReLU()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        left_output = self.left(x)
        right_output = self.right(x)
        output = left_output + right_output
        return self.relu(output)


# In[5]:


class BlockGroup(nn.Module):
    def __init__(self, n_blocks: int, in_feats: int, out_feats: int, first_stride=1):
        """Un groupe de n_blocks ResidualBlock, où seul le premier bloc utilise le first_stride donné."""
        super().__init__()
        blocks = [ResidualBlock(in_feats, out_feats, first_stride)]

        for _ in range(1, n_blocks):
            blocks.append(ResidualBlock(out_feats, out_feats, first_stride=1))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


# In[6]:


class ResNet(nn.Module):
    def __init__(self, n_blocks_per_group: list[int], out_features_per_group: list[int], first_strides_per_group: list[int], n_classes: int):
        super().__init__()

        assert len(n_blocks_per_group) == len(out_features_per_group) == len(first_strides_per_group),             "Les listes n_blocks_per_group, out_features_per_group et first_strides_per_group doivent avoir la même longueur."

        self.in_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.residual_layers = nn.ModuleList()
        in_feats = 64
        for n_blocks, out_feats, first_stride in zip(n_blocks_per_group, out_features_per_group, first_strides_per_group):
            group = BlockGroup(n_blocks, in_feats, out_feats, first_stride)
            self.residual_layers.append(group)
            in_feats = out_feats

        self.out_layers = nn.Sequential(
            AveragePool(),
            nn.Flatten(),
            nn.Linear(in_feats, n_classes)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.in_layers(x)

        for group in self.residual_layers:
            x = group(x)
        x = self.out_layers(x)
        return x


# # (a) Pourquoi les shapes doivent être identiques ?
# # L'addition des deux branches nécessite que leurs shapes soient identiques. Sinon, l'opération d'addition ne serait pas possible.

# # (b) Pourquoi désactiver le biais dans les convolutions ?
# 
# #  car la couche BatchNorm qui suit ajoute déjà un biais. Cela évite la redondance et aide le modèle à mieux apprendre.

# ## Testing (Q3-5)

# # test residual block :

# In[7]:


def test_residual_block():
    x = t.randn(1, 64, 32, 32)

    block = ResidualBlock(in_feats=64, out_feats=64, first_stride=1)
    output = block(x)
    print("Shape sans downsampling:", output.shape)  # Doit être (1, 64, 32, 32)
    assert output.shape == (1, 64, 32, 32), "Erreur de shape sans downsampling"

    block = ResidualBlock(in_feats=64, out_feats=128, first_stride=2)
    output = block(x)
    print("Shape avec downsampling:", output.shape)  # Doit être (1, 128, 16, 16)
    assert output.shape == (1, 128, 16, 16), "Erreur de shape avec downsampling"

test_residual_block()


# In[8]:


def test_block_group():
    x = t.randn(1, 64, 32, 32)

    group = BlockGroup(n_blocks=2, in_feats=64, out_feats=128, first_stride=2)
    output = group(x)
    print("Shape de BlockGroup:", output.shape)
    assert output.shape == (1, 128, 16, 16), "Erreur de shape de BlockGroup"

test_block_group()


# In[9]:


def test_resnet():
    x = t.randn(1, 3, 224, 224)

    n_blocks_per_group = [3, 4, 6, 3]
    out_features_per_group = [64, 128, 256, 512]
    first_strides_per_group = [1, 2, 2, 2]
    n_classes = 1000
    model = ResNet(n_blocks_per_group, out_features_per_group, first_strides_per_group, n_classes)
    output = model(x)
    print("Shape de ResNet:", output.shape)
    assert output.shape == (1, 1000), "Erreur de shape de ResNet"

test_resnet()


# In[10]:


def initialize_my_resnet34():
    return ResNet(n_blocks_per_group=[3, 4, 6, 3], out_features_per_group=[64, 128, 256, 512], first_strides_per_group=[1, 2, 2, 2], n_classes=1000)


# In[11]:


my_resnet = initialize_my_resnet34()
target_resnet = models.resnet34()

# Afficher un résumé des modèles
print("My model:")
print(summary(my_resnet, input_size=(1, 3, 64, 64), depth=4))  # Utiliser summary directement

print("Reference model:")
print(summary(target_resnet, input_size=(1, 3, 64, 64), depth=2))


# ## Inference (Q6-7)

# In[12]:


def copy_weights(my_resnet, pretrained_resnet):
    mydict = my_resnet.state_dict()
    pretraineddict = pretrained_resnet.state_dict()
    assert len(mydict) == len(pretraineddict), "Mismatching state dictionaries."
    state_dict_to_load = {mykey: pretrainedvalue for (mykey, _), (_, pretrainedvalue) in zip(mydict.items(), pretraineddict.items())}
    my_resnet.load_state_dict(state_dict_to_load)
    return my_resnet


pretrained_resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1).to(device)
my_resnet = copy_weights(my_resnet, pretrained_resnet).to(device)


# In[13]:


IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMAGENET_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


# In[14]:


import torch

def predict(model, image_tensor):
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        _, predicted_class = torch.max(probabilities, dim=0)
    return predicted_class.item()

for i in range(3):
    image_tensor = torch.rand(1, 3, 224, 224).to(device)

    pred_torch = predict(pretrained_resnet, image_tensor)
    pred_my = predict(my_resnet, image_tensor)

    print(f"Image aléatoire {i + 1}")
    print(f"Prédiction de torch_resnet: {pred_torch}")
    print(f"Prédiction de my_resnet: {pred_my}")
    print("=" * 50)


# ## Finetuning (Q8-10)

# In[15]:


def get_resnet_for_finetuning(n_classes: int):
    my_resnet = initialize_my_resnet34()

    pretrained_resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1).to(device)

    my_resnet = copy_weights(my_resnet, pretrained_resnet).to(device)

    for param in my_resnet.parameters():
        param.requires_grad = False

    in_features = my_resnet.out_layers[-1].in_features
    my_resnet.out_layers[-1] = nn.Linear(in_features, n_classes)

    return my_resnet


# In[16]:


def get_cifar() -> tuple[datasets.CIFAR10, datasets.CIFAR10]:
    cifar_trainset = datasets.CIFAR10("datasets", train=True, download=True, transform=IMAGENET_TRANSFORM)
    cifar_testset = datasets.CIFAR10("datasets", train=False, download=True, transform=IMAGENET_TRANSFORM)
    return cifar_trainset, cifar_testset

def get_cifar_subset(trainset_size: int = 10_000, testset_size: int = 1_000) -> tuple[Subset, Subset]:
    cifar_trainset, cifar_testset = get_cifar()
    return Subset(cifar_trainset, range(trainset_size)), Subset(cifar_testset, range(testset_size))


# In[17]:


def finetune_to_cifar(model, batch_size=64, epochs=1, learning_rate=1e-3):
    trainset, testset = get_cifar()
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    params_to_optimize = model.out_layers[-1].parameters()
    optimizer = t.optim.Adam(params_to_optimize, lr=learning_rate)

    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / len(trainloader)
        train_acc = 100. * correct / total
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")

        # Validation loop
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        test_loss = running_loss / len(testloader)
        test_acc = 100. * correct / total
        print(f"Epoch {epoch + 1}/{epochs}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")

    return model


# In[18]:


model = get_resnet_for_finetuning(n_classes=10).to(device)
model = finetune_to_cifar(model)


# # Pourquoi la précision est-elle nettement meilleure dans le premier cas ?
# #  Sans copie du poids de modele jai eu 15 %. car le modele  est initialisé avec des poids aleatoire ce qui entraine une lente convergence, pour cela on a besoin de plus de donnees(cifar 50000 !) et plus d epochs pour avoir une bonne precision alors que le modele avec poids est pre-entrainé avec un jeu de donnees plus large .

# In[18]:




