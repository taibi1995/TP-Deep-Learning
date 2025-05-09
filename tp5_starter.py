import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt

torch.manual_seed(42)
MAIN = __name__ == "__main__"
BATCH_SIZE = 32
NUM_EPOCHS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
training_set = torchvision.datasets.FashionMNIST("./datasets", train=True, transform=transform, download=True)
validation_set = torchvision.datasets.FashionMNIST("./datasets", train=False, transform=transform, download=True)
classes = ("T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot")

training_loader = DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=True)
validation_loader = DataLoader(validation_set, batch_size = BATCH_SIZE, shuffle=False)

print(f"Training set size : {len(training_set)}")
print(f"Validation set size : {len(validation_set)} ")

images, labels = next(iter(validation_loader))
print(f"Shape of one image batch : {images.shape}")

sum = 0
for (images, labels) in validation_loader:
    sum += labels[0].item()
print(f"Sum of first label in every batch : {sum}")

class FashionClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6 ,out_channels=16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(in_features=16 * 4 * 4, out_features=120),
            nn.ReLU(),
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
            nn.Linear(in_features=84, out_features=10)
        )
    
    def forward(self,x):
        return self.net(x)
    
    def predict(self,x):
        with torch.no_grad():
            logits = self.net(x)
            pred = torch.argmax(logits, dim=1)
            return pred

model = FashionClassifier()

x = torch.randn(32,1,28,28)
out_put = model(x)
print(f"Shape of output : {out_put.shape}") # res doit etre [32, 10]

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

report_freq = 500
def train_one_epoch(model, loss_fn, optimizer):
    total_loss = 0
    correct = 0
    total_samples = 0
    history = []
    start_time = time.time()

    for i, data in enumerate(training_loader):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = loss_fn(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()
        predictions = torch.argmax(outputs, dim =1)
        correct += (predictions == labels).sum().item()
        total_samples += labels.size(0)

        if(i+1) % report_freq == 0 :
            avg_loss = total_loss / (i+1)
            accuracy = correct / total_samples
            elapsed_time = time.time() - start_time
            print(f"Batch {i+1}/{len(training_loader)} - "
                  f"Average loss : {avg_loss},"
                  f"Accuracy : {accuracy}" 
                  f"Elapsed time : {elapsed_time}"
                  )
            history.append({"batch": i+1, "loss": avg_loss, "accuracy": accuracy, "time": elapsed_time})

    return history

history = train_one_epoch(model, loss_fn, optimizer)
print("Historique d'entraînement :", history)

if MAIN:
    train_losses = []
    val_losses = []
    for epoch in range(NUM_EPOCHS):
        print(f"\n🚀 Époque {epoch+1}/{NUM_EPOCHS} 🚀")

        model.train()  # Mode entraînement
        
        # Entraînement du modèle
        train_history = train_one_epoch(model, loss_fn, optimizer)
        train_loss = train_history[-1]["loss"]
        train_losses.append(train_loss)

        model.eval()  # Mode évaluation
        
        # Évaluer sur validation
