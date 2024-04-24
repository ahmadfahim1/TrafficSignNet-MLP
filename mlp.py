import os
from sklearn.model_selection import train_test_split
import cv2
import torch.nn as nn
from torchsummary import summary
import torch
from torch.utils.data import DataLoader, Dataset, random_split

file_dir = 'GTSRB_subset_new'
class_names = ["class1", "class2"]
dataset = []

for label, cls in enumerate(class_names):
    path = os.path.join(file_dir, cls)

    class_images = [
        (cv2.imread(os.path.join(path, img)) / 255).reshape(3, 64, 64)
        for img in os.listdir(path)
    ]
    dataset.extend([(img_array, label) for img_array in class_images])

train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

class CustomMLP(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.flatten = nn.Flatten(1, -1)
        #self.norm = nn.BatchNorm1d(12288)
        self.dense1 = nn.Linear(12288, 100)
        self.dense2 = nn.Linear(100, 100)
        self.dense3 = nn.Linear(100, 2)

    def forward(self, x):
        x = self.flatten(x)
        #x = self.norm(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x

model = CustomMLP()

print(model)

summary(model,input_size=(3, 64, 64))

epoch = 10
lose_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

def train(model, train_loader, optimizer, loss_fn, epochs):
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for images, labels in train_loader:
            input = images.float()
            outputs = model(input)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss}")
    

train = train(model, train_loader, optimizer, lose_function, epoch)

def eval(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            image = images.float()
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(f'Test Accuracy: {accuracy * 100}%')
print()
eval(model, test_loader)