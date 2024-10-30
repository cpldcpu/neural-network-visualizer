import torch
import json
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Configuration

classes     = [0,1,2,3,4,5,6,7,8,9]     # selection of classes to use for the 4 output neurons
QuantType   = 'None'                  # Quantization type: 'None' or 'Binary'
filename    = 'weights_full_10c_noaug.json'     # file to save the weights
Name        = 'Detects All Number\nFull Precision, No Augmentation\n'            # used in the description of the weights file

class BitLinear(nn.Linear):
    """
    Linear fully connected layer with quantization aware training.
    """
    def __init__(self, in_features, out_features, QuantType='Binary'):
        nn.Linear.__init__(self, in_features, out_features, bias=False)
        self.QuantType = QuantType

    def forward(self, x):
        w = self.weight 

        if self.QuantType == 'None':
            y = F.linear(x, w)
        else:
            # Straight-Through-Estimator (STE)             
            w_int = w.sign()
            w_quant = w + (w_int - w).detach()
            y = F.linear(x, w_quant)
        return y

# Define the model
model = torch.nn.Sequential(
    nn.LayerNorm(64, elementwise_affine=False),
    BitLinear(64, 10, QuantType=QuantType),
    # nn.LayerNorm(10, elementwise_affine=False),
    nn.ReLU(),

    nn.LayerNorm(10, elementwise_affine=False),
    BitLinear(10, 10, QuantType=QuantType),
    # nn.LayerNorm(10, elementwise_affine=False),
    nn.ReLU(),

    nn.LayerNorm(10, elementwise_affine=False),
    BitLinear(10, len(classes), QuantType=QuantType)
)

transform = transforms.Compose([
    # transforms.RandomAffine(degrees=0, translate=(0.125, 0.125)), # data augmentation to capture also off-center inputs
    transforms.Resize((8, 8)),  # Rescale to 8x8 pixels
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Filter datasets
def filter_classes(dataset, classes):
    idx = [i for i, label in enumerate(dataset.targets) if label in classes]
    dataset.data = dataset.data[idx]
    dataset.targets = dataset.targets[idx]
    return dataset

train_dataset = filter_classes(train_dataset, classes)
test_dataset = filter_classes(test_dataset, classes)

label_map = {cls: i for i, cls in enumerate(classes)}
train_dataset.targets = torch.tensor([label_map[label.item()] for label in train_dataset.targets])
test_dataset.targets = torch.tensor([label_map[label.item()] for label in test_dataset.targets])

# DataLoader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Plot examples of the filtered classes
def plot_examples(dataset, classes, num_examples=5):
    fig, axes = plt.subplots(len(classes), num_examples, figsize=(num_examples * 2, len(classes) * 2))
    for i in range(len(classes)):
        cls_indices = [idx for idx, label in enumerate(dataset.targets) if label == i]
        for j in range(num_examples):
            idx = cls_indices[j]
            img = dataset.data[idx].numpy()
            axes[i, j].imshow(img, cmap='gray')
            axes[i, j].axis('off')
            if j == 0:
                axes[i, j].set_title(f'Class {classes[i]}')
    plt.show()

# plot_examples(train_dataset, classes)

criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = optim.AdamW(model.parameters(), lr=0.005)

# Training loop
def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.view(data.size(0), -1).to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

def test(model, device, test_loader, criterion, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.view(data.size(0), -1).to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'[Epoch {epoch}] Average test loss: {test_loss:.8f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    return accuracy

# Train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

accuracy = 0    
for epoch in range(11):
    train(model, device, train_loader, optimizer, criterion, epoch)
    accuracy = test(model, device, test_loader, criterion, epoch)

print(model)
# Save the trained model weights

def quantize(x):
    if QuantType == 'None':
        return round(x, 2)
    elif QuantType == 'Binary':
        return 1 if x > 0 else -1

weights = {
    "description" : f"{Name} Accuracy: {accuracy:.2f}%",
    "classes": classes,
    "weights": {
        "hidden1": [[quantize(value) for value in sublist] for sublist in model[1].weight.data.tolist()],
        "hidden2": [[quantize(value) for value in sublist] for sublist in model[4].weight.data.tolist()],
        "output":  [[quantize(value) for value in sublist] for sublist in model[7].weight.data.tolist()]
    }
}

with open(filename, 'w') as f:
    json.dump(weights, f)
    