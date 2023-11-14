from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, precision_score, confusion_matrix
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
dataset_path = "E:\\dataset\\2-MedImage-TrainSet\\2-MedImage-TrainSet"
classes = ["disease", "normal"]
data = []
labels = []
#对图片大小进行处理
for class_index, class_name in enumerate(classes):
    class_path = os.path.join(dataset_path, class_name)
    for filename in os.listdir(class_path):
        image_path = os.path.join(class_path, filename)
        image = cv2.imread(image_path)
        # 将图像缩放为统一大小（例如，100x100像素）
        image = cv2.resize(image, (224,224))#初始224
        data.append(image)
        labels.append(class_index)

data = np.array(data)
labels = np.array(labels)
data = data.transpose((0, 3, 1, 2))

dataset_path1 = "E:\\dataset\\2-MedImage-TestSet\\2-MedImage-TestSet"
classes = ["disease", "normal"]
data1 = []
labels1 = []
for class_index, class_name in enumerate(classes):
    class_path = os.path.join(dataset_path1, class_name)
    for filename in os.listdir(class_path):
        image_path = os.path.join(class_path, filename)
        image = cv2.imread(image_path)
        # 将图像缩放为统一大小（例如，100x100像素）
        image = cv2.resize(image, (224, 224))
        data1.append(image)
        labels1.append(class_index)

data1 = np.array(data1)
labels1 = np.array(labels1)
data1 = data1.transpose((0, 3, 1, 2))
#device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class ResNet18(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet18, self).__init__()
        # 使用预训练的ResNet-18模型
        self.resnet18 = models.resnet18(pretrained=True)

        # 修改最后的全连接层，使其适应二分类任务
        in_features = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.resnet18(x)

model = ResNet18().cuda()
model.to(device)
data = torch.tensor(data, dtype=torch.float32)
labels = torch.tensor(labels, dtype=torch.long)
data1 = torch.tensor(data1, dtype=torch.float32)
labels1 = torch.tensor(labels1, dtype=torch.long)

dataset = torch.utils.data.TensorDataset(data, labels)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
dataset1 = torch.utils.data.TensorDataset(data1, labels1)
dataloader1 = torch.utils.data.DataLoader(dataset1, batch_size=32, shuffle=True)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

num_epochs = 5
for epoch in range(num_epochs):
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
#weights_path = 'aasd.pth'
#model.load_state_dict(torch.load(weights_path))


model.eval()
correct = 0
total = 0
all_probabilities = []
labll=[]
#all_probabilities.(device), labll.(device),correct.(device),total.(device)
with torch.no_grad():
    for inputs, labels in dataloader1 :
        inputs, labels=inputs.to(device), labels.to(device)
        outputs = model(inputs)
        pprobabilities = torch.nn.functional.sigmoid(outputs)
        probabilities_np = pprobabilities.cpu().numpy()
        labe=labels.cpu().numpy()
        # 取出每个样本的第一个概率
        first_probabilities = probabilities_np[:, 0]
        _, predicted = torch.max(outputs.data, 1)
        all_probabilities.append(first_probabilities)
        labll.append(labe)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f"Accuracy: {accuracy * 100:.2f}%")
final_label = np.concatenate(labll)
final_label = 1 - final_label
final_probabilities = np.concatenate(all_probabilities)
threshold = 0.5
binary_predictions = (final_probabilities> threshold).astype(int)
conf_matrix = confusion_matrix(final_label, binary_predictions)
precision = precision_score(final_label, binary_predictions)
auc = roc_auc_score(final_label, final_probabilities)
print(f"Precision: {precision * 100:.2f}%",f"Auc: {auc * 100:.2f}%")
fpr, tpr, thresholds = roc_curve(final_label, final_probabilities)
plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()