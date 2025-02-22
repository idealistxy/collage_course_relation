import torch
import torchvision
import torchvision.transforms as transforms

# 设置PyTorch随机数种子
torch.manual_seed(42)

# 定义转化函数将数据转化为张量并正则化
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# 加载 Fashion-MNIST 数据集
trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True)

testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False)

# Step 0: 初始化权重和偏置向量
input_size = 28 * 28  # 输入大小
output_size = 10  # 输出类别

# 使用正态分布初始化权重
weights = torch.randn(input_size, output_size) * 0.01  # 乘以0.01来限制权重的初始范围
biases = torch.zeros(output_size)  # 偏置向量置零


# 步骤 0：定义softmax函数
def softmax(x):
    e_x = torch.exp(x - torch.max(x))
    return e_x / e_x.sum(dim=1, keepdim=True)


# 步骤 1: 对于每个类别和输入特征计算得分
def calculate_scores(inputs, weights, biases):
    return torch.matmul(inputs, weights) + biases


# 步骤 3: 计算交叉熵损失
def cross_entropy_loss(scores, targets):
    # 将结果转化为独热
    targets_onehot = torch.zeros_like(scores)
    targets_onehot.scatter_(1, targets.view(-1, 1), 1)
    # 计算两者区别
    return -torch.sum(targets_onehot * torch.log_softmax(scores, dim=1))


# 步骤 4: 计算梯度
def calculate_gradients(inputs, scores, targets):
    # # 将结果转化为独热
    targets_onehot = torch.zeros_like(scores)
    targets_onehot.scatter_(1, targets.view(-1, 1), 1)
    gradients_weights = torch.matmul(inputs.t(), (torch.softmax(scores, dim=1) - targets_onehot))
    gradients_biases = torch.sum(torch.softmax(scores, dim=1) - targets_onehot, dim=0)
    return gradients_weights, gradients_biases


# 步骤 5: 更新权重和偏置
def update_parameters(weights, biases, gradients_weights, gradients_biases, learning_rate):
    weights -= learning_rate * gradients_weights
    biases -= learning_rate * gradients_biases
    return weights, biases


# 训练模型
def train_model(trainloader, weights, biases, learning_rate, epochs=10):
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            inputs = inputs.view(inputs.size(0), -1)
            scores = calculate_scores(inputs, weights, biases)
            probs = softmax(scores)

            loss = cross_entropy_loss(scores, labels)
            running_loss += loss.item()

            gradients_weights, gradients_biases = calculate_gradients(inputs, scores, labels)

            weights, biases = update_parameters(weights, biases, gradients_weights, gradients_biases, learning_rate)

            _, predicted = torch.max(probs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(trainloader)
        epoch_accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss}, Accuracy: {epoch_accuracy}%")

    return weights, biases


# 验证模型
def evaluate_model(testloader, weights, biases):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs = inputs.view(inputs.size(0), -1)
            
            scores = calculate_scores(inputs, weights, biases)
            
            probs = softmax(scores)
            _, predicted = torch.max(probs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100*correct/total}%")


# 定义超参数
learning_rate = 0.002
epochs = 10

# 训练模型
trained_weights, trained_biases = train_model(trainloader, weights, biases, learning_rate, epochs)

# 验证模型
evaluate_model(testloader, trained_weights, trained_biases)

