import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

#pytorch device 결정(GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

class simpleModel(nn.Module):
    def __init__(self):
        super(simpleModel, self).__init__()

        self.conv = nn.Conv2D(in_channels=3, out_channels=16, kernel_size=3, padding=1, strides=1, bias=False)

        self.batchnorm = nn.BatchNorm2D(16, affine=True)
        self.relu = nn.Activation('relu')
        self.avg_pool = AvgPool2D(8)
        self.flatten = Flatten()
        self.fc = nn.Linear(256, 10) #torch에는 softmax가 없고 대신 crossetnrotpyloss를 쓴다 

    def forward(self, x):
        batch = x.size(0) # x.shape[0]
        out0 = self.conv(x)
        out1 = self.batchnorm(out0)
        out2 = self.relu(out1)
        out3 = self.avg_pool(out2)
        out4 = out3.view(batch, -1)
        out5 = self.fc(out4)
        return out5

model = simpleModel().to(device)



batch_size = 32
epochs=3

loss_fn = nn.CrossEntropyLoss() #loss function

#model.parameters 필수 
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) #optimizer

# data가 들어왔을 때 변환
transform = transforms.Compose([
    transforms.ToTensor(), #0~1 사이로 변환 & data가 어떤 거든 tensor로 변환
    transforms.Normalize(mean=(0.5, 0.5, 0,5), std=(0.5, 0.5, 0.5)) #-1~1 사이로 변환

]) #output[channel] = (input[channel] - mean[channel]) / std[channel]


train_dataset = datasets.CIFAR10('train/',
                                    train=True, download=True,
                                    transform=transform)

train_loader = DataLoader(dataset=train_dataset,
                        batch_size=batch_size, shuffle=True)

loss_dict = {}
val_loss_dict = {}


for epoch in range(1, epochs +1):
    loss_list = []

    for img, label in train_loader:
        model.train() #model.eval() #test나 validation 할 때 적절히 변경해서 사용 
        img = img.to(device)
        label = label.to(device)

        output = model(img)
        loss = loss_fn(output, label)

        loss_list.append(loss.item()) # loss값이 loss.item()에 있다 

        # 이하 세 줄은 함께 다닌다 
        optimizer.zero_grad() # weight의 gradient를 다 0으로 만들겠다 (weight gradient == 0)
        loss.backward() # backpropagation (weight gradient update)
        optimizer.step() # gradient로 weight가 update
    
    loss_dict[epoch] = loss_list # epoch로 평균 내면 이게 loss가 된다 
    # earlyStopping 구현 가능 (파이토치엔 es 없음)

    val_loss_list = []

    # validaton 
    for val_img, val_label in valid_loader:

        model.eval()
        
        # 여기서는 가중치 계산 안 한다 -> 메모리 적어지고 속도가 빨라진다 (보통 세트로 사용함)
        with torch.no_grad():
            val_img = val_img.to(device)
            val_label = val_label.to(device)

            val_output = model(val_img)
            val_loss = loss_fn(val_output, val_label)




    print(f"[{epoch}/{epochs}] finished")
    print('===============')

torch.save(model.state_dict(), 'cifar10_model.pt')