from torch import optim
import torch

def train(model, train_dataloader, model_name, criterion, device, max_epochs):
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model = model.to(device)
    model.train(True)
    print(model)

    for epoch in range(0, max_epochs):
        for i, data in enumerate(train_dataloader, 0):
            img0, img1, label = data
            img0, img1, label = img0.to(device), img1.to(device), label.to(device)
            optimizer.zero_grad()
            # output0, output1 = model(img0, img1)
            output0 = model(img0)
            output1 = model(img1)
            loss_contrastive = criterion(output0, output1, label)
            loss_contrastive.backward()
            optimizer.step()
            if i % 10 == 0:
                print("Epoch number {}\n Current loss {}\n".format(epoch, loss_contrastive.item()))

    torch.save(model, model_name)


def trainV2(model, train_dataloader, model_name, criterion, device, max_epochs):
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model = model.to(device)
    model.train(True)
    print(model)

    for epoch in range(0, max_epochs):
        for i, data in enumerate(train_dataloader, 0):
            img0, img1, label = data
            img0, img1, label = img0.to(device), img1.to(device), label.to(device)
            optimizer.zero_grad()
            # output0, output1 = model(img0, img1)
            output0 = model(img0)
            output1 = model(img1)
            loss_contrastive = criterion(output0, output1, label)
            loss_contrastive.backward()
            optimizer.step()
            if i % 10 == 0:
                print("Epoch number {}\n Current loss {}\n".format(epoch, loss_contrastive.item()))
        if epoch == 0:
            torch.save(model, model_name + '_epoch1')
        if epoch == 3:
            torch.save(model, model_name + '_epoch3')
        if epoch == 10:       
            torch.save(model, model_name + '_epoch10')
        if epoch == 20:
            torch.save(model, model_name + '_epoch20')

    torch.save(model, model_name)