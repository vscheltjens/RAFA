import torch

def get_train_test_functions(dataset: str):
    if dataset == 'mnist':

        def train(net, trainloader, optimizer, epochs):
            criterion = torch.nn.CrossEntropyLoss()
            net.train()
            for _ in range(epochs):
                for batch in trainloader:
                    images, labels = batch["img"], batch["label"]
                    optimizer.zero_grad()
                    outputs = net(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
            return net

        def test(net, testloader):
            criterion = torch.nn.CrossEntropyLoss()
            correct, loss = 0, 0.0
            net.eval()
            with torch.no_grad():
                for batch in testloader:
                    images, labels = batch["img"], batch["label"]
                    outputs = net(images)
                    loss += criterion(outputs, labels).item()
                    _, predicted = torch.max(outputs.data, 1)
                    correct += (predicted == labels).sum().item()
            accuracy = correct / len(testloader.dataset)
            return loss, accuracy

        return train, test

    elif dataset == 'cifar10':

        def train(net, trainloader, optimizer, epochs: int, verbose=False):
            criterion = torch.nn.CrossEntropyLoss()
            net.train()
            for epoch in range(epochs):
                correct, total, epoch_loss = 0, 0, 0.0
                for batch in trainloader:
                    images, labels = batch["img"], batch["label"]
                    optimizer.zero_grad()
                    outputs = net(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    total += labels.size(0)
                    correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
                epoch_loss /= len(trainloader.dataset)
                epoch_acc = correct / total
                if verbose:
                    print(f"Epoch {epoch+1}: train loss {epoch_loss:.4f}, accuracy {epoch_acc:.4f}")

        def test(net, testloader):
            criterion = torch.nn.CrossEntropyLoss()
            correct, total, loss = 0, 0, 0.0
            net.eval()
            with torch.no_grad():
                for batch in testloader:
                    images, labels = batch["img"], batch["label"]
                    outputs = net(images)
                    loss += criterion(outputs, labels).item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            loss /= len(testloader.dataset)
            accuracy = correct / total
            return loss, accuracy

        return train, test

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
