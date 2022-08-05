
import torch

def train(model, iterator, optimizer, criterion, device):

    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for i,(x, y,p) in enumerate(iterator):

        x = x.to(device)
        y = y.to(device)
        p = p.to(device)


        y = torch.unsqueeze(y, 1)
        p = torch.unsqueeze(p, 1)

        optimizer.zero_grad()

        y_pred, _ = model(x)

        loss = criterion(y_pred, y, p)
        #loss = criterion(y_pred, y)

        acc = calculate_accuracy(y_pred, y)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_accuracy(y_pred, y):
    top_pred = (y_pred>0.5).float()
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc


def evaluate(model, iterator, criterion, device):

    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():

        for i,(x, y,p) in enumerate(iterator):

            x = x.to(device)
            y = y.to(device)
            p = p.to(device)

            y = torch.unsqueeze(y, 1)
            p = torch.unsqueeze(p, 1)

            y_pred, _ = model(x)

            loss = criterion(y_pred, y)

            acc = calculate_accuracy(y_pred, y)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs