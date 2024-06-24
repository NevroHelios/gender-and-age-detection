import torch


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device):
    model.train()

    train_loss, train_acc = 0.0, 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.unsqueeze(1).to(device)
        
        y_preds = model(X)

        loss = loss_fn(y_preds, y.type(torch.float))
        train_loss += loss

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        # y_pred_class = torch.argmax(y_preds, dim=1)
        # print(y_pred_class.shape, y.shape)
        train_acc += ((y_preds.round() == y).sum().item() / len(y))

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)

    return train_loss.detach().cpu().numpy(), train_acc # tensor.detach() to remove the grad associated

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device):
    model.eval()

    test_loss, test_acc = 0, 0

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.unsqueeze(1).to(device)

            test_preds= model(X)
            
            loss = loss_fn(test_preds, y.type(torch.float))
            test_loss += loss

            # calculate the accuracy
            # test_pred_labels = test_preds.argmax(dim=1)
            test_acc += (test_preds.round() == y).sum().item() / len(y)

        test_loss /= len(dataloader)
        test_acc /= len(dataloader)

    return test_loss.cpu().numpy(), test_acc

from tqdm.auto import tqdm

def train(model:torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device):

    results = {"train loss": [],
               "train acc": [],
               "test loss": [],
               "test acc": []}


    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                        dataloader=train_dataloader,
                                        loss_fn=loss_fn,
                                        optimizer=optimizer,
                                        device=device)

        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn,
                                        device=device)

        print(f"\ntrain loss {train_loss:.4f} | train acc {train_acc:.4f} | test loss {test_loss:.4f} | test acc {test_acc:.4f}")
        print("*"*14)

        # train_losses =

        results['train loss'].append(train_loss)
        results['train acc'].append(train_acc)
        results['test loss'].append(test_loss)
        results['test acc'].append(test_acc)

    return results