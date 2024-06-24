import torch
import torch.utils
import torch.utils.data

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device):
    model.train()

    train_loss = 0.0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.unsqueeze(1).to(device)
        
        y_preds = model(X)

        loss = loss_fn(y_preds, y.type(torch.float))
        train_loss += loss

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


    train_loss /= len(dataloader)
    return train_loss.detach().cpu().numpy() # tensor.detach() to remove the grad associated

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device):
    model.eval()

    test_loss = 0.0

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.unsqueeze(1).to(device)

            test_preds = model(X)
            
            loss = loss_fn(test_preds, y.type(torch.float))
            test_loss += loss


        test_loss /= len(dataloader)
    return test_loss.cpu().numpy()

from tqdm.auto import tqdm

def train(model:torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs:int,
          device: torch.device):

    results = {"train loss": [],
               "test loss": []}


    for epoch in tqdm(range(epochs)):
        train_loss = train_step(model=model,
                                dataloader=train_dataloader,
                                loss_fn=loss_fn,
                                optimizer=optimizer,
                                device=device)

        test_loss = test_step(model=model,
                              dataloader=test_dataloader,
                              loss_fn=loss_fn,
                              device=device)

        print(f"\ntrain loss {train_loss:.4f} | test loss {test_loss:.4f} | ")
        print("*"*14)

        results['train loss'].append(train_loss)
        results['test loss'].append(test_loss)        

    return results