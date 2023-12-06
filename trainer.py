from sklearn.metrics import accuracy_score
from tqdm import tqdm
import torch

def get_acc(preds, y):
    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    rounded_preds = rounded_preds.tolist()
    y = y.tolist()
    acc = accuracy_score(y, rounded_preds)
    return acc

def train(model, iterator, optimizer, criterion, epoch_no, zeta=0.25):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    model.nbr_tokens = 0
    with tqdm(iterator) as it:
        for batch_no, batch in enumerate(it, start=1):
            optimizer.zero_grad()
            predictions = model(batch[0].to(device), batch[1].to(device))
            bce_loss = criterion(predictions, batch[2].to(device))
            pg_loss = torch.tensor(0.0, device=device, requires_grad=True)
            for i in range(predictions.shape[0]):
                x1 = model.batch_nbr_tokens[i].item()
                x2 = torch.sum(batch[1][i]).item()
                nst = x1/x2
                model.nbr_tokens += x1
                if (torch.round(torch.sigmoid(predictions[i]))==batch[2][i]):
                    if nst < zeta:
                        R = 1
                    else:
                        R = 1 + (1-nst)
                else: R = -1
                log_probs = model.saved_log_probs[i,:x2-1].tolist()
                for log_prob in log_probs:
                    pg_loss = pg_loss + R*log_prob
            loss = bce_loss - pg_loss/batch[0].shape[0]
            loss.backward(retain_graph=True)
            optimizer.step()
            acc = get_acc(predictions.to('cpu'), batch[2])
            epoch_loss += loss.item()
            epoch_acc += acc
            it.set_postfix(
                        ordered_dict={
                            "avg_epoch_loss": epoch_loss / batch_no,
                            "avg_epoch_accu": 100*(epoch_acc / batch_no),
                            "epoch": epoch_no,
                        },
                        refresh=True,
                    )

def evaluate(model, iterator, epoch_no=None):
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
        with tqdm(iterator) as it:
            for batch_no, batch in enumerate(it, start=1):
                predictions = model(batch[0].to(device), batch[1].to(device),
                                    training=False)
                acc = get_acc(predictions.to('cpu'), batch[2])
                epoch_acc += acc
                if epoch_no is not None:
                    it.set_postfix(
                            ordered_dict={
                                "avg_epoch_accu": 100*(epoch_acc / batch_no),
                                "epoch": epoch_no,
                            },
                            refresh=True,
                    )
    return epoch_acc/len(iterator)
