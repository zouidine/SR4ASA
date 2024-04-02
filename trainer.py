from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from tqdm import tqdm
import torch

def get_scores(preds, y):
    acc = accuracy_score(y, preds)
    pre = precision_score(y, preds)
    rec = recall_score(y, preds)
    f1s = f1_score(y, preds)
    return acc, pre, rec, f1s

def train(model, iterator, optimizer, criterion, epoch_no):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    with tqdm(iterator) as it:
        for batch_no, batch in enumerate(it, start=1):
            optimizer.zero_grad()
            predictions = model(batch[0].to(device), batch[1].to(device))
            loss = criterion(predictions, batch[2].to(device))
            loss.backward(retain_graph=True)
            optimizer.step()
            acc, _, _, _ = get_scores(torch.round(torch.sigmoid(predictions)).tolist(),
                                      batch[2].tolist())
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

def evaluate(model, iterator):
    model.eval()
    pred = []
    targ = []
    with torch.no_grad():
        with tqdm(iterator) as it:
            for batch_no, batch in enumerate(it, start=1):
                predictions = model(batch[0].to(device), batch[1].to(device))
                pred += torch.round(torch.sigmoid(predictions)).tolist()
                targ += batch[2].tolist()
    return get_scores(pred, targ)
