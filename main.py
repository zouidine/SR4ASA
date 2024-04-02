from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel
from preprocessor import Preprocessor
from trainer import train, evaluate
from model import CNN_RNN
import torch
import LABR

SEED = 42
batch_size = 64
HIDDEN_DIM = 256
N_FILTERS = 100
FLTER_SIZES = [3,5,7]
DROPOUT = 0.5
lr = 1e-3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
zeta = 0.75
N_EPOCHS = 10

#load AraBERT large model
model_name = "aubmindlab/bert-large-arabertv2"
arabert = AutoModel.from_pretrained(model_name, output_hidden_states = True)
tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
arabert_prep = ArabertPreprocessor(model_name=model_name)

# main object that will read the train/test sets from files.
labr = LABR()

datas = [
        dict(name="2-balanced", params=dict(klass="2", balanced="balanced")),
        dict(name="2-unbalanced", params=dict(klass="2", balanced="unbalanced"))
        ]

# Load the data
print(60*"-")
data = datas[0]
print("Loading data:", data['name'])
(d_train, y_train, d_test, y_test) = labr.get_train_test(**data['params'])
d_train, d_valid, y_train, y_valid = train_test_split(d_train, y_train,
                                                      test_size=0.1,
                                                      random_state=SEED,
                                                      stratify=y_train)

preprocess = Preprocessor(tokenizer)

#TRAIN
d_train = preprocess.clean_and_tokenize(list(d_train))
train_src, train_mask = preprocess.creat_tensor(d_train)
train_trg = torch.tensor(y_train, dtype=torch.float)

#VALID
d_valid = preprocess.clean_and_tokenize(list(d_valid))
valid_src, valid_mask = preprocess.creat_tensor(d_valid)
valid_trg = torch.tensor(y_valid, dtype=torch.float)

#TEST
d_test = preprocess.clean_and_tokenize(list(d_test))
test_src, test_mask = preprocess.creat_tensor(d_test)
test_trg = torch.tensor(y_test, dtype=torch.float)

print("Train set size:\t", len(d_train))
print("\t Positif:", sum(y_train))
print("\t Negatif:", len(d_train)-sum(y_train))

print("Valid set size:\t", len(d_valid))
print("\t Positif:", sum(y_valid))
print("\t Negatif:", len(d_valid)-sum(y_valid))

print("Test set size:\t", len(d_test))
print("\t Positif:", sum(y_test))
print("\t Negatif:", len(y_test)-sum(y_test))

#LOADERS
dataset = torch.utils.data.TensorDataset(train_src, train_mask, train_trg)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                           num_workers=0)

dataset = torch.utils.data.TensorDataset(valid_src, valid_mask, valid_trg)
valid_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                           num_workers=0)

dataset = torch.utils.data.TensorDataset(test_src, test_mask, test_trg)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                          num_workers=0)

#Model
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
model = CNN_RNN(marbert, HIDDEN_DIM, N_FILTERS, FLTER_SIZES, DROPOUT, device)
model = model.to(device)

optimizer = torch.optim.Adam([param for param in model.parameters() if 
                              param.requires_grad == True], lr=lr)

criterion = torch.nn.BCEWithLogitsLoss()
criterion = criterion.to(device)

#Training
best_valid_acc = 0
for epoch in range(N_EPOCHS):
    train(model, train_loader, optimizer, criterion, epoch, device, zeta=zeta)
    valid_acc, _, _, _ = evaluate(model, valid_loader, device)
    if valid_acc > best_valid_acc:
        best_valid_acc = valid_acc
        torch.save(model.state_dict(), 'checkpoints/labr_best_model.pt')
        print("\nbest accurcy is updated to ",100*best_valid_acc,"at", epoch,"\n")

#Evaluate
model.load_state_dict(torch.load('checkpoints/labr_best_model.pt'))
acc, pre, rec, f1s = evaluate(model, test_loader, device)
print(f'\nAcc: {acc*100:.2f}%\nPre: {pre*100:.2f}%\nRec: {rec*100:.2f}%\nF1s: {f1s*100:.2f}%')
