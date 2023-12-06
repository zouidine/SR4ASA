from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel
from preprocessor import Preprocessor
from model import CNN_RNN
from trainer import train, evaluate
import torch
import pandas as pd

#Loading data
!git clone https://github.com/elnagara/HARD-Arabic-Dataset
!unzip 'HARD-Arabic-Dataset/data/balanced-reviews.zip'

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

#load BERT model from huggingface
tokenizer = AutoTokenizer.from_pretrained("UBC-NLP/MARBERT")
marbert = AutoModel.from_pretrained("UBC-NLP/MARBERT", output_hidden_states=True)

#load HARD dataset
df_HARD = pd.read_csv("/balanced-reviews.txt", sep="\t", header=0,
                      encoding='utf-16')
df_HARD = df_HARD[["review","rating"]]

# code rating as +ve if > 3, -ve if less, no 3s in dataset
hard_map = {
    5: 1,
    4: 1,
    2: 0,
    1: 0
}
df_HARD["rating"] = df_HARD["rating"].apply(lambda x: hard_map[x])

d_train, d_test, y_train, y_test = train_test_split(df_HARD["review"],
                                                    df_HARD["rating"],
                                                    test_size=0.2,
                                                    random_state=SEED,
                                                    stratify=df_HARD["rating"])
d_train, d_valid, y_train, y_valid = train_test_split(d_train,
                                                      y_train,
                                                      test_size=0.2,
                                                      random_state=SEED,
                                                      stratify=y_train)

preprocess = Preprocessor(tokenizer)

#TRAIN
d_train = preprocess.clean_and_tokenize(list(d_train))
train_src, train_mask = preprocess.creat_tensor(d_train)
train_trg = torch.tensor(list(y_train), dtype=torch.float)

#VALID
d_valid = preprocess.clean_and_tokenize(list(d_valid))
valid_src, valid_mask = preprocess.creat_tensor(d_valid)
valid_trg = torch.tensor(list(y_valid), dtype=torch.float)

#TEST
d_test = preprocess.clean_and_tokenize(list(d_test))
test_src, test_mask = preprocess.creat_tensor(d_test)
test_trg = torch.tensor(list(y_test), dtype=torch.float)

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
    train(model, train_loader, optimizer, criterion, epoch, zeta=zeta)
    valid_acc = evaluate(model, valid_loader, epoch)
    if valid_acc > best_valid_acc:
        best_valid_acc = valid_acc
        torch.save(model.state_dict(), 'models/hard_best_model.pt')
        print("\nbest accurcy is updated to ",100*best_valid_acc,"at", epoch,"\n")

#Evaluate
model.load_state_dict(torch.load('models/hard_best_model.pt'))

acc = evaluate(model, test_loader)
print(f'\nAcc: {acc*100:.2f}%')
