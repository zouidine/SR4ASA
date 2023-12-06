from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.distributions import Categorical
import torch.nn.functional as F
import torch.nn as nn
import torch


class CNN_RNN(nn.Module):
    def __init__(self, bert, hid_dim, n_filters, filter_sizes, dropout, device):
        super().__init__()

        self.hid_dim = hid_dim
        self.device = device
        self.nbr_tokens = 0

        self.embedding = bert
        emb_dim = bert.config.to_dict()['hidden_size']

        self.gru_cell = nn.GRUCell(input_size=emb_dim, hidden_size=hid_dim,
                                   bias=True)
        self.select_linear = nn.Linear(hid_dim, 2)

        self.rnn = nn.GRU(emb_dim, hid_dim, batch_first=True, num_layers=2,
                          bidirectional=False)
        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels=1,
                                              out_channels=n_filters,
                                              kernel_size=(fs, hid_dim))
                                    for fs in filter_sizes
                                    ])
        self.linear_out = nn.Linear(len(filter_sizes)*n_filters, 1)
        self.dropout = nn.Dropout(dropout)
        # Freeze the BERT model
        for param in self.embedding.parameters():
                param.requires_grad = False

    def select_policy(self, embedded, mask, training):
        #embedded = [batch size, sent len, emb dim]
        self.saved_log_probs = torch.zeros(mask.shape, dtype=torch.float)
        self.selected_tokens = torch.zeros(mask.shape)
        lens = torch.sum(mask, dim=1).tolist()
        self.batch_nbr_tokens = 0

        hid = torch.zeros(mask.shape[0], self.hid_dim, requires_grad=True)
        hid = self.gru_cell(embedded[:,0,:].detach(), hid.to(self.device))
        self.selected_tokens[:,0] = torch.ones(mask.shape[0])
        for i in range(1,max(lens)-1):
            hid = self.gru_cell(embedded[:,i,:].detach(), hid)
            probs = self.select_linear(hid)
            if (training):
                probs = F.softmax(probs, dim=1)
                categ = Categorical(probs)
                k = categ.sample()
                self.saved_log_probs[:,i]  = categ.log_prob(k)
            else: k = probs.argmax(dim=1)
            self.selected_tokens[:,i] = k

        new_embedded = torch.zeros(embedded.shape, dtype=embedded.dtype,
                                   device=self.device)
        new_mask = torch.zeros(mask.shape, dtype=mask.dtype)
        for i in range(new_embedded.shape[0]):
            k = 0
            new_mask[i,:lens[i]-1] = self.selected_tokens[i,:lens[i]-1]
            new_mask[i,lens[i]-1] = 1
            for j in range(lens[i]):
                if new_mask[i][j]==1:
                    new_embedded[i,k,:] = embedded[i,j,:].detach()
                    k += 1
        self.batch_nbr_tokens = torch.sum(new_mask, dim=1).to('cpu')
        return new_embedded, new_mask

    def forward(self, text, mask, training=True):
        #text = [batch size, sent len]
        with torch.no_grad():
            embedded = self.embedding(text, attention_mask=mask)[0]
        embedded, mask = self.select_policy(embedded, mask, training)
        embedded = pack_padded_sequence(embedded, torch.sum(mask, 1),
                                        batch_first=True, enforce_sorted=False)
        #embedded = [batch size, sent len, emb dim]
        output, (_, _) = self.rnn(embedded)
        output, _ = pad_packed_sequence(output, batch_first=True)
        #output = [batch_size, sent_len, hidden_dim]
        if output.shape[1]<7:
            catted = torch.zeros(output.shape[0],
                                 7-output.shape[1],
                                 output.shape[2],
                                 dtype=output.dtype,
                                 device=self.device)
            output = torch.cat((output, catted), dim=1)
        output = output.unsqueeze(1)
        #output = [batch size, 1, sent len, hidden_dim]
        conved = [F.relu(conv(output)).squeeze(3) for conv in self.convs]
        #conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        #pooled_n = [batch size, n_filters]
        cat = self.dropout(torch.cat(pooled, dim = 1))
        #cat = [batch size, n_filters * len(filter_sizes)]
        return self.linear_out(cat).squeeze(1)
