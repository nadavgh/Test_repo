import pandas as pd
# import matplotlib.pyplot as plt
import numpy as np
# import seaborn as sns
import os

import torch
from tqdm import tqdm
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import log_softmax
from data_preperation import PatientDataset, dataframe_to_tensor

EPOCHS = 10
LOSS = nn.BCEWithLogitsLoss()


class LSTMPredictor(nn.Module):

    def __init__(self, input_dim, hidden_dim, dropout=0.1, batch_first=True):
        super(LSTMPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.lstm = nn.LSTM(input_dim, hidden_dim, dropout=self.dropout, batch_first=batch_first)

    def forward(self, x):
        lstm_out, _ = self.lstm(x.view(len(x), 1, -1))
        out_linear = self.hidden2tag(lstm_out.view(len(x), -1))
        # scores = log_softmax(out_linear, dim=1)
        return out_linear


def training_phase(train: torch.Tensor, labels):
    # TODO we have to scale the data somewhere
    X = PatientDataset(train, labels)
    dataloader = DataLoader(X, batch_size=16, shuffle=False)

    model = LSTMPredictor()
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    for epoch in range(EPOCHS):
        for (batch_idx, batch) in enumerate(dataloader):
            model.zero_grad()

            outputs = model(batch('Data'))

            loss = LOSS(outputs, batch('Class'))
            loss.backword()
            optimizer.step()



def main():
    pass

if __name__=='__main__':
    main()