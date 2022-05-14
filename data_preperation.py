import pandas as pd
from torch.utils.data import Dataset
import torch
import numpy as np
from tqdm import tqdm
from data_analysis import data_collector
from torch.nn.utils.rnn import pad_sequence


class PatientDataset(Dataset):
    def __init__(self, data, labels):
        self.labels = labels
        self.num_of_hours = data.shape[1]  # TODO data.shape(1)?
        self.data = data

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        # assuming tensor (batch_size, hours, features), we want to get elements by each patient.
        indices = list(range(idx * self.num_of_hours, idx * self.num_of_hours + self.num_of_hours))
        patient = self.data[torch.arange(self.data.size(0)), indices]
        sample = {"Data": patient, "Class": label}
        return sample


def dataframe_to_tensor(df: pd.DataFrame, labels):

    patients = []
    for idx in tqdm(df['id']):
        patient = df[df['id'] == idx]
        patient_tensor = torch.tensor(patient.drop(columns='id').values.astype(np.float32))
        patients.append(patient_tensor)
    # since different patients stayed for varying times in ICU, we padded the data to get equal shapes for everyone.
    data_tensor = pad_sequence(patients, batch_first=True)

    torch_labels = torch.Tensor(labels)

    return data_tensor, torch_labels


def main():
    df = data_collector(1000, phase='train')
    dataframe_to_tensor(df, [])


if __name__ == '__main__':
    main()



