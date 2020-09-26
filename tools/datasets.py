import torch
from torch.utils.data import Dataset


class OpenVaccineDataset(Dataset):
    def __init__(self, mode, df, logger=None, debug=False):
        self.mode = mode
        self.df = df.reset_index(drop=True)
        self.token2int = {x: i for i, x in enumerate('().ACGUBEHIMSX')}
        self.logger = logger
        self.debug = debug

        # initialize df
        self.df['encoded_sequence'] = None
        self.df['encoded_structure'] = None
        self.df['encoded_predicted_loop_type'] = None

        LABEL_COLS = [
            'reactivity_error',
            'deg_error_Mg_pH10',
            'deg_error_pH10',
            'deg_error_Mg_50C',
            'deg_error_50C',
            'reactivity',
            'deg_Mg_pH10',
            'deg_pH10',
            'deg_Mg_50C',
            'deg_50C',
        ]
        for label_col in LABEL_COLS:
            if label_col not in self.df.columns:
                self.df[label_col] = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]

        if row['encoded_sequence'] is None:
            row = self._prep(row)
            self.df.loc[idx, 'encoded_sequence'] = row['encoded_sequence']
            self.df.loc[idx, 'encoded_structure'] = row['encoded_structure']
            self.df.loc[idx, 'encoded_predicted_loop_type'] \
                = row['encoded_predicted_loop_type']

        return {
            'id': row['id'],
            'encoded_sequence': torch.tensor(row['encoded_sequence']),
            'encoded_structure': torch.tensor(row['encoded_structure']),
            'encoded_predicted_loop_type': torch.tensor(row['encoded_predicted_loop_type']),
            'reactivity': torch.tensor(row['reactivity']),
            'deg_Mg_pH10': torch.tensor(row['deg_Mg_pH10']),
            'deg_pH10': torch.tensor(row['deg_pH10']),
            'deg_Mg_50C': torch.tensor(row['deg_Mg_50C']),
            'deg_50C': torch.tensor(row['deg_50C']),
        }

    def _prep(self, row):
        row['encoded_sequence'] = [self.token2int[s] for s in row['sequence']]
        row['encoded_structure'] \
            = [self.token2int[s] for s in row['structure']]
        row['encoded_predicted_loop_type'] \
            = [self.token2int[s] for s in row['predicted_loop_type']]
        return row
