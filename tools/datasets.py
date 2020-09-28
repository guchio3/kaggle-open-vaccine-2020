import warnings

import numpy as np

import torch
from torch.utils.data import Dataset

BPP_DIR = './inputs/origin/bpps'


class OpenVaccineDataset(Dataset):
    def __init__(self, mode, df, logger=None, debug=False):
        self.mode = mode
        self.token2int = {x: i for i, x in enumerate('().ACGUBEHIMSX')}
        # self.df = self._prep(df.reset_index(drop=True))
        self.df = df.reset_index(drop=True)
        self.logger = logger
        self.debug = debug

        # initialize df
        self.df['encoded_sequence'] = None
        self.df['encoded_structure'] = None
        self.df['encoded_predicted_loop_type'] = None
        self.df['bpp_max'] = None
        self.df['bpp_sum'] = None
        self.df['bpp_non_zero_ratio'] = None
        self.df['structure_dist'] = None
        self.df['structure_depth'] = None

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
                self.df[label_col] = np.nan

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]

        row = self._prep(row)

        return {
            'id': row['id'],
            'encoded_sequence': torch.tensor(row['encoded_sequence']),
            'encoded_structure': torch.tensor(row['encoded_structure']),
            'encoded_predicted_loop_type': torch.tensor(row['encoded_predicted_loop_type']),
            'bpp_max': torch.tensor(row['bpp_max']),
            'bpp_max_stand': torch.tensor(row['bpp_max_stand']),
            'bpp_sum': torch.tensor(row['bpp_sum']),
            'bpp_sum_stand': torch.tensor(row['bpp_sum_stand']),
            'bpp_non_zero_ratio': torch.tensor(row['bpp_non_zero_ratio']),
            'bpp_non_zero_ratio_stand': torch.tensor(row['bpp_non_zero_ratio_stand']),
            'structure_dist': torch.tensor(row['structure_dist']),
            'structure_dist_stand': torch.tensor(row['structure_dist_stand']),
            'structure_depth': torch.tensor(row['structure_depth']),
            'structure_depth_stand': torch.tensor(row['structure_depth_stand']),
            'reactivity': torch.tensor(row['reactivity']),
            'deg_Mg_pH10': torch.tensor(row['deg_Mg_pH10']),
            'deg_pH10': torch.tensor(row['deg_pH10']),
            'deg_Mg_50C': torch.tensor(row['deg_Mg_50C']),
            'deg_50C': torch.tensor(row['deg_50C']),
        }

    # def _prep(self, df):
    #     # initialize df
    #     df['encoded_sequence'] = None
    #     df['encoded_structure'] = None
    #     df['encoded_predicted_loop_type'] = None
    #     df['bpp_max'] = None
    #     df['bpp_sum'] = None
    #     df['bpp_non_zero_ratio'] = None
    #     df['structure_dist'] = None
    #     df['structure_depth'] = None
    #     for i, row in df.iterrows():
    #         df.loc[i, 'encoded_sequence'] \
    #             = [self.token2int[s] for s in row['sequence']]
    #         df.loc[i, 'encoded_structure'] \
    #             = [self.token2int[s] for s in row['structure']]
    #         df.loc[i, 'encoded_predicted_loop_type'] \
    #             = [self.token2int[s] for s in row['predicted_loop_type']]
    #         bpp = np.load(f'{BPP_DIR}/{row["id"]}.npy')
    #         df.loc[i, 'bpp_max'] = bpp.max(axis=1).tolist()

    def _prep(self, row):
        warnings.simplefilter('ignore')

        row['encoded_sequence'] = [self.token2int[s] for s in row['sequence']]
        row['encoded_structure'] \
            = [self.token2int[s] for s in row['structure']]
        row['encoded_predicted_loop_type'] \
            = [self.token2int[s] for s in row['predicted_loop_type']]

        bpp = np.load(f'{BPP_DIR}/{row["id"]}.npy')
        row['bpp_max'] = bpp.max(axis=1).tolist()
        row['bpp_max_stand'] = ((bpp.max(axis=1) - 0.4399965348227675)
                                / 0.4396429415011541).tolist()
        row['bpp_sum'] = bpp.sum(axis=1).tolist()
        row['bpp_sum_stand'] = ((bpp.sum(axis=1) - 0.4824247336126996)
                                / 0.44554374501860566).tolist()
        row['bpp_non_zero_ratio'] \
            = ((bpp > 0).sum(axis=1) / bpp.shape[0]).tolist()
        row['bpp_non_zero_ratio_stand'] \
            = ((((bpp > 0).sum(axis=1) / bpp.shape[0])
                - 0.05980655800051093 / 0.0726128883002326)).tolist()
        structure_dist, structure_depth \
            = self._mk_structure_features(row['structure'])
        row['structure_dist'] = structure_dist
        row['structure_dist_stand'] \
            = ((np.asarray(structure_dist)
                - 11.048005394870676) / 18.50391454966171).tolist()
        row['structure_depth'] = structure_depth
        row['structure_depth_stand'] \
            = ((np.asarray(structure_depth)
                - 5.784752215100881) / 6.936187530818166).tolist()
        return row

    def _mk_structure_features(self, structure):
        structure_dist = np.full(len(structure), -1, dtype=int).tolist()
        structure_depth = []
        stack = []
        for i, s in enumerate(structure):
            if s == "(":
                stack.append(i)
            elif s == ")":
                j = stack.pop()
                structure_dist[i] = i - j
                structure_dist[j] = i - j
            structure_depth.append(len(stack))
        return structure_dist, structure_depth
