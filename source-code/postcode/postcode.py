#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import fire
from pathlib import Path

from starfish.core.codebook.codebook import Codebook
from postcode.decoding_functions import decoding_function, decoding_output_to_dataframe

def decode(spot_locations_p: str, spot_profile_p: str, codebook_p: str, keep_noises=True) -> pd.DataFrame:
    """
    Decodes spots using the Postcode algorithm.

    Args:
        spot_locations_p (str): A file path to pandas DataFrame containing the spot locations.
        spot_profile_p (str): A file path to numpy array containing the spot profiles (N x C x R).
        codebook (str): A starfish Codebook object containing the barcode sequences.
        keep_noises (bool, optional): Whether to keep spots that were classified as 'background' or 'infeasible'.
            Defaults to True.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the decoded spots and their locations.
    """
    stem = Path(spot_locations_p).stem
    spot_locations = pd.read_csv(spot_locations_p)
    spot_profile = np.load(spot_profile_p, allow_pickle=True)
    codebook = Codebook.open_json(codebook_p)
    assert spot_locations.shape[0] == spot_profile.shape[0]

    barcodes_01 = np.swapaxes(np.array(codebook), 1, 2)
    K = barcodes_01.shape[0]

    # Decode using postcode
    out = decoding_function(spot_profile, barcodes_01, print_training_progress=False)

    # Reformat output into pandas dataframe
    df_class_names = np.concatenate((codebook.target.values,
                                        ['infeasible', 'background', 'nan']))
    barcodes_0123 = np.argmax(np.array(codebook), axis=2)
    channel_base = ['T', 'G', 'C', 'A']
    barcodes_AGCT = np.empty(K, dtype='object')
    for k in range(K):
        barcodes_AGCT[k] = ''.join(list(np.array(channel_base)[barcodes_0123[k, :]]))
    df_class_codes = np.concatenate((barcodes_AGCT, ['NA', '0000', 'NA']))
    decoded_spots_df = decoding_output_to_dataframe(out, df_class_names, df_class_codes)

    decoded_df_s = pd.concat([decoded_spots_df, spot_locations], axis=1)
    if keep_noises:
        decoded_df_s.to_csv(f"{stem}_decoded_spots.csv", index=False)
    else:
        # Remove infeasible and background codes
        decoded_df_s[~np.isin(decoded_df_s['Name'], ['background', 'infeasible'])].reset_index(drop=True).to_csv(f"{stem}_decoded_spots.csv", index=False)


def main():
    fire.Fire(decode)