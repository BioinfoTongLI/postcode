#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import fire
from pathlib import Path

from decoding_functions import decoding_function, decoding_output_to_dataframe

def ReadPrepCodebook(codebook_path):
    #I consider single channel!
    codebook_in = pd.read_csv(codebook_path)
    codes = codebook_in['code']
    n_genes = len(codes); n_rounds = len(str(codes[0])); 
    codebook_3d = np.zeros((n_genes, 1, n_rounds), dtype =  'uint8')
    for ng in range(n_genes):
        for nr in range(n_rounds):
            codebook_3d[ng, 0, nr] = int(str(codes[ng])[nr])
    gene_list_obj = np.array(codebook_in['gene'], dtype = object)
    return gene_list_obj, codebook_3d, n_genes
    

def decode(spot_locations_p: str, spot_profile_p: str, codebook_p: str, keep_noises=True) -> pd.DataFrame:
    """
    Decodes spots using the Postcode algorithm.

    Args:
        spot_locations_p (str): A file path to pandas DataFrame containing the spot locations.
        spot_profile_p (str): A file path to numpy array containing the spot profiles (N x C x R).
        codebook (str): Cortana-like codebook with only one channel and number of rounds (readouts)
        keep_noises (bool, optional): Whether to keep spots that were classified as 'background' or 'infeasible'.
            Defaults to True.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the decoded spots and their locations.
    """
    stem = Path(spot_profile_p).stem
    spot_locations = pd.read_csv(spot_locations_p)
    spot_profile = np.load(spot_profile_p, allow_pickle=True)
    
    
    gene_list, codebook_arr, K = ReadPrepCodebook(codebook_p)
    #print(codebook_arr)
    #assert spot_locations.shape[0] == spot_profile.shape[0]
    # Decode using postcode
    out = decoding_function(spot_profile, codebook_arr, print_training_progress=False)
    
    
    # Reformat output into pandas dataframe
    df_class_names = np.concatenate((gene_list,
                                        ['infeasible', 'background', 'nan']))
    barcodes_0123 = codebook_arr[:,0,:]
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

if __name__ == "__main__":
    fire.Fire(decode)
