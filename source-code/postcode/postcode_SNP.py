#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import fire
from pathlib import Path
from avg_spot_profile import main as average_spot_profiles
from max_proj_spot_profile import main as max_proj_spot_profiles
from decoding_functions import decoding_function, decoding_output_to_dataframe
from prepare_ISS import main as prepare_iss
'''
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
'''
def prepare_codebook_MERFISH(codebook_path, N_readouts, gene_name = 'gene', readout_name = 'Readout_'):
    codebook_in = pd.read_csv(codebook_path)
    n_genes = codebook_in.shape[0]; n_rounds = N_readouts;
    codebook_3d = np.zeros((n_genes, 1, n_rounds), dtype =  'uint8')
    for ng in range(n_genes):
        for nr in range(n_rounds):
            column_name =  readout_name + str(nr+1)
            codebook_3d[ng, 0, nr] = int(codebook_in[column_name][ng])
    gene_list_obj = np.array(codebook_in[gene_name], dtype = object)
    return gene_list_obj, codebook_3d, n_genes


                             
def decode(spot_locations_p: str, spot_profile_p: str, codebook_p: str, img_p = 'None', readouts_csv = 'None', mode = 'ISS', start_cycle = 2, keep_noises=True, min_prob = 0.9) -> pd.DataFrame:
    """
    Decodes spots using the Postcode algorithm.

    Args:
        spot_locations_p (str): A file path to pandas DataFrame containing the spot locations.
        spot_profile_p (str): A file path to numpy array containing the spot profiles (N x C x R).
        codebook (str): csv Cortana-like codebook with only one channel and number of rounds (readouts) (for MERFISH mode)
        codebook (str): csv Codebook with only one channel and number of rounds (readouts) (for MERFISH mode)
        readouts_csv (str): csv file with table which describes link between cycle-channels and readouts (only for MERFISH mode)
        keep_noises (bool, optional): Whether to keep spots that were classified as 'background' or 'infeasible'.
        min_prob: [0,1] - value of minimum allowed probability of decoded spot
            Defaults to True.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the decoded spots and their locations.
    """
    stem = Path(spot_profile_p).stem
    spot_locations = pd.read_csv(spot_locations_p)
    #spot_profile_raw = np.load(spot_profile_p, allow_pickle=True)
    #spot_profile = max_proj_spot_profiles(spot_profile_p, readouts_csv)
    

    if mode == 'MERFISH':
        spot_profile, N_readouts = average_spot_profiles(spot_profile_p, readouts_csv)
        gene_list, codebook_arr, K = prepare_codebook_MERFISH(codebook_p, N_readouts)
    elif mode =='ISS':
        codebook_arr, spot_profile, gene_list, K = prepare_iss(codebook_p, spot_profile_p, image_path = img_p, start_cycle = start_cycle)
    else:
        raise ValueError('Mode should be "ISS" or "MERFISH"')
    
    #print(codebook_arr)
    #print(spot_profile)

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
    decoded_df_s = decoded_df_s[decoded_df_s['Probability']>min_prob]
    
    if keep_noises:
        decoded_df_s.to_csv(f"{stem}_decoded_spots.csv", index=False)
    else:
        # Remove infeasible and background codes
        decoded_df_s[~np.isin(decoded_df_s['Name'], ['background', 'infeasible'])].reset_index(drop=True).to_csv(f"{stem}_decoded_spots.csv", index=False)
        
    

if __name__ == "__main__":
    fire.Fire(decode)
