#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from starfish.core.codebook.codebook import Codebook
from starfish.core.spots.DecodeSpots.trace_builders import build_spot_traces_exact_match
from starfish.core.types import SpotFindingResults
from postcode.decoding_functions import decoding_function, decoding_output_to_dataframe

def main(spots: SpotFindingResults, codebook: str, keep_noises=True) -> pd.DataFrame:
    """
    Decodes spots using the Postcode algorithm.

    Args:
        spots (SpotFindingResults): A SpotFindingResults object containing the spots to be decoded.
        codebook (str): A JSON string representing the codebook to be used for decoding.
        keep_noises (bool, optional): Whether to keep spots that were classified as 'background' or 'infeasible'.
            Defaults to True.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the decoded spots and their locations.
    """
    codebook = Codebook.from_json(codebook)

    # Format starfish spots for use in postcode
    bd_table = build_spot_traces_exact_match(spots)
    spots_s = np.swapaxes(bd_table.data, 1, 2)
    spots_loc_s = pd.DataFrame(columns=['X', 'Y', 'Z'])
    spots_loc_s['X'] = np.array(bd_table.x)
    spots_loc_s['Y'] = np.array(bd_table.y)
    spots_loc_s['Z'] = np.array(bd_table.z)
    barcodes_01 = np.swapaxes(np.array(codebook), 1, 2)
    K = barcodes_01.shape[0]

    # Decode using postcode
    out = decoding_function(spots_s, barcodes_01, print_training_progress=True)

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

    decoded_df_s = pd.concat([decoded_spots_df, spots_loc_s], axis=1)
    if keep_noises:
        return decoded_df_s
    # Remove infeasible and background codes
    return decoded_df_s[~np.isin(decoded_df_s['Name'], ['background', 'infeasible'])].reset_index(drop=True)