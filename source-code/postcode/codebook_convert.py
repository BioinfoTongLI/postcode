#! /usr/bin/env python3

import pandas as pd
from pathlib import Path
from starfish.core.codebook.codebook import Codebook
from starfish.types import Axes, Features
import fire


def cartana2starfish(codebook:pd.DataFrame) -> str: 
    mappings = []
    for _, row in codebook.iterrows():
        mapping = {}
        mapping[Features.TARGET] = row["gene"]
        codeward = []
        for r, c in enumerate(str(row['code'])):
            codeward.append({Axes.ROUND.value: r, Axes.CH.value: int(c)-1, Features.CODE_VALUE: 1})
        mapping[Features.CODEWORD] = codeward
        mappings.append(mapping)
    return Codebook.from_code_array(mappings, n_round=r+1, n_channel=4) # hardcoded to 4 channels since Cartana only uses 4 channels


def main(codebook_p: str):
    if codebook_p.endswith('.csv'):
        codebook = pd.read_csv(codebook_p)
        codebook = Codebook.from_code_array(codebook.values)
    elif codebook_p.endswith('.xlsx'):
        codebook = pd.read_excel(codebook_p)
    codebook = cartana2starfish(codebook)
    print(codebook)
    codebook.to_json(f"{Path(codebook_p).stem}.json")

if __name__ == '__main__':
    fire.Fire(main)