# import package

import warnings

import hydra
import numpy as np
import pandas as pd
from hydra.utils import to_absolute_path as abspath
from omegaconf import DictConfig


@hydra.main(config_path="../config", config_name="main")
def preprocess_data(config: DictConfig):
    warnings.filterwarnings("ignore")
    """
    This function performs preprocessing on the source
    """
    # getting paths from config
    raw_path = abspath(config.raw.path)
    processed_path = abspath(config.processed.dir)

    # Reading Data
    df = pd.read_excel(f"{raw_path}/{config.process.datasrc}")
    """
    Reading Data
    """
    # set the dataset's index to the data column
    df = df.set_index("date")

    """
    set the dataset's index to the data column
    """

    # config file to for standardization of the normalization block this code

    # Data normalization
    # Various data normalization techniquies are implored due to the
    # nature of the data.

    # Taking Log of selected columns
    dflog = np.log(
        df[
            [
                "ABCPI",
                "ARY",
                "ASI",
                "BLAG",
                "BLMF",
                "BLOG",
                "BLPS",
                "BLSM",
                "BLTL",
                "BLUS",
                "BLXP",
                "C1CPI",
                "C2CPI",
                "CCPI",
                "CCPS",
                "CFCPI",
                "CGRY",
                "COP",
                "CPD",
                "CPS",
                "COS",
                "CRY",
                "ECPI",
                "ER",
                "EUR",
                "EXR",
                "FCPI",
                "FHCPI",
                "FNCPI",
                "FRY",
                "GBP",
                "GRV",
                "GXP",
                "HHCPI",
                "HRY",
                "HWCPI",
                "IEP",
                "IIP",
                "IMAP",
                "IMIP",
                "IMP",
                "IRY",
                "M1",
                "M2",
                "MCPI",
                "MRY",
                "NDC",
                "NFA",
                "NORY",
                "PRY",
                "QM",
                "RCCPI",
                "RHCPI",
                "RINV",
                "RPC",
                "RPDI",
                "RR",
                "RRY",
                "RUCPI",
                "RY",
                "SD",
                "SMRY",
                "SRY",
                "TCPI",
                "TD",
                "TRY",
                "URCPI",
                "URY",
                "USD",
                "EXP",
                "MPMIS",
            ]
        ]
    )
    """
    Taking Log of selected columns
    """

    # Differencing Logged Values of selected columns
    dflogdiff = dflog[
        [
            "C1CPI",
            "C2CPI",
            "FCPI",
            "FNCPI",
            "ABCPI",
            "CFCPI",
            "HWCPI",
            "FHCPI",
            "HHCPI",
            "TCPI",
            "CCPI",
            "RCCPI",
            "ECPI",
            "RHCPI",
            "MCPI",
            "URCPI",
            "RUCPI",
            "CPD",
            "EXP",
            "IMP",
            "GXP",
            "IRY",
            "SMRY",
            "MRY",
            "SRY",
            "TRY",
            "CRY",
            "URY",
            "PRY",
            "RINV",
            "IMAP",
            "IMIP",
            "IEP",
            "IIP",
            "COP",
            "GBP",
            "EUR",
            "ASI",
            "M1",
            "QM",
            "CCPS",
            "COS",
            "SD",
            "TD",
            "RR",
            "ER",
            "NFA",
            "NDC",
            "EXR",
            "BLPS",
            "BLAG",
            "BLSM",
            "BLXP",
            "BLMF",
            "BLUS",
            "BLOG",
            "BLTL",
            "USD",
        ]
    ]
    """
    Differencing Logged Values of selected columns
    """

    dflogdiff = dflogdiff - dflogdiff.shift(4)

    # Differencing of selected values
    dfonlydiff = df[
        ["MDR1", "MDR3", "MDR6", "MDR12", "PLR", "MLR", "IBCR", "TBR", "CRR"]
    ]

    """
    Differencing of selected values
    """

    dfonlydiff = dfonlydiff - dfonlydiff.shift(4)

    # GROWTH RATE of Selec
    # ted Columns
    grdfn = df[
        [
            "HCPI",
            "RY",
            "ARY",
            "ICRY",
            "TTRY",
            "RRY",
            "ERY",
            "HRY",
            "M2",
            "CGRY",
            "NORY",
        ]
    ]
    growth_rate = ((grdfn - grdfn.shift(4)) / grdfn.shift(4)) * 100

    # Grouping of dataframes
    groupL = dflog[["ABCPI", "GRV", "RPC", "RPDI"]]

    # Note: Removed IRY from here
    groupDL = dflogdiff[
        [
            "C1CPI",
            "C2CPI",
            "FCPI",
            "FNCPI",
            "CFCPI",
            "HWCPI",
            "FHCPI",
            "HHCPI",
            "TCPI",
            "CCPI",
            "RCCPI",
            "ECPI",
            "RHCPI",
            "MCPI",
            "URCPI",
            "RUCPI",
            "CPD",
            "EXP",
            "IMP",
            "GXP",
            "RINV",
            "IMAP",
            "IMIP",
            "IEP",
            "IIP",
            "GBP",
            "EUR",
            "ASI",
            "QM",
            "CCPS",
            "COS",
            "SD",
            "TD",
            "ER",
            "NFA",
            "NDC",
            "EXR",
            "BLTL",
        ]
    ]

    # Note: Removed RY from here
    groupGR = growth_rate[["HCPI", "M2", "CGRY", "NORY"]]

    SubGroup = pd.merge(
        pd.merge(groupL, groupDL, on=["date"]),
        pd.merge(dfonlydiff, groupGR, on=["date"]),
        on=["date"],
    )

    # Final Group
    FinalGroup = pd.merge(SubGroup, df["SDR"], on=["date"])

    # Dropping all rows with NaN
    FinalGroup = FinalGroup.dropna()
    FinalGroup.to_excel(
        f"{processed_path}/{config.process.processed_outputs[0]}"
    )
    growth_rate.to_excel(
        f"{processed_path}/{config.process.processed_outputs[1]}"
    )


if __name__ == "__main__":
    preprocess_data()
