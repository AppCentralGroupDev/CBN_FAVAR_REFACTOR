# The basic idea when using PCA as a tool for feature selection is to select
# variables according to the magnitude (from largest to smallest in absolute
# values) of their coefficients (loadings).


import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

import hydra
from hydra.utils import to_absolute_path as abspath
from omegaconf import DictConfig


@hydra.main(config_path="../config", config_name="main")
def Features_PCA(config: DictConfig):

    # getting paths from config
    raw_path = abspath(config.raw.path)
    processed_path = abspath(config.processed.dir)
    final_path = abspath(config.final.dir)

    # Reading Data
    df = pd.read_excel(f"{raw_path}/{config.process.datasrc}")
    # set the dataset's index to the data column
    df = df.set_index("date")
    FinalGroup = pd.read_excel(
        f"{processed_path}/{config.process.processed_outputs[0]}"
    )
    # set the dataset's index to the data column
    FinalGroup = FinalGroup.set_index("date")

    growth_rate = pd.read_excel(
        f"{processed_path}/{config.process.processed_outputs[1]}"
    )
    growth_rate = growth_rate.set_index("date")

    # Generating Principal Components
    pca5 = PCA(n_components=5)
    principalComponents = pca5.fit_transform(FinalGroup)

    # Converting PCs to Dataframe
    Dfpc = pd.DataFrame(
        data=principalComponents,
        columns=["pc1", "pc2", "pc3", "pc4", "pc5"],
        index=FinalGroup.index,
    )

    # Differencing the non stationary PCs to achieve stationarity
    DfpcDiff = Dfpc - Dfpc.shift()

    DfpcDiff.dropna(inplace=True)

    # Create a dataframe with PCs and dlRY
    df_sub = growth_rate["RY"].iloc[3:]
    # Create a dataframe by appending df_sub to DfpcDiff as a new column
    df_pcsRY = DfpcDiff.assign(dlRY=df_sub)
    # Sampling of Endogenous Variables
    endog = df_pcsRY.loc[
        "2017-03-31":"2021-12-31", ["pc1", "pc2", "pc3", "pc4", "pc5", "dlRY"]
    ]

    exog = np.log(
        df[
            [
                "COP50",
                "COP55",
                "COP60",
                "COP65",
                "COP70",
                "COP75",
                "COP80",
                "COP85",
                "COP90",
                "COP95",
                "COP100",
                "COP105",
                "COP110",
                "MPMIS1",
            ]
        ]
    )

    log_gxp = np.log(df["GXP1"])
    dfgxp = log_gxp - log_gxp.shift(4)

    exog = pd.merge(exog, dfgxp, on=["date"])

    # Sampling of Exogenous Variables
    exogC = exog.loc["2017-03-31":"2021-12-31", ["COP50", "GXP1", "MPMIS1"]]

    # Export training data
    endog.to_csv(f"{final_path}/{config.process.final_outputs[0]}")
    exogC.to_csv(f"{final_path}/{config.process.final_outputs[1]}")


if __name__ == "__main__":
    Features_PCA()
