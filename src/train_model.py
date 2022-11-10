"""
This is the demo code that uses hy                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      dra to access the parameters in under the directory config.

Author: Clinton Mbataku
"""

import logging
import os
import warnings
from urllib.parse import urlparse

import hydra
import mlflow
import mlflow.statsmodels
import pandas as pd
from hydra.utils import to_absolute_path as abspath
from omegaconf import DictConfig
from statsmodels.tsa.api import VAR

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


@hydra.main(config_path="../config", config_name="main")
def main(config: DictConfig):

    """
    Main function
    """

    """
    environment variables for s3 bucket
    """

    os.environ["AWS_ACCESS_KEY_ID"] = config.model.AWS_ACCESS_KEY_ID
    os.environ["AWS_SECRET_ACCESS_KEY"] = config.model.AWS_SECRET_ACCESS_KEY
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = config.model.MLFLOW_S3_ENDPOINT_URL

    # getting paths from config
    final_path = abspath(config.final.dir)

    raw_path = abspath(config.raw.path)
    processed_path = abspath(config.processed.dir)

    # Read training data csv
    endog = pd.read_csv(f"{final_path}/{config.process.final_outputs[0]}")
    exogC = pd.read_csv(f"{final_path}/{config.process.final_outputs[1]}")
    """
    Read training data csv
    """

    # set index
    endog = endog.set_index("date")
    exogC = exogC.set_index("date")
    """
    set index
    """

    # from statsmodels.tools.eval_measures import rmse, aic
    warnings.filterwarnings("ignore")

    remote_server_uri = (
        config.model.remote_server_uri
    )  # set to your server URI
    """
    set remote tracking sever uri
    """
    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment(config.model.experiment_name)
    """
    set experiment name
    """

    mlflow.statsmodels.autolog()

    """
    start auto logging
    """

    with mlflow.start_run():
        var = VAR(endog, exogC)

        x = var.select_order()
        print(x.summary())

        results = var.fit(1)
        # We can check the summary of the model by.
        print(results.summary())
        """
        We can check the summary of the model by.
        """

        # Log_likelihood = results.llf
        # AIC = results.aic
        # BIC = results.bic
        # HQIC = results.hqic

        # saving all artifacts
        # saving all files used to mlflow

        mlflow.log_artifact(f"{final_path}/{config.process.final_outputs[0]}")
        mlflow.log_artifact(f"{final_path}/{config.process.final_outputs[1]}")
        mlflow.log_artifact(f"{raw_path}/{config.process.datasrc}")
        mlflow.log_artifact(
            f"{processed_path}/{config.process.processed_outputs[0]}"
        )
        mlflow.log_artifact(
            f"{processed_path}/{config.process.processed_outputs[1]}"
        )

        # # log metrics if you use the manual log method
        # mlflow.log_metrics({"Log likelihood": Log_likelihood,
        #                     "AIC": AIC,
        #                     "BIC": BIC,
        #                     "HQIC": HQIC})

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.statsmodels.log_model(
                results, "model", registered_model_name=config.model.name
            )
            """
            Register Model and name mode
            """
        else:
            mlflow.statsmodels.log_model(results, "model")


if __name__ == "__main__":
    main()
