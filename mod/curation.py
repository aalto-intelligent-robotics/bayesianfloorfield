import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def wrap_to2pi(angle: float) -> float:
    # TODO Description and logging
    return angle % (2 * np.pi)


def mm2m(val: float) -> float:
    # TODO Description and logging
    return val / 1000.0


class DataCurator:
    """Stump of class handing data curation. In current version handles only
    subsampling."""

    def __init__(self, params: dict):
        """Init `DataCurator` class.

        Args:
            params (dict): Parameters guiding the curation (see json schema
            @config/curator_shema.json)
        """
        # load params
        self.params = params
        # variables
        self.input_file = None
        self.output_file = None
        # logging
        logger.info("Parameters for data curation {}".format(self.params))

    def curate(self):
        """Function running main curation loop"""
        if self.params["chunk_size"] != 0:
            self.__chunk_processing()

    def __chunk_processing(self):
        """Function for curating large csv files in chunks."""
        header = True
        logger.info(
            "Processing in chunks of size {}".format(self.params["chunk_size"])
        )
        if "add_header" in self.params:
            self.input_file = pd.read_csv(
                self.params["input_file"],
                chunksize=self.params["chunk_size"],
                names=self.params["add_header"],
            )
        else:
            self.input_file = pd.read_csv(
                self.params["input_file"], chunksize=self.params["chunk_size"]
            )
        for i, chunk in enumerate(self.input_file):
            logger.info("Processing chunk {}".format(i))

            if (
                self.params["subsample"]["method"] != ""
                and self.params["subsample"]["parameter"] != ""
            ):
                chunk = self.__subsample(chunk)
            if self.params["drop_columns"] != {}:
                chunk = self.__drop_columns(chunk)
            if self.params["process_columns"] != {}:
                chunk = self.__process_column(chunk)
            if self.params["replace_header"] != {}:
                chunk = self.__replace_header(chunk)

            chunk.to_csv(
                self.params["output_file"],
                header=header,
                mode="a",
                index=False,
            )

            header = False

    def __subsample(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Function for subsampling data. Depending on the flag different
        subsampling policies can be used. Currently only available one is to
        keep very n-th line.

        Args:
            chunk (pd.Dataframe): Data to be subsampled

        Returns:
            pd.Dataframe: Subsampled data
        """
        result = []
        if self.params["subsample"]["method"] == "line_keep":
            result = chunk.iloc[:: self.params["subsample"]["parameter"], :]
        logger.info(
            "Was {} rows, remained {} rows".format(
                len(chunk.index), len(result.index)  # type: ignore
            )
        )
        return result

    def __drop_columns(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Function dropping redundant columns from the original data

        Args:
            chunk (pd.Dataframe): Data to be edited

        Returns:
            pd.Dataframe: Edited data
        """
        for c in self.params["drop_columns"]:
            chunk = chunk.drop(c, axis=1)
            logger.info("Dropping column: {}".format(c))
        return chunk

    def __process_column(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Function processing columns in the original data

        Args:
            chunk (pd.Dataframe): Data to be edited

        Returns:
            pd.Dataframe: Edited data
        """
        for function_name, column_name in zip(
            self.params["process_columns"]["method"],
            self.params["process_columns"]["columns"],
        ):
            print(function_name, column_name)
            if function_name == "WrapTo2pi":
                chunk[column_name] = chunk[column_name].apply(wrap_to2pi)
            if function_name == "mm2m":
                chunk[column_name] = chunk[column_name].apply(mm2m)
        return chunk

    def __replace_header(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Function replacing the header row from the original data

        Args:
            chunk (pd.Dataframe): Data to be edited

        Returns:
            pd.Dataframe: Edited data
        """
        for old_name, new_name in zip(
            self.params["replace_header"]["input"],
            self.params["replace_header"]["replacement"],
        ):
            chunk = chunk.rename(columns={old_name: new_name})
        return chunk
