import logging

import numpy as np
import pandas as pd

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
ch.setFormatter(formatter)


def wrap_to2pi(angle):
    # TODO Description and logging
    if angle < 0.0:
        angle = 2.0 * np.pi + angle
    return angle


def mm2m(val):
    # TODO Description and logging
    return val / 1000.0


class DataCurator:
    """
    Stump of class handing data curation. In current version handles only
    subsampling.
    """

    # TODO Update description and add logging
    def __init__(self, params):
        """
        Constructor for data curator class.
        :param params: Parameters guiding the curation (see json schema
        @config/curator_shema.json)
        :type params: dict
        """
        # setting logging
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.setLevel(logging.DEBUG)
        self.log.addHandler(ch)
        # load params
        self.params = params
        # variables
        self.input_file = None
        self.output_file = None
        # logging
        self.log.info("Parameters for data curation {}".format(self.params))

    def curate(self):
        """
        Function running main curation loop
        """
        if self.params["chunk_size"] != 0:
            self.__chunk_processing()

    def __chunk_processing(self):
        """
        Function for curating large csv files in chunks.
        """
        header = True
        self.log.info(
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
            self.log.info("Processing chunk {}".format(i))

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

    def __subsample(self, chunk):
        """
        Function for subsampling data. Depending on the flag different
        subsampling policies can be used. Currently only available one is to
        keep very n-th line.

        :param chunk: Data to be subsample :type chunk: dataframe :return:
        Subsampled data :rtype: dataframe
        """
        result = []
        if self.params["subsample"]["method"] == "line_keep":
            result = chunk.iloc[:: self.params["subsample"]["parameter"], :]
        self.log.info(
            "Was {} rows, remained {} rows".format(
                len(chunk.index), len(result.index)
            )
        )
        return result

    def __drop_columns(self, chunk):
        """
        Function dropping redundant columns from the original data

        :param chunk: Data to be edited
        :type chunk: dataframe
        :return: edited chunk
        :rtype: dataframe
        """
        for c in self.params["drop_columns"]:
            chunk = chunk.drop(c, axis=1)
            self.log.info("Dropping column: {}".format(c))
        return chunk

    def __process_column(self, chunk):
        # TODO Description and logging
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

    def __replace_header(self, chunk):
        for old_name, new_name in zip(
            self.params["replace_header"]["input"],
            self.params["replace_header"]["replacement"],
        ):
            chunk = chunk.rename(columns={old_name: new_name})
        return chunk
