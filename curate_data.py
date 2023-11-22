import argparse
import logging

from mod.curation import DataCurator
from mod.utils import extended_validator, get_local_settings

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
ch.setFormatter(formatter)

log = logging.getLogger(__file__)
log.setLevel(logging.DEBUG)
log.addHandler(ch)

parser = argparse.ArgumentParser()
parser.add_argument("config_file", help="JSON configuration file")
parser.add_argument("local_settings_file", help="JSON local settings file")
parser.add_argument("schema_file", help="JSON schema file")
args = parser.parse_args()

log.info("Path to config file {}".format(args.config_file))
log.info("Path to local settings file {}".format(args.local_settings_file))
log.info("Path to schema file {}".format(args.schema_file))
success, params = extended_validator(args.config_file, args.schema_file)
if not success:
    log.error("There is an issue with the config file: {}".format(params))
    exit(-1)
success, local = get_local_settings(args.local_settings_file)
if not success:
    log.error("There is an issue with the config file: {}".format(local))
    exit(-1)
params["input_file"] = local["dataset_folder"] + params["input_file"]
params["output_file"] = local["dataset_folder"] + params["output_file"]
dc = DataCurator(params)
dc.curate()
