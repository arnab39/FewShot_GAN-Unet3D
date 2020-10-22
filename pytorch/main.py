import argparse
from utils.config import *

from agents import *

def get_args():
    # parse the path of the json config file
    arg_parser = argparse.ArgumentParser(description="FewShot 3D Medical Image Segmenation")
    arg_parser.add_argument(
        'config',
        metavar='config_json_file',
        default='None',
        help='The Configuration file in json format')

    args = arg_parser.parse_args()
    return args

def main():
    # parse the config json file
    args = get_args()
    config = process_config(args.config, args)

    # Create the Agent and pass all the configuration to it then run it..
    agent_class = globals()[config.agent]
    agent = agent_class(config)
    agent.run()
    agent.finalize()

if __name__ == '__main__':
    main()
