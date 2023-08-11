"""
The methods in start_annotation.py follow, with slight variations, the work of:

Braker, J., Heinemann, L., Schreieder, T.: Aramis at touché 2022: Argument detection in pictures using machine learning.
Working Notes Papers of the CLEF (2022)

Carnot, M.L., Heinemann, L., Braker, J., Schreieder, T., Kiesel, J., Fröbe,
M., Potthast, M., Stein, B.: On Stance Detection in Image Retrieval for Argumentation. In: Proc. of SIGIR. ACM (2023)

Schreieder, T., Braker, J.: Touché 2022 Best of Labs: Neural Image Retrieval for Argumentation. CLEF (2023)

The original source code can be found at: https://github.com/webis-de/SIGIR-23
"""

import argparse
import logging
import pathlib
import sys
from typing import Any, Dict

from config import Config
from annotation import start_server


args: Dict[str, Any] = None


def init_logging():
    """
    Method where the root logger is setup
    """

    root = logging.getLogger()
    # root.setLevel(logging.DEBUG)
    root.setLevel(logging.INFO)

    root.info('Logging initialised')
    root.debug('Set to debug level')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dir", type=pathlib.Path,
                        dest='data_dir', help='Path to input directory.')
    parser.add_argument("-o", "--output-dir", type=pathlib.Path,
                        dest='out_dir', help='Path to output directory.')

    parser.add_argument("-w", "--working-dir", type=pathlib.Path,
                        dest='work_dir', help='Path to working directory. (Location of index/neural net models)')
    parser.add_argument("-cfg", "--config", default=pathlib.Path('config.json'), type=pathlib.Path,
                        dest='config', help='Path to config.json file.')
    parser.add_argument("-f", "--image_format", action='store_true',
                        dest='image_format', help='Specifies format of input data. See README for definition.')

    parser.add_argument('-web', '--website', action='store_true', dest='annotation',
                        help='Start flask web server.')
    parser.add_argument('-p', '--port', type=int, dest='port', default=5000,
                        help='Port for web server.')
    parser.add_argument('-host', '--host', type=str, dest='host', default='0.0.0.0',
                        help='Host address for web server.')

    global args
    args = parser.parse_args()
    args = vars(args)

    if 'config' in args.keys():
        Config._save_path = args['config']

    cfg = Config.get()
    if 'data_dir' in args.keys() and args['data_dir'] is not None:
        cfg.data_dir = args['data_dir']
    if 'out_dir' in args.keys() and args['out_dir'] is not None:
        cfg.output_dir = args['out_dir']
    if 'work_dir' in args.keys() and args['work_dir'] is not None:
        cfg.working_dir = args['work_dir']
    if 'image_format' in args.keys():
        cfg.data_image_format = args['image_format']

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.working_dir.mkdir(parents=True, exist_ok=True)
    cfg.save()


def handle_args():
    if args['annotation']:
        log.info('Start flask annotation with method tag %s')
        start_server(host=args['host'], port=args['port'])
        sys.exit(0)

    main()


def start_flask():
    start_server()


def main():
    """
    normal program run
    :return:
    """
    log.info('do main stuff')
    pass


if __name__ == '__main__':
    parse_args()
    init_logging()
    log = logging.getLogger('startup')
    try:
        handle_args()
    except Exception as e:
        log.error(e, exc_info=True)
        raise
