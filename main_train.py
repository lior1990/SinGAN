import json
import os
import logging
from typing import List, Tuple
import matplotlib

matplotlib.use('Agg')

from SinGAN.manipulate import SinGAN_generate, get_arguments
from SinGAN.training import train
import SinGAN.functions as functions

logger = logging.getLogger()


def _configure_logger(dir2save):
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')
    file_handler = logging.FileHandler(os.path.join(dir2save, "train.log"))
    stream_handler = logging.StreamHandler()
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)


def _cleanup_logger():
    list(map(logger.removeHandler, logger.handlers))


def parse_arguments(namespace=None):
    parser = get_arguments()
    parser.add_argument('--input_dir', help='input image dir', default='Input/Images')
    parser.add_argument('--input_name1', help='input image name 1', required=namespace is None)
    parser.add_argument('--input_name2', help='input image name 2', required=namespace is None)
    parser.add_argument('--mode', help='task to be done', default='train')
    opt = parser.parse_args(namespace=namespace)
    opt = functions.post_config(opt)
    return opt


def main(opt, generate=True):
    Gs = []
    Zs: List[Tuple] = []
    reals1 = []
    reals2 = []
    NoiseAmp = []
    dir2save = functions.generate_dir2save(opt)

    if (os.path.exists(dir2save)):
        print('trained model already exist')
    else:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass
        _configure_logger(dir2save)

        # dump configuration file to json
        with open(os.path.join(f"{dir2save}", "config.json"), "w") as fp:
            config_dict = {k: str(v) for k, v in opt.__dict__.items()}
            json.dump(config_dict, fp)

        try:
            real1 = functions.read_image(opt, image_name=opt.input_name1)
            real2 = functions.read_image(opt, image_name=opt.input_name2)
            functions.adjust_scales2image(real1, opt)
            functions.adjust_scales2image(real2, opt)
            train(opt, Gs, Zs, reals1, reals2, NoiseAmp)
            logger.info("Done training")
            if generate:
                logger.info("Generating random samples")
                SinGAN_generate(Gs, Zs, reals1, reals2, NoiseAmp, opt)
        except Exception as e:
            logger.exception("Failed")
            raise
        finally:
            logger.info("Cleaning logger")
            _cleanup_logger()


if __name__ == '__main__':
    main(parse_arguments())
