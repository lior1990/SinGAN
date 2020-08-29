from argparse import Namespace

from main_train import main, parse_arguments


IMAGES = [("birds.png", "balloons_size_birds.png"), ("birds1.jpg", "birds2.jpg")]

CONFIGURATIONS = {
                  "mask_activation_fn": ["sigmoid", "relu_sign", "tanh_relu"],
                  "cyclic_lr": [False, True],
                  }
DEFAULT_PARAMS = {
    "scale_factor": 0.8,
    "mixed_imgs_training": False,
    "noise_vectors_merge_method": "sum",
    "gaussian_noise_z_distance": 0,
    "alpha": 40,
    "replace_background": True,
    "nfc": 64,
    "min_nfc": 64,
    "weight_decay_d_mask1": 0.1,
    "weight_decay_d_mask2": 0.1,
    "num_layer": 6,
}


def _generate_opt(configuration: dict):
    namespace = Namespace(**configuration)
    return parse_arguments(namespace=namespace)


def multiple_main():

    for img1, img2 in IMAGES:
        for mask_activation_fn in CONFIGURATIONS["mask_activation_fn"]:
            for cyclic_lr in CONFIGURATIONS["cyclic_lr"]:
                exp_name = f"{mask_activation_fn}_{cyclic_lr}"
                config = {
                    "cyclic_lr": cyclic_lr,
                    "mask_activation_fn": mask_activation_fn,
                    "exp_name": exp_name,
                }
                config.update(DEFAULT_PARAMS)
                # _run_with_images(config, exp_name, img1, img2)
                _run_with_images(config, exp_name, img2, img1)


def _run_with_images(config, exp_name, img1, img2):
    inputs_direction = {
        "input_name1": img1,
        "input_name2": img2,
    }
    config.update(inputs_direction)
    namespace = Namespace(**config)
    opt = parse_arguments(namespace=namespace)
    print(f"Working on {exp_name} with img1: {img1}, img2: {img2}")
    try:
        main(opt, generate=False)
    except:
        print(f"{exp_name} with img1: {img1}, img2: {img2} failed!")


if __name__ == '__main__':
    multiple_main()
