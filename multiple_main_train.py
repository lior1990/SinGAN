from argparse import Namespace

from main_train import main, parse_arguments


IMAGES = [("birds1.jpg", "birds2.jpg"), ("birds.png", "balloons_size_birds.png")]

CONFIGURATIONS = {"scale_factor": [0.8],
                  "mixed_imgs_training": [False],
                  "noise_vectors_merge_method": ["sum"],
                  "gaussian_noise_z_distance": [0],
                  "alpha": [15],
                  "replace_background": [True],
                  "nfc": [128, 32]
                  }


def _generate_opt(configuration: dict):
    namespace = Namespace(**configuration)
    return parse_arguments(namespace=namespace)


def multiple_main():

    for img1, img2 in IMAGES:
        for scale_factor in CONFIGURATIONS["scale_factor"]:
            for mixed_imgs_training in CONFIGURATIONS["mixed_imgs_training"]:
                for noise_vectors_merge_method in CONFIGURATIONS["noise_vectors_merge_method"]:
                    for gaussian_noise_z_distance in CONFIGURATIONS["gaussian_noise_z_distance"]:
                        for alpha in CONFIGURATIONS["alpha"]:
                            for replace_background in CONFIGURATIONS["replace_background"]:
                                for nfc in CONFIGURATIONS["nfc"]:
                                    mixed_str = "mixed" if mixed_imgs_training else ""
                                    exp_name = f"{mixed_str}_{noise_vectors_merge_method}_g-dist{gaussian_noise_z_distance}_alpha{alpha}_nfc{nfc}"
                                    config = {
                                        "scale_factor": scale_factor,
                                        "mixed_imgs_training": mixed_imgs_training,
                                        "noise_vectors_merge_method": noise_vectors_merge_method,
                                        "gaussian_noise_z_distance": gaussian_noise_z_distance,
                                        "alpha": alpha,
                                        "exp_name": exp_name,
                                        "replace_background": replace_background,
                                        "nfc": nfc,
                                        "min_nfc": nfc,
                                    }
                                    _run_with_images(config, exp_name, img1, img2)
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
