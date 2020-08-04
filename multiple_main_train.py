from argparse import Namespace

from main_train import main, parse_arguments


CONFIGURATIONS = {"scale_factor": [0.75, 0.85],
                  "mixed_imgs_training": [True, False],
                  "noise_vectors_merge_method": ["sum", "cat"],
                  "gaussian_noise_z_distance": [0,1,2],
                  "alpha": [10, 20]}


def _generate_opt(configuration: dict):
    namespace = Namespace(**configuration)
    return parse_arguments(namespace=namespace)


def multiple_main():

    for scale_factor in CONFIGURATIONS["scale_factor"]:
        for mixed_imgs_training in CONFIGURATIONS["mixed_imgs_training"]:
            for noise_vectors_merge_method in CONFIGURATIONS["noise_vectors_merge_method"]:
                for gaussian_noise_z_distance in CONFIGURATIONS["gaussian_noise_z_distance"]:
                    for alpha in CONFIGURATIONS["alpha"]:
                        mixed_str = "mixed" if mixed_imgs_training else ""
                        exp_name = f"{mixed_str}_{noise_vectors_merge_method}_g-dist{gaussian_noise_z_distance}_alpha{alpha}"
                        namespace = Namespace(**{"scale_factor": scale_factor,
                                                 "mixed_imgs_training": mixed_imgs_training,
                                                 "noise_vectors_merge_method": noise_vectors_merge_method,
                                                 "gaussian_noise_z_distance": gaussian_noise_z_distance,
                                                 "alpha": alpha,
                                                 "exp_name": exp_name})
                        opt = parse_arguments(namespace=namespace)
                        print(f"Working on {exp_name}")
                        main(opt, generate=False)


if __name__ == '__main__':
    multiple_main()
