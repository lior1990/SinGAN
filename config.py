import argparse


def get_arguments():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--mode', help='task to be done', default='train')
    #workspace:
    parser.add_argument('--not_cuda', action='store_true', help='disables cuda', default=0)
    parser.add_argument('--cuda_id', help='the cuda id of the GPU to train on',default='0')

    
    #load, input, save configurations:
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--netD_mask1', default='', help="path to netD (to continue training)")
    parser.add_argument('--netD_mask2', default='', help="path to netD (to continue training)")
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--nc_z',type=int,help='noise # channels',default=3)
    parser.add_argument('--nc_im',type=int,help='image # channels',default=3)
    parser.add_argument('--out',help='output folder',default='Output')
        
    #networks hyper parameters:
    parser.add_argument('--nfc', type=int, default=32)
    parser.add_argument('--min_nfc', type=int, default=32)
    parser.add_argument('--ker_size',type=int,help='kernel size',default=3)
    parser.add_argument('--num_layer',type=int,help='number of layers',default=5)
    parser.add_argument('--stride',help='stride',default=1)
    parser.add_argument('--padd_size',type=int,help='net pad size',default=0)#math.floor(opt.ker_size/2)
        
    #pyramid parameters:
    parser.add_argument('--scale_factor',type=float,help='pyramid scale factor',default=0.75)#pow(0.5,1/6))
    parser.add_argument('--noise_amp',type=float,help='addative noise cont weight',default=0.1)
    parser.add_argument('--min_size',type=int,help='image minimal size at the coarser scale',default=25)
    parser.add_argument('--max_size', type=int,help='image minimal size at the coarser scale', default=250)

    #optimization hyper parameters:
    parser.add_argument('--niter', type=int, default=2000, help='number of epochs to train per scale')
    parser.add_argument('--gamma',type=float,help='scheduler gamma',default=0.1)
    parser.add_argument('--lr_g', type=float, default=0.0005, help='learning rate, default=0.0005')
    parser.add_argument('--lr_d', type=float, default=0.0005, help='learning rate, default=0.0005')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--Gsteps',type=int, help='Generator inner steps',default=3)
    parser.add_argument('--Dsteps',type=int, help='Discriminator inner steps',default=3)
    parser.add_argument('--lambda_grad',type=float, help='gradient penelty weight',default=0.1)
    parser.add_argument('--alpha',type=float, help='reconstruction loss weight',default=10)

    # New parameters:
    parser.add_argument("--noise_vectors_merge_method", type=str, default="cat",
                        help="Determine how to treat the different noise vectors (how to merge them to a single vector)"
                        )
    parser.add_argument("--exp_name", type=str, default="default_exp",
                        help="Unique experiment name"
                        )
    parser.add_argument("--gaussian_noise_z_distance", type=int, default=1,
                        help="Distance between z1 and z2 gaussians")
    parser.add_argument("--replace_background", type=bool, default=False,
                        help="Use image1 as primary image and replace image2's background")
    parser.add_argument("--weight_decay_d", type=float, default=0,
                        help="weight decay for D discriminator")
    parser.add_argument("--weight_decay_d_mask1", type=float, default=4e-3,
                        help="weight decay for D mask1 discriminator")
    parser.add_argument("--weight_decay_d_mask2", type=float, default=4e-3,
                        help="weight decay for D mask2 discriminator")
    parser.add_argument("--cyclic_lr", type=bool, default=False,
                        help="use cyclic learning rate scheduler")
    parser.add_argument("--mask_activation_fn", type=str, default="sigmoid",
                        help="activation function to apply on the Generator's masks. "
                             "options are: sigmoid, relu_sign, tanh_relu, down_up or tanh_sign")
    parser.add_argument("--enable_mask", type=bool, default=True,
                        help="enable mask as extra output of the generator")
    return parser
