import logging
from typing import Tuple

import SinGAN.functions as functions
import SinGAN.models as models
import os
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import math
import matplotlib.pyplot as plt

from SinGAN.functions import NoiseMode
from SinGAN.imresize import imresize
from SinGAN.my_functions import create_img_over_background, create_background

logger = logging.getLogger()


def get_reals(reals, opt, image_name, regular_resize=False):
    real_ = functions.read_image(opt, image_name)
    real = imresize(real_,opt.scale1,opt)
    reals = functions.creat_reals_pyramid(real,reals,opt, regular_resize)
    return reals


def train(opt,Gs,Zs,reals1, reals2,NoiseAmp):
    logger.info("Starting to train...")

    reals1 = get_reals(reals1, opt, opt.input_name1)
    reals2 = get_reals(reals2, opt, opt.input_name2)
    in_s1 = 0
    in_s2 = 0
    scale_num = 0
    nfc_prev = 0

    while scale_num<opt.stop_scale+1:
        opt.nfc = min(opt.nfc_init * pow(2, math.floor(scale_num / 4)), 128)
        opt.min_nfc = min(opt.min_nfc_init * pow(2, math.floor(scale_num / 4)), 128)

        opt.out_ = functions.generate_dir2save(opt)
        opt.outf = '%s/%d' % (opt.out_,scale_num)
        try:
            os.makedirs(opt.outf)
        except OSError:
                pass

        #plt.imsave('%s/in.png' %  (opt.out_), functions.convert_image_np(real), vmin=0, vmax=1)
        #plt.imsave('%s/original.png' %  (opt.out_), functions.convert_image_np(real_), vmin=0, vmax=1)
        plt.imsave('%s/real_scale1.png' %  (opt.outf), functions.convert_image_np(reals1[scale_num]), vmin=0, vmax=1)
        plt.imsave('%s/real_scale2.png' % (opt.outf), functions.convert_image_np(reals2[scale_num]), vmin=0, vmax=1)

        D_curr, D_mask1_curr, D_mask2_curr, G_curr = init_models(opt, reals1[len(Gs)].shape)
        if (nfc_prev==opt.nfc):
            G_curr.load_state_dict(torch.load('%s/%d/netG.pth' % (opt.out_,scale_num-1)))
            D_curr.load_state_dict(torch.load('%s/%d/netD.pth' % (opt.out_,scale_num-1)))
            D_mask1_curr.load_state_dict(torch.load('%s/%d/netD_mask1.pth' % (opt.out_, scale_num - 1)))
            D_mask2_curr.load_state_dict(torch.load('%s/%d/netD_mask2.pth' % (opt.out_, scale_num - 1)))

        logger.info(f"Starting to train scale {scale_num}")
        z_curr_tuple, in_s_tuple, G_curr = train_single_scale(D_curr, D_mask1_curr, D_mask2_curr,G_curr,reals1, reals2, Gs,Zs,in_s1, in_s2,NoiseAmp,opt,
                                                              )
        in_s1, in_s2 = in_s_tuple
        logger.info(f"Done training scale {scale_num}")

        G_curr = functions.reset_grads(G_curr,False)
        G_curr.eval()
        D_curr = functions.reset_grads(D_curr,False)
        D_curr.eval()
        D_mask1_curr = functions.reset_grads(D_mask1_curr,False)
        D_mask1_curr.eval()
        D_mask2_curr = functions.reset_grads(D_mask2_curr,False)
        D_mask2_curr.eval()

        Gs.append(G_curr)
        Zs.append(z_curr_tuple)
        NoiseAmp.append((opt.noise_amp1, opt.noise_amp2))

        torch.save(Zs, '%s/Zs.pth' % (opt.out_))
        torch.save(Gs, '%s/Gs.pth' % (opt.out_))
        torch.save(reals1, '%s/reals1.pth' % (opt.out_))
        torch.save(reals2, '%s/reals2.pth' % (opt.out_))
        torch.save(NoiseAmp, '%s/NoiseAmp.pth' % (opt.out_))

        scale_num+=1
        nfc_prev = opt.nfc
        del D_curr, D_mask1_curr, D_mask2_curr,G_curr
    return


def train_single_scale(netD, netD_mask1, netD_mask2,netG,reals1, reals2, Gs,Zs,in_s1, in_s2,NoiseAmp,opt):

    real1 = reals1[len(Gs)]
    real2 = reals2[len(Gs)]

    if opt.replace_background:
        background_real1 = create_background(functions.convert_image_np(real1))
        real2 = create_img_over_background(functions.convert_image_np(real2), background_real1)

        plt.imsave('%s/background_real_scale1.png' % (opt.outf), background_real1, vmin=0, vmax=1)
        plt.imsave('%s/real_scale2_new.png' % (opt.outf), real2, vmin=0, vmax=1)

        real2 = functions.np2torch(real2, opt)

    # assumption: the images are the same size
    opt.nzx = real1.shape[2]#+(opt.ker_size-1)*(opt.num_layer)
    opt.nzy = real1.shape[3]#+(opt.ker_size-1)*(opt.num_layer)
    opt.receptive_field = opt.ker_size + ((opt.ker_size-1)*(opt.num_layer-1))*opt.stride
    pad_noise = int(((opt.ker_size - 1) * opt.num_layer) / 2)
    pad_image = int(((opt.ker_size - 1) * opt.num_layer) / 2)

    m_noise = nn.ZeroPad2d(int(pad_noise))
    m_image = nn.ZeroPad2d(int(pad_image))

    alpha = opt.alpha

    fixed_noise = functions.generate_noise([opt.nc_z,opt.nzx,opt.nzy],device=opt.device)
    z_opt = torch.full(fixed_noise.shape, 0, device=opt.device)
    z_opt = m_noise(z_opt)

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay_d)
    optimizerD_masked1 = optim.Adam(netD_mask1.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay_d_mask1)
    optimizerD_masked2 = optim.Adam(netD_mask2.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay_d_mask2)
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))

    if opt.cyclic_lr:
        schedulerD = torch.optim.lr_scheduler.CyclicLR(optimizer=optimizerD, base_lr=opt.lr_d*opt.gamma, max_lr=opt.lr_d,
                                                       step_size_up=opt.niter/10, mode="triangular2",
                                                       cycle_momentum=False)
        schedulerD_masked1 = torch.optim.lr_scheduler.CyclicLR(optimizer=optimizerD_masked1, base_lr=opt.lr_d*opt.gamma, max_lr=opt.lr_d,
                                                               step_size_up=opt.niter/10, mode="triangular2",
                                                               cycle_momentum=False)
        schedulerD__masked2 = torch.optim.lr_scheduler.CyclicLR(optimizer=optimizerD_masked2, base_lr=opt.lr_d*opt.gamma, max_lr=opt.lr_d,
                                                                step_size_up=opt.niter/10, mode="triangular2",
                                                                cycle_momentum=False)
        schedulerG = torch.optim.lr_scheduler.CyclicLR(optimizer=optimizerG, base_lr=opt.lr_d*opt.gamma, max_lr=opt.lr_d,
                                                       step_size_up=opt.niter/10, mode="triangular2",
                                                       cycle_momentum=False)
    else:
        schedulerD = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerD,milestones=[1600],gamma=opt.gamma)
        schedulerD_masked1 = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerD_masked1, milestones=[1600], gamma=opt.gamma)
        schedulerD__masked2 = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerD_masked2, milestones=[1600], gamma=opt.gamma)
        schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerG,milestones=[1600],gamma=opt.gamma)

    discriminators = [netD, netD_mask1, netD_mask2]
    discriminators_optimizers = [optimizerD, optimizerD_masked1, optimizerD_masked2]
    discriminators_schedulers = [schedulerD, schedulerD_masked1, schedulerD__masked2]

    err_D_img1_2plot = []
    err_D_img2_2plot = []
    err_D_mask1_2plot = []
    err_D_mask2_2plot = []
    errG_total_loss_2plot = []
    errG_total_loss1_2plot = []
    errG_total_loss2_2plot = []
    errG_fake1_2plot = []
    errG_fake2_2plot = []
    D1_real2plot = []
    D2_real2plot = []
    D1_fake2plot = []
    D2_fake2plot = []
    mask_loss2plot = []
    reconstruction_loss1_2plot = []
    reconstruction_loss2_2plot = []

    for epoch in range(opt.niter):
        """
        We want to ensure that there exists a specific set of input noise maps, which generates the original image x.
        We specifically choose {z*, 0, 0, ..., 0}, where z* is some fixed noise map.
        In the first scale, we create that z* (aka z_opt). On other scales, z_opt is just zeros (initialized above)
        """
        is_first_scale = len(Gs) == 0

        noise1_, z_opt1 = _create_noise_for_iteration(is_first_scale, m_noise, opt, z_opt, NoiseMode.Z1)
        noise2_, z_opt2 = _create_noise_for_iteration(is_first_scale, m_noise, opt, z_opt, NoiseMode.Z2)

        ############################
        # (1) Update D networks:
        # - maximize D1(x1)
        # - maximize D2(x2)
        # - maximize D1(G(z1||0))
        # - maximize D2(G(0||z2))
        # - maximize D#(G(z1||z2)): only if mixed_imgs_training is true
        ###########################
        for j in range(opt.Dsteps):
            # train with real
            for discriminator in discriminators:
                discriminator.zero_grad()

            errD_real1, D_x1 = discriminator_train_with_real(netD, opt, real1)
            errD_real2, D_x2 = discriminator_train_with_real(netD, opt, real2)

            # single discriminator for each image
            errD_mask1_real1, _ = discriminator_train_with_real(netD_mask1, opt, real1)
            errD_mask2_real2, _ = discriminator_train_with_real(netD_mask2, opt, real2)

            # train with fake
            in_s1, noise1, prev1, new_z_prev1 = _prepare_discriminator_train_with_fake_input(Gs, NoiseAmp, Zs, epoch,
                                                                                             in_s1, is_first_scale, j,
                                                                                             m_image, m_noise, noise1_,
                                                                                             opt, real1, reals1,
                                                                                             NoiseMode.Z1)
            in_s2, noise2, prev2, new_z_prev2 = _prepare_discriminator_train_with_fake_input(Gs, NoiseAmp, Zs, epoch,
                                                                                             in_s2, is_first_scale, j,
                                                                                             m_image, m_noise, noise2_,
                                                                                             opt, real2, reals2,
                                                                                             NoiseMode.Z2)
            if new_z_prev1 is not None:
                z_prev1 = new_z_prev1
            if new_z_prev2 is not None:
                z_prev2 = new_z_prev2

            # Z1 only:
            mixed_noise1 = functions.merge_noise_vectors(noise1, torch.zeros(noise1.shape, device=opt.device),
                                               opt.noise_vectors_merge_method)
            D_G_z_1, errD_fake1, gradient_penalty1, fake1, _, _ = _train_discriminator_with_fake(netD, netG, mixed_noise1, opt, prev1, real1)

            # Z2 only:
            mixed_noise2 = functions.merge_noise_vectors(torch.zeros(noise2.shape, device=opt.device), noise2,
                                               opt.noise_vectors_merge_method)
            D_G_z_2, errD_fake2, gradient_penalty2, fake2, _, _ = _train_discriminator_with_fake(netD, netG, mixed_noise2, opt, prev2, real2)

            _, errD_mask1_fake1, _, _, fake1_mask1, fake1_mask2 = _train_discriminator_with_fake(netD_mask1, netG, mixed_noise1, opt, prev1, real1)
            _, errD_mask2_fake2, _, _, fake2_mask1, fake2_mask2 = _train_discriminator_with_fake(netD_mask2, netG, mixed_noise2, opt, prev2, real2)

            errD_image1 = errD_real1 + errD_fake1 + gradient_penalty1
            errD_image2 = errD_real2 + errD_fake2 + gradient_penalty2
            errD_mask1 = errD_mask1_real1 + errD_mask1_fake1
            errD_mask2 = errD_mask2_real2 + errD_mask2_fake2

            for discriminator_optimizer in discriminators_optimizers:
                discriminator_optimizer.step()

        err_D_img1_2plot.append(errD_image1.detach())
        err_D_img2_2plot.append(errD_image2.detach())
        err_D_mask1_2plot.append(errD_mask1.detach())
        err_D_mask2_2plot.append(errD_mask2.detach())

        ############################
        # (2) Update G network:
        # - maximize D1(G(z1||0))
        # - maximize D2(G(0||z2))
        # - maximize D#(G(z1||z2)): only if mixed_imgs_training is true
        ###########################

        for j in range(opt.Gsteps):
            netG.zero_grad()
            errG_fake1, D_fake1_map = _generator_train_with_fake(fake1, netD)
            errG_fake2, D_fake2_map = _generator_train_with_fake(fake2, netD)
            rec_loss1, Z_opt1 = _reconstruction_loss(alpha, netG, opt, z_opt1, z_prev1, real1, NoiseMode.Z1, opt.noise_amp1)
            rec_loss2, Z_opt2 = _reconstruction_loss(alpha, netG, opt, z_opt2, z_prev2, real2, NoiseMode.Z2, opt.noise_amp2)

            mask_loss_fake1_mask1, D_mask1_fake1_mask1_map = _generator_train_with_fake(fake1_mask1, netD_mask1)
            # mask_loss_fake2_mask1, D_mask1_fake2_mask1_map = _generator_train_with_fake(fake2_mask1, netD_mask1)
            # mask_loss_fake1_mask2, D_mask2_fake1_mask2_map = _generator_train_with_fake(fake1_mask2, netD_mask2)
            mask_loss_fake2_mask2, D_mask2_fake2_mask2_map = _generator_train_with_fake(fake2_mask2, netD_mask2)
            # mask_loss = mask_loss_fake1_mask1 + mask_loss_fake2_mask1 + mask_loss_fake1_mask2 + mask_loss_fake2_mask2
            mask_loss = mask_loss_fake1_mask1 + mask_loss_fake2_mask2

            optimizerG.step()

        errG_total_loss1_2plot.append(errG_fake1.detach()+rec_loss1)
        errG_total_loss2_2plot.append(errG_fake2.detach()+rec_loss2)
        errG_fake1_2plot.append(errG_fake1.detach())
        errG_fake2_2plot.append(errG_fake2.detach())
        G_total_loss = errG_fake1.detach()+rec_loss1 + errG_fake2.detach()+rec_loss2
        errG_total_loss_2plot.append(G_total_loss)
        D1_real2plot.append(D_x1)
        D2_real2plot.append(D_x2)
        D1_fake2plot.append(D_G_z_1)
        D2_fake2plot.append(D_G_z_2)
        reconstruction_loss1_2plot.append(rec_loss1)
        reconstruction_loss2_2plot.append(rec_loss2)
        mask_loss2plot.append(mask_loss.detach())

        if epoch % 25 == 0 or epoch == (opt.niter-1):
            logger.info('scale %d:[%d/%d]' % (len(Gs), epoch, opt.niter))

        if epoch % 500 == 0 or epoch == (opt.niter-1):
            plt.imsave('%s/fake_sample1.png' %  (opt.outf), functions.convert_image_np(fake1.detach()), vmin=0, vmax=1)
            plt.imsave('%s/fake_sample2.png' % (opt.outf), functions.convert_image_np(fake2.detach()), vmin=0, vmax=1)
            plt.imsave('%s/G(z_opt1).png'    % (opt.outf),
                       functions.convert_image_np(netG(Z_opt1.detach(), z_prev1)[0].detach()), vmin=0, vmax=1)
            plt.imsave('%s/G(z_opt2).png' % (opt.outf),
                       functions.convert_image_np(netG(Z_opt2.detach(), z_prev2)[0].detach()), vmin=0, vmax=1)

            torch.save((mixed_noise1, prev1), '%s/fake1_noise_source.pth' % (opt.outf))
            torch.save((mixed_noise2, prev2), '%s/fake2_noise_source.pth' % (opt.outf))
            torch.save((Z_opt1, z_prev1), '%s/G(z_opt1)_noise_source.pth' % (opt.outf))
            torch.save((Z_opt2, z_prev2), '%s/G(z_opt2)_noise_source.pth' % (opt.outf))
            torch.save(z_opt1, '%s/z_opt1.pth' % (opt.outf))
            torch.save(z_opt2, '%s/z_opt2.pth' % (opt.outf))

            if epoch == (opt.niter-1):
                _imsave_discriminator_map(D_fake1_map, "D_fake1_map", opt)
                _imsave_discriminator_map(D_fake2_map, "D_fake2_map", opt)
                _imsave_discriminator_map(D_mask1_fake1_mask1_map, "D_mask1_fake1_mask1_map", opt)
                # _imsave_discriminator_map(D_mask1_fake2_mask1_map, "D_mask1_fake2_mask1_map", opt)
                # _imsave_discriminator_map(D_mask2_fake1_mask2_map, "D_mask2_fake1_mask2_map", opt)
                _imsave_discriminator_map(D_mask2_fake2_mask2_map, "D_mask2_fake2_mask2_map", opt)

        for discriminator_scheduler in discriminators_schedulers:
            discriminator_scheduler.step()
        schedulerG.step()

    functions.save_networks(netG,netD, netD_mask1, netD_mask2,z_opt1, z_opt2,opt)

    functions.plot_learning_curves("G_loss", opt.niter, [errG_total_loss_2plot,
                                                         errG_total_loss1_2plot, errG_total_loss2_2plot,
                                                         errG_fake1_2plot, errG_fake2_2plot,
                                                         reconstruction_loss1_2plot,
                                                         reconstruction_loss2_2plot,
                                                         mask_loss2plot],
                                   ["G_total_loss", "G_total_loss1", "G_total_loss2",
                                    "G_fake1_loss", "G_fake2_loss",
                                    "G_recon_loss_1", "G_recon_loss_2", "mask_loss"], opt.outf)
    d_plots = [err_D_img1_2plot, err_D_img2_2plot]
    d_labels = ["D1_total_loss", "D2_total_loss"]

    functions.plot_learning_curves("D_loss", opt.niter, d_plots, d_labels, opt.outf)
    functions.plot_learning_curves("G_vs_D_loss", opt.niter,
                                   [errG_total_loss_2plot, errG_total_loss1_2plot, errG_total_loss2_2plot,
                                    err_D_img1_2plot, err_D_img2_2plot],
                                   ["G_total_loss", "G_total_loss1", "G_total_loss2", "D1_total_loss", "D2_total_loss"],
                                   opt.outf)
    functions.plot_learning_curves("D_mask_loss", opt.niter, [err_D_mask1_2plot, err_D_mask2_2plot], ["D_mask1", "D_mask2"], opt.outf)
    return (z_opt1, z_opt2), (in_s1, in_s2), netG


def _imsave_discriminator_map(tensor, img_name, opt):
    plt.imsave(f"{opt.outf}/{img_name}.png", functions.move_to_cpu(tensor.squeeze()), cmap='gray', vmin=0, vmax=1)


def _generator_train_with_fake(fake, netD):
    output = netD(fake)
    errG = -output.mean()
    errG.backward(retain_graph=True)
    return errG, output.detach()


def _reconstruction_loss(alpha, netG, opt, z_opt, z_prev, real, noise_mode: NoiseMode, noise_amp: float):
    if alpha != 0:
        # reconstruction loss calculation
        loss = nn.MSELoss()
        Z_opt = noise_amp * z_opt + z_prev
        z_zero = torch.zeros(Z_opt.shape, device=opt.device)

        if noise_mode.Z1:
            Z_opt = functions.merge_noise_vectors(Z_opt, z_zero, opt.noise_vectors_merge_method)
        elif noise_mode.Z2:
            Z_opt = functions.merge_noise_vectors(z_zero, Z_opt, opt.noise_vectors_merge_method)
        else:
            pass

        fake, _, _ = netG(Z_opt.detach(), z_prev)

        rec_loss = alpha * loss(fake, real)
        rec_loss.backward(retain_graph=True)
        rec_loss = rec_loss.detach()
    else:
        Z_opt = z_opt
        rec_loss = 0
    return rec_loss, Z_opt


def _train_discriminator_with_fake(netD, netG, noise, opt, prev, real):
    fake, mask1, mask2 = netG(noise.detach(), prev)
    output = netD(fake.detach())
    errD_fake = output.mean()
    errD_fake.backward(retain_graph=True)
    D_G_z = output.mean().item()
    gradient_penalty = functions.calc_gradient_penalty(netD, real, fake, opt.lambda_grad, opt.device)
    gradient_penalty.backward()
    return D_G_z, errD_fake, gradient_penalty, fake, mask1, mask2


def _prepare_discriminator_train_with_fake_input(Gs, NoiseAmp, Zs, epoch, in_s, is_first_scale, j, m_image, m_noise,
                                                 noise_, opt, real, reals, noise_mode: NoiseMode):
    if (j == 0) & (epoch == 0):
        if is_first_scale:
            prev = torch.full([1, opt.nc_z, opt.nzx, opt.nzy], 0, device=opt.device)
            in_s = prev
            prev = m_image(prev)
            z_prev = torch.full([1, opt.nc_z, opt.nzx, opt.nzy], 0, device=opt.device)
            z_prev = m_noise(z_prev)
            if noise_mode in [NoiseMode.Z1, NoiseMode.BACKGROUND]:
                opt.noise_amp1 = 1
            elif noise_mode == NoiseMode.Z2:
                opt.noise_amp2 = 1
            else:
                pass  # todo: implement
        else:
            prev = draw_concat(Gs, Zs, reals, NoiseAmp, in_s, 'rand', m_noise, m_image, opt, noise_mode)
            prev = m_image(prev)
            z_prev = draw_concat(Gs, Zs, reals, NoiseAmp, in_s, 'rec', m_noise, m_image, opt, noise_mode)
            criterion = nn.MSELoss()
            RMSE = torch.sqrt(criterion(real, z_prev))
            if noise_mode == NoiseMode.Z1:
                opt.noise_amp1 = opt.noise_amp_init * RMSE
            elif noise_mode == NoiseMode.Z2:
                opt.noise_amp2 = opt.noise_amp_init * RMSE
            else:
                pass  # todo: implement
            z_prev = m_image(z_prev)
    else:
        prev = draw_concat(Gs, Zs, reals, NoiseAmp, in_s, 'rand', m_noise, m_image, opt, noise_mode)
        prev = m_image(prev)
        z_prev = None
    if is_first_scale:
        noise = noise_
    else:
        if noise_mode in [NoiseMode.Z1, NoiseMode.BACKGROUND]:
            noise_amp = opt.noise_amp1
        elif noise_mode == NoiseMode.Z2:
            noise_amp = opt.noise_amp2
        else:
            noise_amp = 1  # todo: implement
        noise = noise_amp * noise_ + prev
    return in_s, noise, prev, z_prev


def _create_noise_for_iteration(is_first_scale, m_noise, opt, default_z_opt, noise_mode):
    if is_first_scale:
        z_opt = functions.generate_noise([1, opt.nzx, opt.nzy], device=opt.device, noise_mode=noise_mode,
                                         gaussian_noise_z_distance=opt.gaussian_noise_z_distance)
        z_opt = m_noise(z_opt.expand(1, 3, opt.nzx, opt.nzy))
        noise_ = functions.generate_noise([1, opt.nzx, opt.nzy], device=opt.device, noise_mode=noise_mode,
                                          gaussian_noise_z_distance=opt.gaussian_noise_z_distance)
        noise_ = m_noise(noise_.expand(1, 3, opt.nzx, opt.nzy))
    else:
        z_opt = default_z_opt
        noise_ = functions.generate_noise([opt.nc_z, opt.nzx, opt.nzy], device=opt.device, noise_mode=noise_mode,
                                          gaussian_noise_z_distance=opt.gaussian_noise_z_distance)
        noise_ = m_noise(noise_)
    return noise_, z_opt


def discriminator_train_with_real(netD, opt, real):
    output = netD(real).to(opt.device)
    # D_real_map = output.detach()
    errD_real = -output.mean()  # -a
    errD_real.backward(retain_graph=True)
    D_x = -errD_real.item()
    return errD_real, D_x


def draw_concat(Gs, Zs, reals, NoiseAmp, in_s, mode, m_noise, m_image, opt, noise_mode: NoiseMode):
    G_z = in_s
    if len(Gs) > 0:
        if mode == 'rand':
            count = 0
            pad_noise = int(((opt.ker_size-1)*opt.num_layer)/2)
            for G,(Z_opt1, Z_opt2),real_curr,real_next,(noise_amp1, noise_amp2) in zip(Gs,Zs,reals,reals[1:],NoiseAmp):

                noise_amp = None
                if noise_mode == NoiseMode.Z1:
                    z1 = _create_noise_for_draw_concat(opt, count, pad_noise, m_noise, Z_opt1, noise_mode)
                    z2 = torch.zeros(z1.shape, device=opt.device)
                    noise_amp = noise_amp1
                elif noise_mode == NoiseMode.Z2:
                    z2 = _create_noise_for_draw_concat(opt, count, pad_noise, m_noise, Z_opt2, noise_mode)
                    z1 = torch.zeros(z2.shape, device=opt.device)
                    noise_amp = noise_amp2
                elif noise_mode == NoiseMode.MIXED:
                    z1 = _create_noise_for_draw_concat(opt, count, pad_noise, m_noise, Z_opt1, noise_mode)
                    z2 = _create_noise_for_draw_concat(opt, count, pad_noise, m_noise, Z_opt2, noise_mode)
                elif noise_mode == NoiseMode.BACKGROUND:
                    z = _create_noise_for_draw_concat(opt, count, pad_noise, m_noise, Z_opt_bg, noise_mode)
                else:
                    raise NotImplementedError

                if noise_mode != NoiseMode.BACKGROUND:
                    z = functions.merge_noise_vectors(z1, z2, opt.noise_vectors_merge_method)

                G_z = G_z[:,:,0:real_curr.shape[2],0:real_curr.shape[3]]
                G_z = m_image(G_z)
                z_in = noise_amp*z+G_z
                G_z = G(z_in.detach(), G_z)[0]
                G_z = imresize(G_z,1/opt.scale_factor,opt)
                G_z = G_z[:,:,0:real_next.shape[2],0:real_next.shape[3]]
                count += 1
        if mode == 'rec':
            count = 0
            for G,(Z_opt1, Z_opt2),real_curr,real_next,(noise_amp1, noise_amp2) in zip(Gs,Zs,reals,reals[1:],NoiseAmp):
                G_z = G_z[:, :, 0:real_curr.shape[2], 0:real_curr.shape[3]]
                G_z = m_image(G_z)

                noise_amp = None
                if noise_mode == NoiseMode.Z1:
                    Z_opt2_zeros = torch.zeros(Z_opt2.shape, device=opt.device)
                    Z_opt = functions.merge_noise_vectors(Z_opt1, Z_opt2_zeros, opt.noise_vectors_merge_method)
                    noise_amp = noise_amp1
                elif noise_mode == NoiseMode.Z2:
                    Z_opt1_zeros = torch.zeros(Z_opt1.shape, device=opt.device)
                    Z_opt = functions.merge_noise_vectors(Z_opt1_zeros, Z_opt2, opt.noise_vectors_merge_method)
                    noise_amp = noise_amp2
                elif noise_mode == NoiseMode.MIXED:
                    Z_opt = functions.merge_noise_vectors(Z_opt1, Z_opt2, opt.noise_vectors_merge_method)
                else:
                    raise NotImplementedError

                z_in = noise_amp*Z_opt+G_z
                G_z = G(z_in.detach(),G_z)[0]
                G_z = imresize(G_z,1/opt.scale_factor,opt)
                G_z = G_z[:,:,0:real_next.shape[2],0:real_next.shape[3]]
                #if count != (len(Gs)-1):
                #    G_z = m_image(G_z)
                count += 1
    return G_z


def _create_noise_for_draw_concat(opt, count, pad_noise, m_noise, Z_opt, noise_mode):
    if count == 0:
        z = functions.generate_noise([1, Z_opt.shape[2] - 2 * pad_noise, Z_opt.shape[3] - 2 * pad_noise],
                                     device=opt.device, noise_mode=noise_mode,
                                     gaussian_noise_z_distance=opt.gaussian_noise_z_distance)
        z = z.expand(1, 3, z.shape[2], z.shape[3])
    else:
        z = functions.generate_noise([opt.nc_z, Z_opt.shape[2] - 2 * pad_noise, Z_opt.shape[3] - 2 * pad_noise],
                                     device=opt.device, noise_mode=noise_mode,
                                     gaussian_noise_z_distance=opt.gaussian_noise_z_distance)
    z = m_noise(z)
    return z


def init_models(opt, img_shape) -> "Tuple[models.WDiscriminator, models.WDiscriminator, models.WDiscriminator, models.GeneratorConcatSkip2CleanAdd]":

    #generator initialization:
    netG = models.GeneratorConcatSkip2CleanAdd(opt, img_shape).to(opt.device)
    netG.apply(models.weights_init)
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))
    logger.info(netG)

    # general discriminator initialization for both images:
    netD = models.WDiscriminator(opt).to(opt.device)
    netD.apply(models.weights_init)
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    logger.info(netD)

    # discriminator initialization for identifying the mask of the first image:
    netD_mask1 = models.WDiscriminator(opt).to(opt.device)
    netD_mask1.apply(models.weights_init)
    if opt.netD_mask1 != '':
        netD_mask1.load_state_dict(torch.load(opt.netD_mask1))
    logger.info(netD_mask1)

    # discriminator initialization for identifying the mask of the second image:
    netD_mask2 = models.WDiscriminator(opt).to(opt.device)
    netD_mask2.apply(models.weights_init)
    if opt.netD_mask2 != '':
        netD_mask2.load_state_dict(torch.load(opt.netD_mask2))
    logger.info(netD_mask2)

    return netD, netD_mask1, netD_mask2, netG
