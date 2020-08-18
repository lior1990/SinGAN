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

logger = logging.getLogger()


def get_reals(reals, opt, image_name, regular_resize=False):
    real_ = functions.read_image(opt, image_name)
    real = imresize(real_,opt.scale1,opt)
    reals = functions.creat_reals_pyramid(real,reals,opt, regular_resize)
    return reals


def train(opt,Gs,Zs,reals1, reals2,NoiseAmp):
    logger.info("Starting to train...")

    masked_reals2 = []
    background_reals1 = []
    reals1 = get_reals(reals1, opt, opt.input_name1)
    reals2 = get_reals(reals2, opt, opt.input_name2)
    masked_reals2 = get_reals(masked_reals2, opt, f"masked_{opt.input_name2}", regular_resize=True)
    background_reals1 = get_reals(background_reals1, opt, f"background_{opt.input_name1}", regular_resize=True)
    in_s1 = 0
    in_s2 = 0
    in_s_mixed = 0
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
        plt.imsave('%s/masked_real_scale2.png' % (opt.outf), functions.convert_image_np(masked_reals2[scale_num]), vmin=0, vmax=1)

        D1_curr, D2_curr, D_mixed_curr, G_curr = init_models(opt)
        if (nfc_prev==opt.nfc):
            G_curr.load_state_dict(torch.load('%s/%d/netG.pth' % (opt.out_,scale_num-1)))
            D1_curr.load_state_dict(torch.load('%s/%d/netD1.pth' % (opt.out_,scale_num-1)))
            D2_curr.load_state_dict(torch.load('%s/%d/netD2.pth' % (opt.out_, scale_num - 1)))
            D_mixed_curr.load_state_dict(torch.load('%s/%d/netD_mixed.pth' % (opt.out_, scale_num - 1)))

        mixed_imgs_training = bool(scale_num >= opt.stop_scale/2) if opt.mixed_imgs_training else False
        logger.info(f"Starting to train scale {scale_num}. Mixed imgs status: {mixed_imgs_training}")
        z_curr_tuple, in_s_tuple, G_curr = train_single_scale(D1_curr, D2_curr, D_mixed_curr,G_curr,reals1, reals2, background_reals1, masked_reals2,Gs,Zs,in_s1, in_s2, in_s_mixed,NoiseAmp,opt,
                                                              mixed_imgs_training=mixed_imgs_training)
        in_s1, in_s2, in_s_mixed, in_s_bg = in_s_tuple
        logger.info(f"Done training scale {scale_num}")

        G_curr = functions.reset_grads(G_curr,False)
        G_curr.eval()
        D1_curr = functions.reset_grads(D1_curr,False)
        D1_curr.eval()
        D2_curr = functions.reset_grads(D2_curr,False)
        D2_curr.eval()
        D_mixed_curr = functions.reset_grads(D_mixed_curr,False)
        D_mixed_curr.eval()

        Gs.append(G_curr)
        Zs.append(z_curr_tuple)
        NoiseAmp.append(opt.noise_amp)

        torch.save(Zs, '%s/Zs.pth' % (opt.out_))
        torch.save(Gs, '%s/Gs.pth' % (opt.out_))
        torch.save(reals1, '%s/reals1.pth' % (opt.out_))
        torch.save(reals2, '%s/reals2.pth' % (opt.out_))
        torch.save(NoiseAmp, '%s/NoiseAmp.pth' % (opt.out_))

        scale_num+=1
        nfc_prev = opt.nfc
        del D1_curr, D2_curr, D_mixed_curr,G_curr
    return



def train_single_scale(netD1, netD2, netD_mixed,netG,reals1, reals2, background_reals1, masked_reals2,Gs,Zs,in_s1, in_s2, in_s_mixed,NoiseAmp,opt, mixed_imgs_training=False):

    real1 = reals1[len(Gs)]
    real2 = reals2[len(Gs)]
    masked_real2 = masked_reals2[len(Gs)]
    background_real1 = background_reals1[len(Gs)]

    assert (masked_real2[0][0] == masked_real2[0][1]).all().all() and (masked_real2[0][1] == masked_real2[0][2]).all().all()

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
    optimizerD1 = optim.Adam(netD1.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
    optimizerD2 = optim.Adam(netD2.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
    optimizerD_mixed = optim.Adam(netD_mixed.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))
    schedulerD1 = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerD1,milestones=[1600],gamma=opt.gamma)
    schedulerD2 = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerD2, milestones=[1600], gamma=opt.gamma)
    schedulerD_mixed = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerD_mixed, milestones=[1600], gamma=opt.gamma)
    schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerG,milestones=[1600],gamma=opt.gamma)

    err_D1_2plot = []
    err_D2_2plot = []
    errG_total_loss_2plot = []
    errG_total_loss1_2plot = []
    errG_total_loss2_2plot = []
    errG_fake1_2plot = []
    errG_fake2_2plot = []
    errG_mixed_fake_2plot = []
    D1_real2plot = []
    D2_real2plot = []
    D1_fake2plot = []
    D2_fake2plot = []
    errD_mixed_fake2plot = []
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
        noise_mixed_, z_opt_mixed = _create_noise_for_iteration(is_first_scale, m_noise, opt, z_opt, NoiseMode.MIXED)
        noise_bg_, z_opt_bg = _create_noise_for_iteration(is_first_scale, m_noise, opt, z_opt, NoiseMode.BACKGROUND)

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
            netD1.zero_grad()
            netD2.zero_grad()

            errD1_real, D1_x1 = discriminator_train_with_real(netD1, opt, real1)
            errD2_real, D2_x2 = discriminator_train_with_real(netD2, opt, real2)
            # errD2_real, D2_x2 = discriminator_train_with_real(netD2, opt, masked_real2)

            if mixed_imgs_training:
                # train mixed on both
                errD_mixed_real1, D_mixed_x1 = discriminator_train_with_real(netD_mixed, opt, real1)
                errD_mixed_real2, D_mixed_x2 = discriminator_train_with_real(netD_mixed, opt, real2)
                _, _ = discriminator_train_with_real(netD_mixed, opt, masked_real2)
            _, _ = discriminator_train_with_real(netD_mixed, opt, background_real1)


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
            in_s_mixed, noise_mixed, prev_mixed, new_z_prev_mixed = _prepare_discriminator_train_with_fake_input(Gs, NoiseAmp, Zs, epoch,
                                                                                             in_s_mixed, is_first_scale, j,
                                                                                             m_image, m_noise, noise_mixed_,
                                                                                             opt, real2, reals2,
                                                                                             NoiseMode.MIXED)
            in_s_bg, noise_bg, prev_bg, new_z_prev_bg = _prepare_discriminator_train_with_fake_input(Gs, NoiseAmp, Zs, epoch,
                                                                                             in_s_mixed, is_first_scale, j,
                                                                                             m_image, m_noise, noise_bg_,
                                                                                             opt, background_real1, background_reals1,
                                                                                             NoiseMode.BACKGROUND)

            if new_z_prev1 is not None:
                z_prev1 = new_z_prev1
            if new_z_prev2 is not None:
                z_prev2 = new_z_prev2
            if new_z_prev_mixed is not None:
                z_prev_mixed = new_z_prev_mixed
            if new_z_prev_bg is not None:
                z_prev_bg = new_z_prev_bg

            # Z1 only:
            mixed_noise1 = functions.merge_noise_vectors(noise1, torch.zeros(noise1.shape, device=opt.device),
                                               opt.noise_vectors_merge_method)
            D1_G_z, errD1_fake, gradient_penalty1, fake1 = _train_discriminator_with_fake(netD1, netG, mixed_noise1,
                                                                                          opt, prev1, real1)

            # Z2 only:
            mixed_noise2 = functions.merge_noise_vectors(torch.zeros(noise2.shape, device=opt.device), noise2,
                                               opt.noise_vectors_merge_method)
            D2_G_z, errD2_fake, gradient_penalty2, fake2 = _train_discriminator_with_fake(netD2, netG, mixed_noise2,
                                                                                          opt, prev2, real2)

            errD1 = errD1_real + errD1_fake + gradient_penalty1
            errD2 = errD2_real + errD2_fake + gradient_penalty2

            if mixed_imgs_training:
                mixed_fake = netG(noise_mixed.detach(), prev_mixed)
                output = netD_mixed(mixed_fake.detach())
                errD_mixed_fake = output.mean()
                errD_mixed_fake.backward(retain_graph=True)
                D_mixed_G_z = errD_mixed_fake.item()

            bg_fake = netG(noise_bg.detach(), prev_bg)
            bg_output = netD_mixed(bg_fake.detach())
            errD_bg_fake = bg_output.mean()
            errD_bg_fake.backward(retain_graph=True)
            D_bg_G_z = errD_bg_fake.item()

            optimizerD1.step()
            optimizerD2.step()
            optimizerD_mixed.step()

        err_D1_2plot.append(errD1.detach())
        err_D2_2plot.append(errD2.detach())

        ############################
        # (2) Update G network:
        # - maximize D1(G(z1||0))
        # - maximize D2(G(0||z2))
        # - maximize D#(G(z1||z2)): only if mixed_imgs_training is true
        ###########################

        for j in range(opt.Gsteps):
            netG.zero_grad()
            errG1 = _generator_train_with_fake(fake1, netD1)
            errG2 = _generator_train_with_fake(fake2, netD2)
            rec_loss1, Z_opt1 = _reconstruction_loss(alpha, netG, opt, z_opt1, z_prev1, real1, NoiseMode.Z1)
            rec_loss2, Z_opt2 = _reconstruction_loss(alpha, netG, opt, z_opt2, z_prev2, real2, NoiseMode.Z2)

            if mixed_imgs_training:
                # output1 = netD1(mixed_fake)
                # output2 = netD2(mixed_fake)
                # errG_mixed = -(opt.D_img1_regularization_loss * output1 + opt.D_img2_regularization_loss * output2).mean()
                output = netD_mixed(mixed_fake)
                errG_mixed = -output.mean()
                errG_mixed.backward(retain_graph=True)

                _reconstruction_loss(alpha, netG, opt, z_opt_mixed, z_prev_mixed, masked_real2, NoiseMode.MIXED, mask=masked_real2)
                _, Z_opt_mixed = _reconstruction_loss(alpha, netG, opt, z_opt_mixed, z_prev_mixed, real1, NoiseMode.MIXED)

            bg_output = netD_mixed(bg_fake)
            errG_bg = -bg_output.mean()
            errG_bg.backward(retain_graph=True)
            _, Z_opt_bg = _reconstruction_loss(alpha, netG, opt, z_opt_bg, z_prev_bg, background_real1, NoiseMode.BACKGROUND)

            optimizerG.step()

        errG_total_loss1_2plot.append(errG1.detach()+rec_loss1)
        errG_total_loss2_2plot.append(errG2.detach()+rec_loss2)
        errG_fake1_2plot.append(errG1.detach())
        errG_fake2_2plot.append(errG1.detach())
        G_total_loss = errG1.detach()+rec_loss1 + errG2.detach()+rec_loss2
        if mixed_imgs_training:
            errG_mixed_fake_2plot.append(errG_mixed.detach())
            errD_mixed_fake2plot.append(D_mixed_G_z)
            G_total_loss += errG_mixed.detach()
        errG_total_loss_2plot.append(G_total_loss)
        D1_real2plot.append(D1_x1)
        D2_real2plot.append(D2_x2)
        D1_fake2plot.append(D1_G_z)
        D2_fake2plot.append(D2_G_z)
        reconstruction_loss1_2plot.append(rec_loss1)
        reconstruction_loss2_2plot.append(rec_loss2)

        if epoch % 25 == 0 or epoch == (opt.niter-1):
            logger.info('scale %d:[%d/%d]' % (len(Gs), epoch, opt.niter))

        if epoch % 500 == 0 or epoch == (opt.niter-1):
            plt.imsave('%s/fake_sample1.png' %  (opt.outf), functions.convert_image_np(fake1.detach()), vmin=0, vmax=1)
            plt.imsave('%s/fake_sample2.png' % (opt.outf), functions.convert_image_np(fake2.detach()), vmin=0, vmax=1)
            if mixed_imgs_training:
                plt.imsave('%s/fake_sample_mixed.png' % (opt.outf), functions.convert_image_np(mixed_fake.detach()), vmin=0, vmax=1)
                plt.imsave('%s/G(z_opt_mixed).png' % (opt.outf),
                           functions.convert_image_np(netG(Z_opt_mixed.detach(), z_prev_mixed).detach()), vmin=0, vmax=1)

            plt.imsave('%s/fake_sample_bg.png' % (opt.outf), functions.convert_image_np(bg_fake.detach()), vmin=0, vmax=1)
            plt.imsave('%s/G(z_opt_bg).png' % (opt.outf),
                           functions.convert_image_np(netG(Z_opt_bg.detach(), z_prev_bg).detach()), vmin=0, vmax=1)
            plt.imsave('%s/G(z_opt1).png'    % (opt.outf),
                       functions.convert_image_np(netG(Z_opt1.detach(), z_prev1).detach()), vmin=0, vmax=1)
            plt.imsave('%s/G(z_opt2).png' % (opt.outf),
                       functions.convert_image_np(netG(Z_opt2.detach(), z_prev2).detach()), vmin=0, vmax=1)
            #plt.imsave('%s/D_fake.png'   % (opt.outf), functions.convert_image_np(D_fake_map))
            #plt.imsave('%s/D_real.png'   % (opt.outf), functions.convert_image_np(D_real_map))
            #plt.imsave('%s/z_opt.png'    % (opt.outf), functions.convert_image_np(z_opt.detach()), vmin=0, vmax=1)
            #plt.imsave('%s/prev.png'     %  (opt.outf), functions.convert_image_np(prev), vmin=0, vmax=1)
            #plt.imsave('%s/noise.png'    %  (opt.outf), functions.convert_image_np(noise), vmin=0, vmax=1)
            #plt.imsave('%s/z_prev.png'   % (opt.outf), functions.convert_image_np(z_prev), vmin=0, vmax=1)


            torch.save(z_opt1, '%s/z_opt1.pth' % (opt.outf))
            torch.save(z_opt2, '%s/z_opt2.pth' % (opt.outf))

        schedulerD1.step()
        schedulerD2.step()
        schedulerD_mixed.step()
        schedulerG.step()

    functions.save_networks(netG,netD1, netD2, netD_mixed,z_opt1, z_opt2,opt)

    if mixed_imgs_training:
        functions.plot_learning_curves("mixed_loss", opt.niter, [errG_mixed_fake_2plot, errD_mixed_fake2plot],
                                       ["G_mixed", "D_mixed"], opt.outf)
    functions.plot_learning_curves("G_loss", opt.niter, [errG_total_loss_2plot,
                                                         errG_total_loss1_2plot, errG_total_loss2_2plot,
                                                         errG_fake1_2plot, errG_fake2_2plot,
                                                         reconstruction_loss1_2plot,
                                                         reconstruction_loss2_2plot],
                                   ["G_total_loss", "G_total_loss1", "G_total_loss2",
                                    "G_fake1_loss", "G_fake2_loss",
                                    "G_recon_loss_1", "G_recon_loss_2"], opt.outf)
    d_plots = [err_D1_2plot, err_D2_2plot]
    d_labels = ["D1_total_loss", "D2_total_loss"]
    if mixed_imgs_training:
        d_plots.append(errD_mixed_fake2plot)
        d_labels.append("D_mixed_loss")

    functions.plot_learning_curves("D_loss", opt.niter, d_plots, d_labels, opt.outf)
    functions.plot_learning_curves("G_vs_D_loss", opt.niter,
                                   [errG_total_loss_2plot, errG_total_loss1_2plot, errG_total_loss2_2plot,
                                    err_D1_2plot, err_D2_2plot],
                                   ["G_total_loss", "G_total_loss1", "G_total_loss2", "D1_total_loss", "D2_total_loss"],
                                   opt.outf)
    return (z_opt1, z_opt2, z_opt_bg), (in_s1, in_s2, in_s_mixed, in_s_bg), netG


def _generator_train_with_fake(fake, netD):
    output = netD(fake)
    errG = -output.mean()
    errG.backward(retain_graph=True)
    return errG


def _reconstruction_loss(alpha, netG, opt, z_opt, z_prev, real, noise_mode: NoiseMode, mask=None):
    if alpha != 0:
        # reconstruction loss calculation
        loss = nn.MSELoss()
        Z_opt = opt.noise_amp * z_opt + z_prev
        z_zero = torch.zeros(Z_opt.shape, device=opt.device)

        if noise_mode.Z1:
            Z_opt = functions.merge_noise_vectors(Z_opt, z_zero, opt.noise_vectors_merge_method)
        elif noise_mode.Z2:
            Z_opt = functions.merge_noise_vectors(z_zero, Z_opt, opt.noise_vectors_merge_method)
        else:
            pass

        fake = netG(Z_opt.detach(), z_prev)
        if mask is not None:
            fake = fake * mask

        rec_loss = alpha * loss(fake, real)
        rec_loss.backward(retain_graph=True)
        rec_loss = rec_loss.detach()
    else:
        Z_opt = z_opt
        rec_loss = 0
    return rec_loss, Z_opt


def _train_discriminator_with_fake(netD, netG, noise, opt, prev, real):
    fake = netG(noise.detach(), prev)
    output = netD(fake.detach())
    errD_fake = output.mean()
    errD_fake.backward(retain_graph=True)
    D_G_z = output.mean().item()
    gradient_penalty = functions.calc_gradient_penalty(netD, real, fake, opt.lambda_grad, opt.device)
    gradient_penalty.backward()
    return D_G_z, errD_fake, gradient_penalty, fake


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
            for G,(Z_opt1, Z_opt2, Z_opt_bg),real_curr,real_next,noise_amp in zip(Gs,Zs,reals,reals[1:],NoiseAmp):
                if noise_mode == NoiseMode.Z1:
                    z1 = _create_noise_for_draw_concat(opt, count, pad_noise, m_noise, Z_opt1, noise_mode)
                    z2 = torch.zeros(z1.shape, device=opt.device)
                elif noise_mode == NoiseMode.Z2:
                    z2 = _create_noise_for_draw_concat(opt, count, pad_noise, m_noise, Z_opt2, noise_mode)
                    z1 = torch.zeros(z2.shape, device=opt.device)
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
                G_z = G(z_in.detach(), G_z)
                G_z = imresize(G_z,1/opt.scale_factor,opt)
                G_z = G_z[:,:,0:real_next.shape[2],0:real_next.shape[3]]
                count += 1
        if mode == 'rec':
            count = 0
            for G,(Z_opt1, Z_opt2, Z_opt_bg),real_curr,real_next,noise_amp in zip(Gs,Zs,reals,reals[1:],NoiseAmp):
                G_z = G_z[:, :, 0:real_curr.shape[2], 0:real_curr.shape[3]]
                G_z = m_image(G_z)

                if noise_mode == NoiseMode.Z1:
                    Z_opt2_zeros = torch.zeros(Z_opt2.shape, device=opt.device)
                    Z_opt = functions.merge_noise_vectors(Z_opt1, Z_opt2_zeros, opt.noise_vectors_merge_method)
                elif noise_mode == NoiseMode.Z2:
                    Z_opt1_zeros = torch.zeros(Z_opt1.shape, device=opt.device)
                    Z_opt = functions.merge_noise_vectors(Z_opt1_zeros, Z_opt2, opt.noise_vectors_merge_method)
                elif noise_mode == NoiseMode.MIXED:
                    Z_opt = functions.merge_noise_vectors(Z_opt1, Z_opt2, opt.noise_vectors_merge_method)
                elif noise_mode == NoiseMode.BACKGROUND:
                    Z_opt = Z_opt_bg
                else:
                    raise NotImplementedError

                z_in = noise_amp*Z_opt+G_z
                G_z = G(z_in.detach(),G_z)
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


def init_models(opt) -> "Tuple[models.WDiscriminator, models.WDiscriminator, models.WDiscriminator, models.GeneratorConcatSkip2CleanAdd]":

    #generator initialization:
    netG = models.GeneratorConcatSkip2CleanAdd(opt).to(opt.device)
    netG.apply(models.weights_init)
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))
    logger.info(netG)

    # discriminator initialization for image1:
    netD1 = models.WDiscriminator(opt).to(opt.device)
    netD1.apply(models.weights_init)
    if opt.netD1 != '':
        netD1.load_state_dict(torch.load(opt.netD1))
    logger.info(netD1)

    # discriminator initialization for image2:
    netD2 = models.WDiscriminator(opt).to(opt.device)
    netD2.apply(models.weights_init)
    if opt.netD2 != '':
        netD2.load_state_dict(torch.load(opt.netD2))
    logger.info(netD2)

    # discriminator initialization for mixed image:
    netD_mixed = models.WDiscriminator(opt).to(opt.device)
    netD_mixed.apply(models.weights_init)
    if opt.netD_mixed != '':
        netD_mixed.load_state_dict(torch.load(opt.netD_mixed))
    logger.info(netD_mixed)

    return netD1, netD2, netD_mixed, netG
