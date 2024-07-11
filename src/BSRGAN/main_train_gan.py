import os.path
import math
import argparse
import time
import random
import numpy as np
from collections import OrderedDict
import logging
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist

from data.select_dataset import define_Dataset
from models.select_model import define_Model


'''
# --------------------------------------------
# training code for GAN-based model, such as ESRGAN, DPSRGAN
# --------------------------------------------
# Kai Zhang (cskaizhang@gmail.com)
# github: https://github.com/cszn/KAIR
# --------------------------------------------
# https://github.com/xinntao/BasicSR
# --------------------------------------------
'''


def main(json_path='options/train_msrresnet_gan.json'):

    '''
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', default=False)

    opt = option.parse(parser.parse_args().opt, is_train=True)
    if opt["wandb"]["project"]:
        import wandb
        wandb.login(
            key="",
            relogin=True
        )
        if opt["wandb"]["resume_id"]:
            wandb_id = opt["wandb"]["resume_id"]
            resume = 'allow'
        else:
            wandb_id = wandb.util.generate_id()
            resume = 'never'
        wb = wandb.init(project=opt["wandb"]["project"],
                        name=opt["wandb"]["name"],
                        resume=resume,
                        id=wandb_id,
                        config=opt
        )
    opt['dist'] = parser.parse_args().dist

    # ----------------------------------------
    # distributed settings
    # ----------------------------------------
    if opt['dist']:
        init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()

    if opt['rank'] == 0:
        util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    # ----------------------------------------
    # update opt
    # ----------------------------------------
    # -->-->-->-->-->-->-->-->-->-->-->-->-->-
    if opt['train']['resume_train']:
        init_iter_G, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
        init_iter_D, init_path_D = option.find_last_checkpoint(opt['path']['models'], net_type='D')
        init_iter_E, init_path_E = option.find_last_checkpoint(opt['path']['models'], net_type='E')
        opt['path']['pretrained_netG'] = init_path_G
        opt['path']['pretrained_netD'] = init_path_D
        opt['path']['pretrained_netE'] = init_path_E
        init_iter_optimizerG, init_path_optimizerG = option.find_last_checkpoint(opt['path']['models'], net_type='optimizerG')
        init_iter_optimizerD, init_path_optimizerD = option.find_last_checkpoint(opt['path']['models'], net_type='optimizerD')
        opt['path']['pretrained_optimizerG'] = init_path_optimizerG
        opt['path']['pretrained_optimizerD'] = init_path_optimizerD
        current_step = max(init_iter_G, init_iter_D, init_iter_E, init_iter_optimizerG, init_iter_optimizerD)
        start_epoch = 1
    else:
        current_step = 0
        start_epoch = 1
    # opt['path']['pretrained_netG'] = ''
    # current_step = 0
    border = opt['scale']
    loss_type = opt['train']["G_lossfn_type"].upper()
    # --<--<--<--<--<--<--<--<--<--<--<--<--<-

    # ----------------------------------------
    # save opt to  a '../option.json' file
    # ----------------------------------------
    if opt['rank'] == 0:
        option.save(opt)

    # ----------------------------------------
    # return None for missing key
    # ----------------------------------------
    opt = option.dict_to_nonedict(opt)

    # ----------------------------------------
    # configure logger
    # ----------------------------------------
    if opt['rank'] == 0:
        logger_name = 'train'
        utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
        logger = logging.getLogger(logger_name)
        logger.info(option.dict2str(opt))

    # ----------------------------------------
    # seed
    # ----------------------------------------
    # seed = 6929
    seed = opt['train']['manual_seed']
    # seed = 5582
    if seed is None:
        seed = random.randint(1, 10000)
    print('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    '''
    # ----------------------------------------
    # Step--2 (creat dataloader)
    # ----------------------------------------
    '''

    # ----------------------------------------
    # 1) create_dataset
    # 2) creat_dataloader for train and test
    # ----------------------------------------
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = define_Dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
            if opt['rank'] == 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
            if opt['dist']:
                train_sampler = DistributedSampler(train_set, shuffle=dataset_opt['dataloader_shuffle'], drop_last=True, seed=seed)
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size']//opt['num_gpu'],
                                          shuffle=False,
                                          num_workers=dataset_opt['dataloader_num_workers']//opt['num_gpu'],
                                          drop_last=True,
                                          pin_memory=True,
                                          sampler=train_sampler)
            else:
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size'],
                                          shuffle=dataset_opt['dataloader_shuffle'],
                                          drop_last=True,
                                          pin_memory=True)

        elif phase == 'test':
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    '''
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    '''

    model = define_Model(opt)

    model.init_train()
    # if opt['rank'] == 0:
    #     logger.info(model.info_network())
    #     logger.info(model.info_params())

    '''
    # ----------------------------------------
    # Step--4 (main training)
    # ----------------------------------------
    '''
    best_psnr = 0
    best_ssim = 0
    for epoch in range(start_epoch, opt['train']['epoch'] + 1):  # keep running
        # Epoch loss
        losses = dict()
        losses['epochs'] = dict()
        losses['epochs']['epoch'] = epoch
        losses['epochs'][f'{loss_type}_loss'] = 0
        if opt['train']["F_lossfn_weight"] > 0:
            losses['epochs']['Per_loss'] = 0
        losses['epochs']['GAN_loss'] = 0
        losses['epochs']['Total_G_loss'] = 0
        losses['epochs']['Total_D_loss'] = 0
        if opt['dist']:
            train_sampler.set_epoch(epoch)

        for i, train_data in enumerate(train_loader, start=1):

            current_step += 1
            # -------------------------------
            # 2) feed patch pairs
            # -------------------------------
            model.feed_data(train_data)

            # -------------------------------
            # 3) optimize parameters
            # -------------------------------
            model.optimize_parameters(current_step)
            log = model.current_log()
            losses['epochs'][f'{loss_type}_loss'] += log['G_loss']
            if opt['train']["F_lossfn_weight"] > 0:
                losses['epochs']['Per_loss'] += log['F_loss']
            losses['epochs']['GAN_loss'] += log['D_loss']
            losses['epochs']['Total_G_loss'] += log['Total_G_loss']
            losses['epochs']['Total_D_loss'] += log['Total_D_loss']
            
            # -------------------------------
            # 4) training information
            # -------------------------------
            if current_step % opt['train']['checkpoint_print'] == 0 and opt['rank'] == 0:
                logs = model.current_log()  # such as loss
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(epoch, current_step, model.current_learning_rate())
                for k, v in logs.items():  # merge log information into message
                    message += '{:s}: {:.3e} '.format(k, v)
                logger.info(message)
                if opt["wandb"]["project"]:
                    wb.log({f'step/losses/{loss_type}_loss': logs['G_loss']})
                    if opt['train']["F_lossfn_weight"] > 0:
                        wb.log({f'step/losses/Per_loss': logs['F_loss']})
                    wb.log({f'step/losses/GAN_loss': logs['D_loss']})
                    wb.log({f'step/losses/Total_G_loss': logs['Total_G_loss']})
                    wb.log({f'step/losses/Total_D_loss': logs['Total_D_loss']})
        # -----------------------
        # calculate PSNR, SSIM
        # -----------------------
        avg_psnr = 0.0
        avg_ssim = 0.0
        idx = 0

        for test_data in test_loader:
            idx += 1
            image_name_ext = os.path.basename(test_data['L_path'][0])
            img_name, ext = os.path.splitext(image_name_ext)

            img_dir = os.path.join(opt['path']['images'], img_name)
            util.mkdir(img_dir)

            model.feed_data(test_data)
            model.test()

            visuals = model.current_visuals()
            E_img = util.tensor2uint(visuals['E'])
            H_img = util.tensor2uint(visuals['H'])

            # -----------------------
            # save estimated image E
            # -----------------------
            if opt['train']['save']:
                save_img_path = os.path.join(img_dir, '{:s}_{:d}.jpg'.format(img_name, current_step))
                util.imsave(E_img, save_img_path)

            # -----------------------
            # calculate PSNR
            # -----------------------
            current_psnr = util.calculate_psnr(E_img, H_img, border=border)
            current_ssim = util.calculate_ssim(E_img, H_img, border=border)

            # logger.info('{:->4d}--> {:>10s} | {:<4.2f}dB'.format(idx, image_name_ext, current_psnr))

            avg_psnr += current_psnr
            avg_ssim += current_ssim
        avg_psnr = avg_psnr / idx
        avg_ssim = avg_ssim / idx
        logger.info('<epoch:{:3d}, iter:{:8,d}, Average PSNR : {:<.2f}dB,  Average SSIM : {:<.2f}\n'.format(epoch, current_step, avg_psnr, avg_ssim))
        
        for k in losses['epochs'].keys():
            if k != 'epoch':
                losses['epochs'][k] /= i

        # if epoch  % 10 == 0: 
        if (epoch  % 10 == 0) or (epoch == 15) or (epoch == 25):
            model.save(epoch)
        if best_psnr < avg_psnr:
            logger.info(f"Save best psnr at epoch {epoch}")
            model.save("psnr")
            best_psnr = avg_psnr
        if best_ssim < avg_ssim:
            logger.info(f"Save best ssim at epoch {epoch}")
            model.save("ssim")
            best_ssim = avg_ssim
            
        message = '[epoch:{:3d}]'.format(epoch)
        for k in losses['epochs'].keys():
            if k != 'epoch':
                loss = losses['epochs'][k]
                message += f'- [{k}: {loss:.4f}]'
        logger.info(message)

        if opt["wandb"]["project"]:
            for k in losses['epochs'].keys():
                if k != 'epoch':
                    wb.log({f'epoch/losses/{k}': losses['epochs'][k], "epoch": epoch})
            wb.log({'val/epoch/psnr': avg_psnr, "epoch": epoch})
            wb.log({'val/epoch/ssim': avg_ssim, "epoch": epoch})
    model.save("last")

if __name__ == '__main__':
    main()