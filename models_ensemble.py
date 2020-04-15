import argparse
import torch
import time
import numpy as np
import os
from classification_net import ClassifyNet, eval, ensemble_aug_eval
from utils import ConfusionMatrix, compute_calibration_measures

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    net_parser = argparse.ArgumentParser()
    parser.add_argument('--avg', default=None, help='name of the file with ensemble parameters')
    parser.add_argument('--da_n_iter', type=int, default=0, help='number of iterations for Data Augmentation ensemble')
    parser.add_argument('--calibrated', action='store_true', help='Boolean flag for applying temperature scaling')
    parser.add_argument('--dataset', default='isic2019', help='name of the dataset to use')
    parser.add_argument('--validation', action='store_true', help='Boolean flag for using validation set')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size during the training')


    opt = parser.parse_args()
    if opt.dataset == 'isic2018_test' and opt.da_n_iter != 0:
        opt.dataset = 'isic2018_test_waugm'
    if opt.dataset == 'isic2019' and opt.da_n_iter != 0:
        opt.dataset = 'isic2019_test_waugm'

    print(opt)

    net_parser.add_argument('--network', default='resnet50')
    net_parser.add_argument('--save_dir', help='directory where to save model weights')
    net_parser.add_argument('--dropout', action='store_true', help='Boolean flag for DropOut inclusion')
    net_parser.add_argument('--classes', '-c', type=int, nargs='+',
                            action='append', help='classes to train the model with')
    net_parser.add_argument('--load_epoch', type=int, default=0, help='load custom-trained models')
    net_parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    net_parser.add_argument('--batch_size', type=int, default=16, help='batch size during the training')
    net_parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    net_parser.add_argument('--loss', default='cross_entropy')
    net_parser.add_argument('--optimizer', default='SGD', choices=['SGD', 'Adam'])
    net_parser.add_argument('--scheduler', default='plateau', choices=['plateau', 'None'])
    net_parser.add_argument('--epochs', type=int, default=150, help='number of epochs to train')
    net_parser.add_argument('--size', type=int, default=512, help='size of images')
    net_parser.add_argument('--from_scratch', action='store_true',
                            help='Boolean flag for training a model from scratch')
    net_parser.add_argument('--augm_config', type=int, default=1,
                            help='configuration code for augmentation techniques choice')
    net_parser.add_argument('--cutout_pad', nargs='+', type=int, default=[0], help='cutout pad.')
    net_parser.add_argument('--cutout_holes', nargs='+', type=int, default=[0], help='number of cutout holes.')


    ens_preds = None
    counter = 0.0
    w_acc_test = 0.0
    acc_test = 0.0
    start_time = time.time()
    f = open(opt.avg, "r")
    for line in f:
        if not line.strip():
            continue
        print(line)
        counter += 1.0
        net_opt = net_parser.parse_args(line.split())
        print(net_opt)

        n = ClassifyNet(net=net_opt.network, dname=opt.dataset, dropout=net_opt.dropout,
                        classes=net_opt.classes, l_r=net_opt.learning_rate, loss=net_opt.loss,
                        optimizer=net_opt.optimizer, scheduler=net_opt.scheduler, size=net_opt.size,
                        batch_size=opt.batch_size, n_workers=net_opt.workers, pretrained=(not net_opt.from_scratch),
                        augm_config=net_opt.augm_config, save_dir=net_opt.save_dir,
                        cutout_params=[net_opt.cutout_holes, net_opt.cutout_pad], total_epochs=net_opt.epochs,
                        no_logs=True, optimize_temp_scal=opt.calibrated)

        if not net_opt.load_epoch == 0:
            n.load(net_opt.load_epoch)

        if opt.validation:
            n.test_data_loader = n.valid_data_loader
            n.calibration_variables[2] = n.calibration_variables[1]

        if opt.da_n_iter != 0:
            acc, w_acc, preds, true_lab = ensemble_aug_eval(opt.da_n_iter, n, opt.calibrated)

        else:
            acc, w_acc, calib, conf_matrix, _ = eval(n, n.test_data_loader, *n.calibration_variables[2],
                                                     opt.calibrated)
            _, preds, true_lab = calib

        acc_test += acc
        w_acc_test += w_acc
        if ens_preds is None:
            ens_preds = preds

        else:
            ens_preds += preds

    conf_matrix_test = ConfusionMatrix(n.num_classes)
    temp_ens_preds = ens_preds / counter

    check_output, res = torch.max(torch.tensor(temp_ens_preds, device='cuda'), 1)
    conf_matrix_test.update_matrix(res, torch.tensor(true_lab, device='cuda'))

    ens_acc, ens_w_acc = conf_matrix_test.get_metrics()
    ECE_test, MCE_test, BRIER_test, NNL_test = compute_calibration_measures(temp_ens_preds, true_lab,
                                                                            apply_softmax=False,
                                                                            bins=15)

    print("\n ----- FINAL PRINT ----- \n")

    print("\n|| took {:.1f} minutes \n"
          "| Mean Accuracy statistics: weighted Acc test: {:.3f} Acc test: {:.3f} \n"
          "| Ensemble Accuracy statistics: weighted Acc test: {:.3f} Acc test: {:.3f} \n"
          "| Calibration test: ECE: {:.5f} MCE: {:.5f} BRIER: {:.5f}  NNL: {:.5f}\n\n".
          format((time.time() - start_time) / 60., w_acc_test / counter, acc_test / counter, ens_w_acc, ens_acc,
                 ECE_test * 100, MCE_test * 100, BRIER_test, NNL_test))
    print(conf_matrix_test.conf_matrix)

    avgname = os.path.basename(opt.avg)
    fname = opt.dataset + "_" + os.path.splitext(avgname)[0]
    if opt.calibrated:
        fname += "_calibrated"
    if opt.da_n_iter > 0:
        fname += "_" + str(opt.da_n_iter) + "DAiter"
    if opt.validation:
        fname += "_validation"

    np.savetxt(opt.save_dir + "/output_" + fname + ".csv", temp_ens_preds, delimiter=",")
    np.save(opt.save_dir + "/output_" + fname + ".npy", temp_ens_preds)
