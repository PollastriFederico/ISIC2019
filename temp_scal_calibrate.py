import argparse
import torch

if torch.__version__ != '1.1.0':
    raise (Exception('Torch version must be 1.1.0'))
import time
from classification_net import ClassifyNet, train_temperature_scaling_decoupled, eval
from utils import compute_calibration_measures

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    net_parser = argparse.ArgumentParser()

    net_parser.add_argument('--temp_scal_learning_rate', type=float, default=0.1,
                            help='temperature scaling learning rate')
    net_parser.add_argument('--temp_scal_epochs', type=int, default=1000,
                            help='temperature scaling total epochs. Run model until convergence')

    net_parser.add_argument('--network', default='resnet50')
    net_parser.add_argument('--save_dir', help='directory where to save model weights')
    net_parser.add_argument('--dataset', default='isic2019', help='name of the dataset to use')
    net_parser.add_argument('--dropout', action='store_true', help='Boolean flag for DropOut inclusion')
    net_parser.add_argument('--classes', '-c', type=int, nargs='+',
                            # , default=[[0], [1], [2], [3], [4], [5], [6], [7]]
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

    start_time = time.time()
    net_opt = net_parser.parse_args()
    n = ClassifyNet(net=net_opt.network, dname=net_opt.dataset, dropout=net_opt.dropout,
                    classes=net_opt.classes, l_r=net_opt.learning_rate, loss=net_opt.loss,
                    optimizer=net_opt.optimizer, scheduler=net_opt.scheduler, size=net_opt.size,
                    batch_size=net_opt.batch_size, n_workers=net_opt.workers, pretrained=(not net_opt.from_scratch),
                    augm_config=net_opt.augm_config, save_dir=net_opt.save_dir,
                    cutout_params=[net_opt.cutout_holes, net_opt.cutout_pad], total_epochs=net_opt.epochs,
                    no_logs=True, optimize_temp_scal=True)

    if not net_opt.load_epoch == 0:
        n.load_mode_one(net_opt.load_epoch)

    train_temperature_scaling_decoupled(n, temp_scal_lr=net_opt.temp_scal_learning_rate,
                                        temp_scal_epochs=net_opt.temp_scal_epochs)
    n.save(net_opt.load_epoch)

    pred_variable, labs_variable = n.calibration_variables[2]

    acc, w_acc, calib_stats_withTS, conf_matrix_test, _ = eval(n, n.test_data_loader, pred_variable, labs_variable,
                                                               with_temp_scal=True)
    _, preds, true_lab = calib_stats_withTS
    ECE_test_calib, MCE_test_calib, BRIER_test_calib, NNL_test_calib = compute_calibration_measures(preds, true_lab,
                                                                                                    apply_softmax=False,
                                                                                                    bins=15)

    acc, w_acc, calib_stats_withNOTS, conf_matrix, _ = eval(n, n.test_data_loader, pred_variable, labs_variable,
                                                            with_temp_scal=False)
    _, preds, true_lab = calib_stats_withNOTS
    ECE_test_NOcalib, MCE_test_NOcalib, BRIER_test_NOcalib, NNL_test_NOcalib = compute_calibration_measures(preds,
                                                                                                            true_lab,
                                                                                                            apply_softmax=False,
                                                                                                            bins=15)

    print("\n ----- FINISH ----- \n")
    print("---------TEST------------")
    print("\n|| took {:.1f} minutes \n"
          "| Accuracy statistics: weighted Acc test: {:.3f} Acc test: {:.3f} \n"
          "| Uncalibrated test: ECE: {:.5f} MCE: {:.5f} BRIER: {:.5f}  NNL: {:.5f}\n"
          "| Calibrated test: ECE: {:.5f} MCE: {:.5f} BRIER: {:.5f}  NNL: {:.5f}\n\n".
          format((time.time() - start_time) / 60., w_acc, acc,
                 ECE_test_NOcalib * 100, MCE_test_NOcalib * 100, BRIER_test_NOcalib, NNL_test_NOcalib,
                 ECE_test_calib * 100, MCE_test_calib * 100, BRIER_test_calib, NNL_test_calib))

    pred_variable, labs_variable = n.calibration_variables[1]

    acc, w_acc, calib_stats_withTS, conf_matrix_test, _ = eval(n, n.valid_data_loader, pred_variable, labs_variable,
                                                               with_temp_scal=True)
    _, preds, true_lab = calib_stats_withTS
    ECE_test_calib, MCE_test_calib, BRIER_test_calib, NNL_test_calib = compute_calibration_measures(preds, true_lab,
                                                                                                    apply_softmax=False,
                                                                                                    bins=15)

    acc, w_acc, calib_stats_withNOTS, conf_matrix, _ = eval(n, n.valid_data_loader, pred_variable, labs_variable,
                                                            with_temp_scal=False)
    _, preds, true_lab = calib_stats_withNOTS
    ECE_test_NOcalib, MCE_test_NOcalib, BRIER_test_NOcalib, NNL_test_NOcalib = compute_calibration_measures(preds,
                                                                                                            true_lab,
                                                                                                            apply_softmax=False,
                                                                                                            bins=15)

    print("---------VALID------------")
    print("\n|| took {:.1f} minutes \n"
          "| Accuracy statistics: weighted Acc valid: {:.3f} Acc valid: {:.3f} \n"
          "| Uncalibrated valid: ECE: {:.5f} MCE: {:.5f} BRIER: {:.5f}  NNL: {:.5f}\n"
          "| Calibrated valid: ECE: {:.5f} MCE: {:.5f} BRIER: {:.5f}  NNL: {:.5f}\n\n".
          format((time.time() - start_time) / 60., w_acc, acc,
                 ECE_test_NOcalib * 100, MCE_test_NOcalib * 100, BRIER_test_NOcalib, NNL_test_NOcalib,
                 ECE_test_calib * 100, MCE_test_calib * 100, BRIER_test_calib, NNL_test_calib))
