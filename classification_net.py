import logging
import os

os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import torch

from torch import nn
from torch.utils.data import DataLoader
import time
import numpy as np
from model import get_model, get_criterion, get_optimizer, get_scheduler, TemperatureScaling
from data import get_dataset
from utils import ConfusionMatrix, compute_calibration_measures, entropy_categorical


class ClassifyNet:
    def __init__(self, net, dname, dropout, l_r, loss, optimizer, scheduler, size, batch_size, n_workers, augm_config,
                 save_dir, cutout_params, total_epochs,
                 classes=[[0], [1], [2], [3], [4], [5], [6], [7]], pretrained=True, no_logs=False,
                 optimize_temp_scal=False, drop_last=True):
        # Hyper-parameters
        self.net = net
        self.dropout = dropout
        self.dname = dname
        if classes is None:
            self.classes = [[0], [1], [2], [3], [4], [5], [6], [7]]
        elif len(classes) == 1:
            self.classes = [[c] for c in classes[0]]
        else:
            self.classes = classes
        self.num_classes = len(self.classes)
        self.learning_rate = l_r
        self.lossname = loss
        self.optname = optimizer
        self.schedname = scheduler
        self.size = size
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.augm_config = augm_config
        self.pretrained = pretrained
        self.save_dir = save_dir
        self.best_acc = 0.0
        self.cutout_nholes = cutout_params[0]
        self.cutout_pad_size = cutout_params[1]
        self.no_logs = no_logs
        self.optimize_temp_scal = optimize_temp_scal

        self.nname = self.net + '_ISIC2019'
        if self.dropout:
            self.nname = 'dropout_' + self.nname

        self.n = get_model(self.net, self.pretrained, self.num_classes, self.dropout, self.size)

        self.temp_scal_model = None
        if optimize_temp_scal:
            self.temp_scal_model = TemperatureScaling().to('cuda')  # no wrapping for efficiency in training

        self.data_loader, self.test_data_loader, self.valid_data_loader = get_dataset(dname=self.dname,
                                                                                      size=self.size,
                                                                                      dataset_classes=self.classes,
                                                                                      batch_size=self.batch_size,
                                                                                      n_workers=self.n_workers,
                                                                                      augm_config=self.augm_config,
                                                                                      cutout_params=cutout_params,
                                                                                      drop_last_flag=drop_last)

        self.criterion = get_criterion(self.lossname, self.classes)
        self.optimizer = get_optimizer(self.n, self.learning_rate, self.optname)
        self.scheduler = get_scheduler(self.optimizer, self.schedname)

        # to measures calibration stuff
        predictions_train = torch.zeros(len(self.data_loader.dataset), self.num_classes).float()
        labels_train = torch.zeros(len(self.data_loader.dataset), ).long()
        predictions_valid = torch.zeros(len(self.valid_data_loader.dataset), self.num_classes).float()
        labels_valid = torch.zeros(len(self.valid_data_loader.dataset), ).long()
        predictions_test = torch.zeros(len(self.test_data_loader.dataset), self.num_classes).float()
        labels_test = torch.zeros(len(self.test_data_loader.dataset), ).long()

        self.calibration_variables = [[predictions_train, labels_train], [predictions_valid, labels_valid],
                                      [predictions_test, labels_test]]

        # logger
        if not self.no_logs:
            model_log_dir = os.path.join(self.save_dir,
                                         self.nname + '_epoch.' + str(total_epochs) + '_augmentidx.' + str(
                                             self.augm_config) + '_cutout.holes' + str(
                                             self.cutout_nholes) + '.pad.' + str(
                                             self.cutout_pad_size) + '.classes.' + str(self.classes) + '_logger.log')

            logging.basicConfig(filename=model_log_dir, level=logging.INFO)
            self.logger = logging

    def save(self, n_epoch=0):
        if self.optimize_temp_scal:
            self.save_with_temp_scal(n_epoch)
        else:
            self.save_mode_one(n_epoch)

    def save_mode_one(self, n_epoch=0):
        try:
            torch.save(self.n.state_dict(),
                       os.path.join(self.save_dir, self.nname + '_epoch.' + str(n_epoch) + '_augmentidx' + str(
                           self.augm_config) + '_cutout.holes' + str(
                           self.cutout_nholes) + '.pad.' + str(self.cutout_pad_size) + '.classes.' + str(
                           self.classes) + '_net.pth'))
            torch.save(self.optimizer.state_dict(),
                       os.path.join(self.save_dir, self.nname + '_epoch.' + str(n_epoch) + '_augmentidx' + str(
                           self.augm_config) + '_cutout.holes' + str(
                           self.cutout_nholes) + '.pad.' + str(self.cutout_pad_size) + '.classes.' + str(
                           self.classes) + '_opt.pth'))
            print("mode one model weights successfully saved")
        except Exception:
            print("Error during Saving")

    def save_with_temp_scal(self, n_epoch=0):
        try:
            saved_dict = {'model_parameters': self.n.state_dict(),
                          'temperature_scal': self.temp_scal_model.state_dict()}
            torch.save(saved_dict,
                       os.path.join(self.save_dir, self.nname + '_epoch.' + str(n_epoch) + '_augmentidx' + str(
                           self.augm_config) + '_cutout.holes' + str(
                           self.cutout_nholes) + '.pad.' + str(
                           self.cutout_pad_size) + '_temperature.scaling.decoupled_net.pth'))
            torch.save(self.optimizer.state_dict(),
                       os.path.join(self.save_dir, self.nname + '_epoch.' + str(n_epoch) + '_augmentidx' + str(
                           self.augm_config) + '_cutout.holes' + str(
                           self.cutout_nholes) + '.pad.' + str(
                           self.cutout_pad_size) + '_temperature.scaling.decoupled_opt.pth'))
            print("model weights and temp scal T successfully saved")
        except Exception:
            print("Error during Saving")

    def load(self, n_epoch=0):

        if self.optimize_temp_scal:
            try:
                self.load_with_temp_scal(n_epoch)
            except:
                try:
                    self.load_mode_one(n_epoch)
                except:
                    raise FileExistsError
                train_temperature_scaling_decoupled(self, 0.1, 1000)
                self.save_with_temp_scal(n_epoch)
                self.load_with_temp_scal(n_epoch)

        else:
            try:
                self.load_mode_one(n_epoch)
            except:
                raise FileExistsError

    def load_mode_one(self, n_epoch=0):
        self.n.load_state_dict(
            torch.load(os.path.join(self.save_dir, self.nname + '_epoch.' + str(n_epoch) + '_augmentidx' + str(
                self.augm_config) + '_cutout.holes' + str(
                self.cutout_nholes) + '.pad.' + str(self.cutout_pad_size) + '.classes.' + str(
                self.classes) + '_net.pth')))
        self.optimizer.load_state_dict(
            torch.load(os.path.join(self.save_dir, self.nname + '_epoch.' + str(n_epoch) + '_augmentidx' + str(
                self.augm_config) + '_cutout.holes' + str(
                self.cutout_nholes) + '.pad.' + str(self.cutout_pad_size) + '.classes.' + str(
                self.classes) + '_opt.pth')))
        print("mode one model weights successfully loaded")

    def load_with_temp_scal(self, n_epoch=0):
        saved_dict = torch.load(
            os.path.join(self.save_dir, self.nname + '_epoch.' + str(n_epoch) + '_augmentidx' + str(
                self.augm_config) + '_cutout.holes' + str(
                self.cutout_nholes) + '.pad.' + str(self.cutout_pad_size) + '_temperature.scaling.decoupled_net.pth'))

        self.n.load_state_dict(saved_dict['model_parameters'])
        self.temp_scal_model.load_state_dict(saved_dict['temperature_scal'])
        self.optimizer.load_state_dict(
            torch.load(os.path.join(self.save_dir, self.nname + '_epoch.' + str(n_epoch) + '_augmentidx' + str(
                self.augm_config) + '_cutout.holes' + str(
                self.cutout_nholes) + '.pad.' + str(self.cutout_pad_size) + '_temperature.scaling.decoupled_opt.pth')))
        print("model weights and temp scal T successfully loaded")


def train(class_model, num_epochs, starting_e=0):
    for epoch in range(starting_e + 1, num_epochs):
        class_model.n.train()
        losses = []
        start_time = time.time()
        for idx, (x, target, _) in enumerate(class_model.data_loader):
            # measure data loading time
            # print("data time: " + str(time.time() - start_time))

            # compute output
            x = x.to('cuda')
            target = target.to('cuda', torch.long)
            output = torch.squeeze(class_model.n(x))
            loss = class_model.criterion(output, target)
            losses.append(loss.item())
            # compute gradient and do SGD step
            class_model.optimizer.zero_grad()
            loss.backward()
            class_model.optimizer.step()

        acc_valid, w_acc_valid, calibration_statistics, conf_matrix_valid, _ = eval(class_model,
                                                                                    class_model.valid_data_loader,
                                                                                    *class_model.calibration_variables[
                                                                                        1],
                                                                                    class_model.optimize_temp_scal)

        _, preds, true_lab = calibration_statistics
        ECE_valid, MCE_valid, BRIER_valid, NNL_valid = compute_calibration_measures(preds, true_lab,
                                                                                    apply_softmax=False, bins=15)

        acc_test, w_acc_test, calibration_statistics, conf_matrix_test, _ = eval(class_model,
                                                                                 class_model.test_data_loader,
                                                                                 *class_model.calibration_variables[2],
                                                                                 class_model.optimize_temp_scal)

        _, preds, true_lab = calibration_statistics
        ECE_test, MCE_test, BRIER_test, NNL_test = compute_calibration_measures(preds, true_lab, apply_softmax=False,
                                                                                bins=15)

        class_model.logger.info("\n|| Epoch {} took {:.1f} minutes \t LossCE {:.5f} \n"
                                "| Accuracy statistics: weighted Acc valid: {:.3f}  weighted Acc test: {:.3f} Acc valid: {:.3f}  Acc test: {:.3f} \n"
                                "| Calibration valid: ECE: {:.5f} MCE: {:.3f} BRIER: {:.3f} NNL: {:.5f} \n"
                                "| Calibration test: ECE: {:.5f} MCE: {:.5f} BRIER: {:.5f}  NNL: {:.5f}\n\n".format(
            epoch, (time.time() - start_time) / 60., np.mean(losses), w_acc_valid, w_acc_test, acc_valid,
            acc_test, ECE_valid * 100., MCE_valid * 100., BRIER_valid, NNL_valid,
                   ECE_test * 100., MCE_test * 100., BRIER_test, NNL_test))

        print("\n|| Epoch {} took {:.1f} minutes \t LossCE {:.5f} \n"
              "| Accuracy statistics: weighted Acc valid: {:.3f}  weighted Acc test: {:.3f} Acc valid: {:.3f}  Acc test: {:.3f} \n"
              "| Calibration valid: ECE: {:.5f} MCE: {:.3f} BRIER: {:.3f} NNL: {:.5f} \n"
              "| Calibration test: ECE: {:.5f} MCE: {:.5f} BRIER: {:.5f}  NNL: {:.5f}\n\n".format(epoch, (
                time.time() - start_time) / 60., np.mean(losses), w_acc_valid, w_acc_test, acc_valid,
                                                                                                  acc_test,
                                                                                                  ECE_valid * 100.,
                                                                                                  MCE_valid * 100.,
                                                                                                  BRIER_valid,
                                                                                                  NNL_valid,
                                                                                                  ECE_test * 100.,
                                                                                                  MCE_test * 100.,
                                                                                                  BRIER_test, NNL_test))

        print(conf_matrix_valid)
        print('\n')
        print(conf_matrix_test)

        if (w_acc_valid > class_model.best_acc or epoch % 20 == 0) and epoch > 20:
            print("SAVING MODEL")
            class_model.save(epoch)
            class_model.best_acc = w_acc_valid

        if class_model.schedname == 'plateau':
            class_model.scheduler.step(w_acc_valid)


def eval(class_model, e_loader, predictions, labels, with_temp_scal=False, compute_separate_metrics_for_errors=False):
    with torch.no_grad():
        entropy_of_predictions = torch.zeros_like(labels).float()

        '''to measure stuff on correct and incorrect classified samples'''
        if compute_separate_metrics_for_errors:
            corr_entropy = torch.zeros_like(labels).float()
            incorr_entropy = torch.zeros_like(labels).float()

            corr_labels = torch.zeros_like(labels).float()
            incorr_labels = torch.zeros_like(labels).float()

            corr_predictions = torch.zeros_like(predictions).float()
            incorr_predictions = torch.zeros_like(predictions).float()

            corr_count = 0
            incorr_count = 0

        class_model.n.eval()
        sofmx = nn.Softmax(dim=-1)
        conf_matrix = ConfusionMatrix(class_model.num_classes)

        start_time = time.time()
        for idx, (x, target, img_name) in enumerate(e_loader):
            # measure data loading time
            # print("data time: " + str(time.time() - start_time))
            # compute output
            x = x.to('cuda')
            out = class_model.n(x)

            if with_temp_scal:
                out = class_model.temp_scal_model(out)

            # output = torch.squeeze(out)
            output = out

            target = target.to('cuda', torch.long)
            check_output_all = sofmx(output)
            check_output, res = torch.max(check_output_all, -1)

            aux = target.size(0)
            predictions[idx * class_model.batch_size:idx * class_model.batch_size + aux,
            :] = check_output_all.data.cpu()
            labels[idx * class_model.batch_size:idx * class_model.batch_size + aux] = target.data.cpu()

            entropy_of_predictions[
            idx * class_model.batch_size:idx * class_model.batch_size + aux] = entropy_categorical(
                check_output_all).cpu()

            # update the confusion matrix
            conf_matrix.update_matrix(res, target)

            if compute_separate_metrics_for_errors:
                # if true we compute the entropy and calibration measures on correct and incorrect samples separately
                corr_idx = check_output_all.argmax(dim=1) == target
                incorr_idx = check_output_all.argmax(dim=1) != target

                corr_samples_prob = check_output_all[corr_idx, :]
                incorr_samples_prob = check_output_all[incorr_idx, :]

                corr_numel = corr_idx.sum().long()
                incorr_numel = incorr_idx.sum().long()

                corr_entropy[corr_count:corr_count + corr_numel] = entropy_categorical(corr_samples_prob).cpu()
                incorr_entropy[incorr_count:incorr_count + incorr_numel] = entropy_categorical(
                    incorr_samples_prob).cpu()

                corr_predictions[corr_count:corr_count + corr_numel] = corr_samples_prob.cpu()
                incorr_predictions[incorr_count:incorr_count + incorr_numel] = incorr_samples_prob.cpu()

                corr_labels[corr_count:corr_count + corr_numel] = target[corr_idx].cpu()
                incorr_labels[incorr_count:incorr_count + incorr_numel] = target[incorr_idx].cpu()

                corr_count += corr_numel
                incorr_count += incorr_numel

        # filter out the zeros
        per_samples_stats = None
        if compute_separate_metrics_for_errors:
            corr_entropy = corr_entropy[0:corr_count]
            incorr_entropy = incorr_entropy[0:incorr_count]

            corr_predictions = corr_predictions[0:corr_count]
            incorr_predictions = incorr_predictions[0:incorr_count]

            corr_labels = corr_labels[0:corr_count]
            incorr_labels = incorr_labels[0:incorr_count]

            per_samples_stats = {
                'corr': [corr_entropy, corr_predictions, corr_labels],
                'incorr': [incorr_entropy, incorr_predictions, incorr_labels]
            }

        acc, w_acc = conf_matrix.get_metrics()

        return acc, w_acc, [entropy_of_predictions, predictions, labels], conf_matrix.conf_matrix, per_samples_stats


def ensemble_aug_eval(n_iter, class_model, with_temp_scal=False):
    acc_test = 0
    w_acc_test = 0
    ens_preds = torch.zeros_like(class_model.calibration_variables[2][0])

    start_time = time.time()

    data_loader, test_data_loader, valid_data_loader = class_model.data_loader, class_model.test_data_loader, class_model.valid_data_loader

    for i in range(1, n_iter + 1):
        acc_test_temp, w_acc_test_temp, calibration_statistics, conf_matrix_temp, _ = \
            eval(class_model, test_data_loader, *class_model.calibration_variables[2], with_temp_scal)
        acc_test += acc_test_temp
        w_acc_test += w_acc_test_temp

        _, preds, true_lab = calibration_statistics
        ens_preds += preds

    conf_matrix_test = ConfusionMatrix(class_model.num_classes)
    temp_ens_preds = ens_preds / n_iter
    check_output, res = torch.max(torch.tensor(temp_ens_preds, device='cuda'), 1)
    conf_matrix_test.update_matrix(res, torch.tensor(true_lab, device='cuda'))

    ens_acc, ens_w_acc = conf_matrix_test.get_metrics()
    ECE_test, MCE_test, BRIER_test, NNL_test = compute_calibration_measures(temp_ens_preds, true_lab,
                                                                            apply_softmax=False,
                                                                            bins=15)
    print("\n|| took {:.1f} minutes \n"
          "| Mean Accuracy statistics: weighted Acc test: {:.3f} Acc test: {:.3f} \n"
          "| Ensemble Accuracy statistics: weighted Acc test: {:.3f} Acc test: {:.3f} \n"
          "| Calibration test: ECE: {:.5f} MCE: {:.5f} BRIER: {:.5f}  NNL: {:.5f}\n\n".
          format((time.time() - start_time) / 60., w_acc_test / i, acc_test / i, ens_w_acc, ens_acc,
                 ECE_test * 100, MCE_test * 100, BRIER_test, NNL_test))
    print(conf_matrix_test.conf_matrix)

    return ens_acc, ens_w_acc, (ens_preds / n_iter), true_lab


def train_temperature_scaling_decoupled(class_model, temp_scal_lr, temp_scal_epochs):
    class_model.n.eval()

    data_loader, test_data_loader, valid_data_loader = get_dataset(dname='isic2019',
                                                                   dataset_classes=class_model.classes,
                                                                   size=class_model.size,
                                                                   batch_size=class_model.batch_size,
                                                                   n_workers=class_model.n_workers,
                                                                   augm_config=class_model.augm_config,
                                                                   cutout_params=[class_model.cutout_nholes,
                                                                                  class_model.cutout_pad_size])

    validation_logit_storage = torch.zeros((len(valid_data_loader.dataset), class_model.num_classes)).float()
    validation_label_storage = torch.zeros((len(valid_data_loader.dataset))).long()
    acc = 0

    start_time = time.time()
    with torch.no_grad():
        for idx, (x, target, img_name) in enumerate(valid_data_loader):
            # measure data loading time
            # print("data time: " + str(time.time() - start_time))
            # compute output
            x = x.to('cuda')
            output = torch.squeeze(class_model.n(x))

            target = target.to('cuda', torch.long)

            validation_logit_storage[acc:acc + target.size(0), :] = output.to('cpu')
            validation_label_storage[acc:acc + target.size(0)] = target.to('cpu')

            acc += target.size(0)

    dataset_logit = torch.utils.data.TensorDataset(validation_logit_storage, validation_label_storage)
    data_loader_logit = DataLoader(dataset_logit, batch_size=100, shuffle=True, num_workers=0, drop_last=False,
                                   pin_memory=True)

    optim = torch.optim.SGD(class_model.temp_scal_model.parameters(), lr=temp_scal_lr, momentum=0.9)
    scheduler = get_scheduler(optim, 'plateau')

    for e in range(temp_scal_epochs):
        LOSS = 0.0

        for logit, target in data_loader_logit:
            logit, target = logit.to('cuda'), target.to('cuda')
            out = class_model.temp_scal_model(logit)
            loss = class_model.temp_scal_model.loss(out, target)
            loss.backward()
            LOSS += loss
            optim.step()
            optim.zero_grad()

        scheduler.step(LOSS)
        for param_group in optim.param_groups:
            current_lr = param_group['lr']
        print('Run until convergence!!!. On epoch {} with lr {} loss {:.5f} with T {:.5f}\r'.format(e, current_lr, LOSS,
                                                                                                    class_model.temp_scal_model.T.item()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--network', default='resnet50')
    parser.add_argument('--save_dir', required=True, help='directory where to save model weights')
    parser.add_argument('--dataset', default='isic2019', help='name of the dataset to use')
    parser.add_argument('--dropout', action='store_true', help='Boolean flag for DropOut inclusion')
    parser.add_argument('--classes', '-c', type=int, nargs='+',  # , default=[[0], [1], [2], [3], [4], [5], [6], [7]]
                        action='append', help='classes to train the model with')
    parser.add_argument('--load_epoch', type=int, default=0, help='load custom-trained models')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size during the training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--loss', default='cross_entropy', choices=['cross_entropy', 'focal'])
    parser.add_argument('--optimizer', default='SGD', choices=['SGD', 'Adam'])
    parser.add_argument('--scheduler', default='plateau', choices=['plateau', 'None'])
    parser.add_argument('--epochs', type=int, default=150, help='number of epochs to train')
    parser.add_argument('--size', type=int, default=512, help='size of images')
    parser.add_argument('--from_scratch', action='store_true', help='Boolean flag for training a model from scratch')
    parser.add_argument('--augm_config', type=int, default=1,
                        help='configuration code for augmentation techniques choice')
    parser.add_argument('--cutout_pad', nargs='+', type=int, default=[0], help='cutout pad.')
    parser.add_argument('--cutout_holes', nargs='+', type=int, default=[0], help='number of cutout holes.')
    parser.add_argument('--calibrated', action='store_true', help='Boolean flag for applying temperature scaling')

    opt = parser.parse_args()
    print(opt)

    n = ClassifyNet(net=opt.network, dname=opt.dataset, dropout=opt.dropout, classes=opt.classes,
                    l_r=opt.learning_rate, loss=opt.loss, optimizer=opt.optimizer, scheduler=opt.scheduler,
                    size=opt.size, batch_size=opt.batch_size, n_workers=opt.workers, pretrained=(not opt.from_scratch),
                    augm_config=opt.augm_config, save_dir=opt.save_dir,
                    cutout_params=[opt.cutout_holes, opt.cutout_pad], total_epochs=opt.epochs,
                    optimize_temp_scal=opt.calibrated)

    if not opt.load_epoch == 0:
        n.load(opt.load_epoch)
        acc, w_acc, calib, conf_matrix, _ = eval(n, n.test_data_loader, *n.calibration_variables[2], opt.calibrated)
        print(acc)
        print(w_acc)
        print(conf_matrix)

    train(n, opt.epochs, opt.load_epoch)
    n.save(opt.epochs)
