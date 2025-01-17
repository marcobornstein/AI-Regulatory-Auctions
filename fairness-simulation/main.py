"""
Adapted from Transferring Fairness under Distribution Shifts via Fair Consistency Regularization
by Bang An, Zora Che, Mucong Ding and Furong Huang
https://github.com/umd-huang-lab/transfer-fairness
"""

import pandas as pd
import torch.multiprocessing
from argparse import ArgumentParser
from utils import *
import random
import os
from make_models import *
import torchvision.models as models


parser = ArgumentParser()

parser.add_argument('--num-maj', type=int, default=5000)
parser.add_argument('--per-min', type=float, default=0.2)
parser.add_argument('--num-labels', type=int, default=2)
parser.add_argument('--num-groups', type=int, default=2)
parser.add_argument('--num-workers', type=int, default=1)
parser.add_argument('--epoch', type=int, default=10)
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--test-batch-size', type=int, default=256)
parser.add_argument('--model', choices=['vgg16', 'resnet18', "mlp", 'cnn'], default='vgg16')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--weight-decay', type=float, default=5e-4)
parser.add_argument('--step-lr', type=int, default=100)
parser.add_argument('--step-lr-gamma', type=float, default=0.1)
parser.add_argument('--val-epoch', type=int, default=1)
parser.add_argument('--fair-type', choices=['dp', 'eql_op_0', 'eql_op_1', 'eql_odd'], default='eql_odd')
parser.add_argument('--image-list-train', type=str, default='train_white_black.csv')
parser.add_argument('--image-list-val', type=str, default='val_white_black.csv')
parser.add_argument('--image-list-test', type=str, default='test_white_black.csv')
parser.add_argument('--root', type=str, default='data/fairface')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--save-csv-path', type=str, default='results')
parser.add_argument('--save-path', type=str, default='checkpoint')
parser.add_argument('--save-model', action='store_true', default=False)
args = parser.parse_args()


def gen_data_mixture(num_majority_class, pct_minority_class, image_list, seed):
    df = pd.read_csv(image_list)
    pct_majority_class = 1-pct_minority_class
    total = num_majority_class // pct_majority_class
    num_minority_class = int(total * pct_minority_class)
    majority_class = df[df['race'] == 0]
    minority_class = df[df['race'] == 1]
    # Sample the rows without replacement
    sampled_maj = majority_class.sample(n=num_majority_class, replace=False, random_state=seed)
    sampled_min = minority_class.sample(n=num_minority_class, replace=False, random_state=seed)
    mixture = pd.concat([sampled_maj, sampled_min], ignore_index=True)
    return mixture


def load_model(args):
    if args.model == 'vgg16':
        features = models.vgg16(pretrained=False).features
        model = Face(features, args.num_labels)
    return model


def main(args):
    # filling additional args for specific dataset
    # args = fill_args(args)
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(args)

    torch.manual_seed(args.seed)
    # np.random.seed(seed)
    random.seed(args.seed)


    print("Num of majority group", args.num_maj)
    print("Percentage of minority group", args.per_min)


    args.save_name = f"{args.num_maj}-{args.per_min}-{args.seed}"

    # load training data with given skew
    mixture_df = gen_data_mixture(args.num_maj, args.per_min, os.path.join(args.root, args.image_list_train), args.seed)

    # load validation data with same skew
    val_df = gen_data_mixture(500, args.per_min, os.path.join(args.root, args.image_list_val), args.seed)
    # val_df = pd.read_csv(os.path.join(args.root,args.image_list_val))

    # load regular test data
    test_df = pd.read_csv(os.path.join(args.root, args.image_list_test))

    # create datasets
    train_dataset = data_loader.FairFaceDataset(root=args.root, images_file=mixture_df,
                                                transform=transforms.ToTensor(), transform_strong=None)

    val_dataset = data_loader.FairFaceDataset(root=args.root, images_file=val_df,
                                              transform=transforms.ToTensor(), transform_strong=None)

    test_dataset = data_loader.FairFaceDataset(root=args.root, images_file=test_df,
                                               transform=transforms.ToTensor(), transform_strong=None)

    print('Train dataset size: {}'.format(len(train_dataset)))
    print('Val dataset size: {}'.format(len(val_dataset)))
    print('Test dataset size: {}'.format(len(test_dataset)))

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                                    shuffle=True,
                                    num_workers=args.num_workers)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size,
                                    shuffle=True,
                                    num_workers=args.num_workers)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.test_batch_size,
                                   shuffle=True,
                                   num_workers=args.num_workers)

    loaders = (train_dataloader, val_dataloader, test_dataloader)

    # make model
    model = load_model(args)
    model.to(args.device)
    return train_model(args, model, loaders)


def train_model(args, model, loaders):

    # unpack datasets
    train_dataloader, val_dataloader, test_dataloader = loaders

    # initialize the loader
    optimizer = torch.optim.SGD((list(model.parameters())), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_lr, gamma=args.step_lr_gamma)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    # statistic
    best_t_acc_fair_odd = 0
    best_t_acc_odd = 0
    best_t_unfair_odd = 0
    best_epoch_odd = 0
    best_t_acc_fair_var = 0
    best_t_acc_var = 0
    best_t_unfair_var = 0
    best_epoch_var = 0
    results = []

    for epoch in range(args.epoch):

        # train
        _, _, _ = train_loop(args, epoch, train_dataloader, model, optimizer)

        # validation
        if epoch % args.val_epoch == args.val_epoch - 1:
            t_val_loss, t_val_prec, t_val_unfair_var, t_val_unfair_odd, t_result = eval_loop(args,
                                                                                             epoch,
                                                                                             val_dataloader,
                                                                                             model)
            results.append(t_result)

            run = False
            if t_val_prec - t_val_unfair_odd >= best_t_acc_fair_odd:
                run = True
                best_t_acc_fair_odd = t_val_prec - t_val_unfair_odd
                # best_t_acc_odd = t_val_prec
                # best_t_unfair_odd = t_val_unfair_odd
                # best_epoch_odd = epoch
                # if args.save_model:
                #     sd_info = {
                #         'model': model.state_dict(),
                #         'optimizer': optimizer.state_dict(),
                #         'scheduler': (scheduler and scheduler.state_dict()),
                #         'epoch': epoch
                #     }
                #     save_checkpoint(args, "best_odd", sd_info)

                _, _, _, _, t_test_result_acc_fair = eval_loop(args, epoch, test_dataloader, model, test=True)

            if t_val_prec >= best_t_acc_odd:
                best_t_acc_odd = t_val_prec
                if run:
                    t_test_result_acc = t_test_result_acc_fair
                else:
                    _, _, _, _, t_test_result_acc = eval_loop(args, epoch, test_dataloader, model,test=True)


        if scheduler: scheduler.step()

    results.append(t_test_result_acc_fair)
    results.append(t_test_result_acc)

    # save results to csv
    fields = ["name", "epoch", "acc", "acc_A0Y0", "acc_A0Y1", "acc_A1Y0", "acc_A1Y1", "acc_var", "acc_dis",
              "err_op_0", "err_op_1", "err_odd"]

    with open(os.path.join(args.save_csv_path, args.save_name) + '.csv', 'w') as f:
        write = csv.writer(f)
        write.writerow(fields)
        write.writerows(results)

    return model


def train_loop(args, epoch, dataloader, model, optimizer):
    # init statistics
    cls_losses = AverageMeter()
    fair_losses = AverageMeter()
    accs = AverageMeter()
    fair_adv_accs = AverageMeter()
    group_correct = torch.zeros((args.num_groups, args.num_labels))
    group_cnt = torch.zeros((args.num_groups, args.num_labels))

    # switch to training mode
    model = model.train()

    # training criterion
    loss_fn = nn.CrossEntropyLoss()


    iterator = enumerate(dataloader)


    # train
    for i, sample_batch in iterator:
        # load source data
        inputs = sample_batch['image'].to(args.device)
        labels = sample_batch['label']['gender'].to(args.device)
        groups = sample_batch['label']['race'].to(args.device)
        # forward
        y, f = model(inputs)
        f_0 = f[(groups == 0)]
        f_1 = f[(groups == 1)]

        # main loss
        cls_loss = loss_fn(y, labels)
        loss = cls_loss

        # fairness loss
        if args.fair_type == 'dp':
            fair_loss = fair_adv(f_0, f_1)
        else:
            labels_a0 = labels[(groups == 0)]
            labels_a1 = labels[(groups == 1)]
            # fair_loss = fair_adv(f_0, f_1, labels_a0, labels_a1, args.fair_type)
        # loss = loss + args.fair_weight * fair_loss

        # statistics
        cls_losses.update(cls_loss.item(), inputs.size(0))
        prec = accuracy(y, labels)[0]
        accs.update(prec.item(), inputs.size(0))
        batch_group_correct, batch_group_cnt = group_accuracy(args, y, labels, groups)
        group_correct += batch_group_correct
        group_cnt += batch_group_cnt
        # fair_losses.update(fair_loss.item(), inputs.size(0))
        # fair_adv_acc = fair_adv.domain_discriminator_accuracy
        # fair_adv_accs.update(fair_adv_acc.item(), inputs.size(0))

        # back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # measure unfairness in source domain
    acc_dis, max_id_acc, min_id_acc = acc_disparity(group_correct, group_cnt)
    group_acc = torch.nan_to_num(group_correct / group_cnt) * 100
    acc_var = torch.std(group_acc, unbiased=False)
    err_op0, max_id_op0, min_id_op0 = eql_op(group_acc, 0)
    err_op1, max_id_op1, min_id_op1 = eql_op(group_acc)
    err_odd, max_id_odd, min_id_odd = eql_odd(group_acc)

    # log
    # print(
    #     '{0} | Cls_Loss:{loss_cls:.4f} Fair_Loss:{loss_fair:.4f} '
    #     'Acc:{acc:.2f} FAdv_Acc:{fair_adv_acc:.2f} | Unfairness acc_var:{acc_var:.2f} '
    #     'acc_dis:{acc_dis:.2f} e_op_0:{err_op0:.2f} e_op_1:{err_op1:.2f} e_odd:{err_odd:.2f}'.format(
    #         epoch, loss_cls=cls_losses.avg, loss_fair=fair_losses.avg,
    #         acc=accs.avg, fair_adv_acc=fair_adv_accs.avg, acc_var=acc_var,
    #         acc_dis=acc_dis, err_op0=err_op0, err_op1=err_op1, err_odd=err_odd,
    #     ))
    print(
    '{0} | Cls_Loss:{loss_cls:.4f} '
    'Acc:{acc:.2f}  | Unfairness acc_var:{acc_var:.2f} '
    'acc_dis:{acc_dis:.2f} e_op_0:{err_op0:.2f} e_op_1:{err_op1:.2f} e_odd:{err_odd:.2f}'.format(
        epoch, loss_cls=cls_losses.avg,
        acc=accs.avg, acc_var=acc_var,
        acc_dis=acc_dis, err_op0=err_op0, err_op1=err_op1, err_odd=err_odd,
    ))

    return cls_losses.avg, accs.avg, err_odd


def eval_loop(args, epoch, dataloader, model,test=False):
    # init statistics
    losses = AverageMeter()
    accs = AverageMeter()
    group_correct = torch.zeros((args.num_groups, args.num_labels))
    group_cnt = torch.zeros((args.num_groups, args.num_labels))

    # switch to eval mode
    model.eval()

    # training criterion
    loss_fn = torch.nn.CrossEntropyLoss()

    # dataloader
    iterator = enumerate(dataloader)

    with torch.no_grad():
        for i, sample_batch in iterator:
            inputs = sample_batch['image']
            inputs = inputs.to(args.device)
            labels = sample_batch['label']['gender'].to(args.device)
            groups = sample_batch['label']['race'].to(args.device)

            # forward
            outputs, features = model(inputs)
            loss = loss_fn(outputs, labels)
            losses.update(loss.item(), inputs.size(0))

            # statistics
            prec = accuracy(outputs, labels)[0]
            accs.update(prec.item(), inputs.size(0))
            batch_group_correct, batch_group_cnt = group_accuracy(args, outputs, labels, groups)
            group_correct += batch_group_correct
            group_cnt += batch_group_cnt

        # measure unfairness
        acc_dis, max_id_acc, min_id_acc = acc_disparity(group_correct, group_cnt)
        group_acc = torch.nan_to_num(group_correct / group_cnt) * 100
        acc_var = torch.std(group_acc, unbiased=False)
        err_op0, max_id_op0, min_id_op0 = eql_op(group_acc, 0)
        err_op1, max_id_op1, min_id_op1 = eql_op(group_acc)
        err_odd, max_id_odd, min_id_odd = eql_odd(group_acc)

        # log

        # save result
        if test:
            result = [args.save_name, f"Test {epoch+1}", accs.avg, group_acc[0][0].item(),
                    group_acc[0][1].item(),
                    group_acc[1][0].item(), group_acc[1][1].item(), acc_var.item(), acc_dis.item(),
                    err_op0.item(),
                    err_op1.item(), err_odd.item()]
        else:
            print('Val Epoch:{0} | Loss {loss:.4f} | Acc {acc:.2f} '
                '[{acc_a0_y0:.2f} {acc_a0_y1:.2f} {acc_a1_y0:.2f} {acc_a1_y1:.2f}]|'
                'acc_var {acc_var:.2f}|'
                'acc_dis {acc_dis:.2f}, ({max_id_acc}, {min_id_acc})|'
                'err_op_0 {err_op0:.2f}, ({max_id_op0}, {min_id_op0})|'
                'err_op_1 {err_op1:.2f}, ({max_id_op1}, {min_id_op1})|'
                'err_odd {err_odd:.2f}, ({max_id_odd}, {min_id_odd})|'.format(
                epoch, loss=losses.avg, acc=accs.avg, acc_var=acc_var,
                err_op0=err_op0, max_id_op0=max_id_op0, min_id_op0=min_id_op0,
                err_op1=err_op1, max_id_op1=max_id_op1, min_id_op1=min_id_op1,
                err_odd=err_odd, max_id_odd=max_id_odd, min_id_odd=min_id_odd,
                acc_dis=acc_dis, max_id_acc=max_id_acc, min_id_acc=min_id_acc,
                acc_a0_y0=group_acc[0][0], acc_a0_y1=group_acc[0][1], acc_a1_y0=group_acc[1][0],
                acc_a1_y1=group_acc[1][1]))
            result = [args.save_name, f"Val {epoch+1}", accs.avg, group_acc[0][0].item(),
                    group_acc[0][1].item(),
                    group_acc[1][0].item(), group_acc[1][1].item(), acc_var.item(), acc_dis.item(),
                    err_op0.item(),
                    err_op1.item(), err_odd.item()]

    return losses.avg, accs.avg, acc_var, err_odd, result


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    model = main(args)