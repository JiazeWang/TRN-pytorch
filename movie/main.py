import datasets_video
import argparse
import os
import time
import shutil
import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm
import datetime
#from dataset import TSNDataSet
from dataset_test import TSNDataSetMovie
from models import TSN
from transforms import *
from opts import parser

best_prec1 = 0



def main():
    global args, best_prec1
    args = parser.parse_args()

    if args.dataset == 'ucf101':
        num_class = 101
    elif args.dataset == 'hmdb51':
        num_class = 51
    elif args.dataset == 'kinetics':
        num_class = 400
    elif args.dataset == 'movie':
        num_class = 21
    else:
        raise ValueError('Unknown dataset '+args.dataset)

    model = TSN(num_class, args.num_segments, args.modality,
                base_model=args.arch,
                consensus_type=args.consensus_type, dropout=args.dropout, partial_bn=not args.no_partialbn)

    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std
    policies = model.get_optim_policies()
    train_augmentation = model.get_augmentation()

    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            #best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print(("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))

    cudnn.benchmark = True

    # Data loading code
    if args.modality != 'RGBDiff':
        normalize = GroupNormalize(input_mean, input_std)
    else:
        normalize = IdentityTransform()

    if args.modality == 'RGB':
        data_length = 1
    elif args.modality in ['Flow', 'RGBDiff']:
        data_length = 5

    train_loader = torch.utils.data.DataLoader(
        TSNDataSetMovie("", args.train_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl="frame_{:04d}.jpg" if args.modality in ["RGB", "RGBDiff"] else args.flow_prefix+"{}_{:05d}.jpg",
                   transform=torchvision.transforms.Compose([
                       train_augmentation,
                       Stack(roll=args.arch == 'BNInception'),
                       ToTorchFormatTensor(div=args.arch != 'BNInception'),
                       normalize,
                   ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        TSNDataSetMovie("", args.val_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl="frame_{:04d}.jpg" if args.modality in ["RGB", "RGBDiff"] else args.flow_prefix+"{}_{:05d}.jpg",
                   random_shift=False,
                   transform=torchvision.transforms.Compose([
                       GroupScale(int(scale_size)),
                       GroupCenterCrop(crop_size),
                       Stack(roll=args.arch == 'BNInception'),
                       ToTorchFormatTensor(div=args.arch != 'BNInception'),
                       normalize,
                   ])),
        batch_size=int(args.batch_size/2), shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    #if args.loss_type == 'nll':
        #criterion = torch.nn.CrossEntropyLoss().cuda()
    #else:
        #raise ValueError("Unknown loss type")
    #class_weight = torch.tensor([1] * 21).cuda().float()
    #pos_weight = torch.tensor([1] * 21).cuda().float()
    criterion =  torch.nn.BCEWithLogitsLoss().cuda()
    for group in policies:
        print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
            group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))

    optimizer = torch.optim.SGD(policies,
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    zero_time = time.time()
    best_map = 0
    print ('Start training...')
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr_steps)

        # train for one epoch
        start_time = time.time()
        trainloss = train(train_loader, model, criterion, optimizer, epoch)
        print('Traing loss %4f Epoch %d'% (trainloss, epoch))
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            valloss, mAP, wAP, output_mtx = validate(val_loader, model, criterion)
            end_time = time.time()
            epoch_time = end_time - start_time
            total_time = end_time - zero_time
            print ('Total time used: %s Epoch %d time uesd: %s'%(
    				str(datetime.timedelta(seconds=int(total_time))),
    				epoch, str(datetime.timedelta(seconds=int(epoch_time)))))
            print ('Train loss: {0:.4f} val loss: {1:.4f} mAP: {2:.4f} wAP: {3:.4f}'.format(
    		   			trainloss, valloss, mAP, wAP))
            # evaluate on validation set
            is_best = mAP > best_map
            if mAP > best_map:
                best_map = mAP
    			# checkpoint_name = "%04d_%s" % (epoch+1, "checkpoint.pth.tar")
                checkpoint_name = "best_checkpoint.pth.tar"
                save_checkpoint({
    			    'epoch': epoch+1,
    			    'state_dict': model.state_dict(),
    			    'optimizer': optimizer.state_dict(),
    			    }, is_best, epoch)
                npy_name = str(epoch)+args.result_path
                np.save(npy_name, output_mtx)
            with open(args.record_path, 'a') as file:
                file.write('Epoch:[{0}]'
    		   		   'Train loss: {1:.4f} val loss: {2:.4f} map: {3:.4f}\n'.format(
    		   			epoch+1, trainloss, valloss, mAP))


    print ('************ Done!... ************')

def class_precision(scores, labels):
	sortidx = np.argsort(-scores)
	tp = (labels[sortidx]==1).astype(int)
	fp = (labels[sortidx]!=1).astype(int)
	npos = labels.sum()
	fp=np.cumsum(fp)
	tp=np.cumsum(tp)
	rec=tp/npos
	prec=tp/(fp+tp)

	ap = 0
	tmp = (labels[sortidx]==1).astype(int)
	for i in range(len(scores)):
		if tmp[i]==1:
			ap=ap+prec[i];
	ap=ap/npos
	return ap

def compute_map(labels, test_scores):
    nclasses = labels.shape[1]
    if nclasses != 21:
        print ('class num wrong! ')
        sys.exit()
    ap_all = np.zeros(labels.shape[1])
    for i in range(nclasses):
        ap_all[i] = class_precision(test_scores[:, i], labels[:, i])
    mAP = np.mean(ap_all)
    #print(ap_all)
    #print(mAP)
    wAP = np.sum(ap_all*np.sum(labels,0))/np.sum(labels);
    return mAP, wAP

def train(train_loader, model, criterion, optimizer, epoch):
    model.train().float()
    losses = 0
    #end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        #data_time.update(time.time() - end)

        target = target.cuda(non_blocking=True)
        input_var = torch.autograd.Variable(input).float()
        target_var = torch.autograd.Variable(target).float()
        #print("target_var.shape:",target_var.size())
        #print(target_var[0])
        #print(target_var[1])

        # compute output
        output = model(input_var)
        output_float = output.float()
        #print(output_float[0])
        loss = criterion(output_float, target_var)
        losses += loss.item()
        """
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1,5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))
        """

        # compute gradient and do SGD step
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
    return losses / (i+1)



def validate(val_loader, model, criterion, logger=None):
    #batch_time = AverageMeter()
    #losses = AverageMeter()
    #top1 = AverageMeter()
    #top5 = AverageMeter()

    # switch to evaluate mode
    model.eval().float()
    losses = 0
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(non_blocking=True)
        input_var = torch.autograd.Variable(input, volatile=True).float()
        target_var = torch.autograd.Variable(target, volatile=True).float()

        # compute output
        output = model(input_var).float()
        loss = criterion(output, target_var)
        losses += loss.item()
        if i == 0:
            output_mtx = output.data.cpu().numpy()
        else:
            output_mtx = np.concatenate((output_mtx, output.data.cpu().numpy()), axis=0)
        # measure accuracy and record loss
        # measure elapsed time
        #batch_time.update(time.time() - end)
        #end = time.time()
        #label_path = '../results/labels.npy'
        #label_path = 'data/ceshi.npy'
        #label_path = 'data/val.npy'
        label_path = '/home/jzwang/code/RGB-FLOW/MovieNet/data/val.npy'
        labels = np.load(label_path)
        mAP, wAP = compute_map(labels, output_mtx)

    return losses / (i+1), mAP, wAP, output_mtx


def save_checkpoint(state, is_best, epoch):
    file_name='checkpoint.pth.tar'
    filename = '_'.join((str(epoch), args.snapshot_pref, args.modality.lower(), file_name))
    torch.save(state, filename)
    if is_best:
        best_name = '_'.join((args.snapshot_pref, args.modality.lower(), 'model_best.pth.tar'))
        shutil.copyfile(filename, best_name)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    lr = args.lr * decay
    decay = args.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']


if __name__ == '__main__':
    main()
