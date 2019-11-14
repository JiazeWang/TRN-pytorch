import argparse
import time

import numpy as np
import torch.nn.parallel
import torch.optim
from sklearn.metrics import confusion_matrix
#from dataset import TSNDataSet
from dataset_test import TSNDataSetMovie
from models import TSN
from transforms import *
from ops import ConsensusModule
import datasets_video
import pdb
from torch.nn import functional as F

def get_map(output, target):
    n_sample, n_label = output.shape
    class_ap = average_precision_score(
        target, output, average=None)
    return class_ap.mean()

# options
parser = argparse.ArgumentParser(
    description="TRN testing on the full validation set")
parser.add_argument('dataset', type=str, choices=['something','jester','moments','charades','movie'])
parser.add_argument('modality', type=str, choices=['RGB', 'Flow', 'RGBDiff'])
parser.add_argument('weights', type=str)
parser.add_argument('--arch', type=str, default="resnet101")
parser.add_argument('--save_scores', type=str, default=None)
parser.add_argument('--test_segments', type=int, default=8)
parser.add_argument('--max_num', type=int, default=-1)
parser.add_argument('--test_crops', type=int, default=1)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--crop_fusion_type', type=str, default='TRNmultiscale',
                    choices=['avg', 'TRN','TRNmultiscale'])
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--img_feature_dim',type=int, default=256)
parser.add_argument('--num_set_segments',type=int, default=1,help='TODO: select multiply set of n-frames from a video')
parser.add_argument('--softmax', type=int, default=0)
parser.add_argument('--val_list', type = str, default = '/home/jzwang/code/Video_3D/movienet/data/movie/movie_test.txt')

args = parser.parse_args()




#categories, args.train_list, args.val_list, args.root_path, prefix = datasets_video.return_dataset(args.dataset, args.modality)
num_class = 21
args.root_path =''
net = TSN(num_class, args.test_segments if args.crop_fusion_type in ['TRN','TRNmultiscale'] else 1, args.modality,
          base_model=args.arch,
          consensus_type=args.crop_fusion_type,
          img_feature_dim=args.img_feature_dim,
          )

checkpoint = torch.load(args.weights)
#print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))

base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(checkpoint['state_dict'].items())}
net.load_state_dict(base_dict)

if args.test_crops == 1:
    cropping = torchvision.transforms.Compose([
        GroupScale(net.scale_size),
        GroupCenterCrop(net.input_size),
    ])
elif args.test_crops == 10:
    cropping = torchvision.transforms.Compose([
        GroupOverSample(net.input_size, net.scale_size)
    ])
else:
    raise ValueError("Only 1 and 10 crops are supported while we got {}".format(args.test_crops))

data_loader = torch.utils.data.DataLoader(
        TSNDataSetMovie(args.root_path, args.val_list, num_segments=args.test_segments,
                   new_length=1 if args.modality == "RGB" else 5,
                   modality=args.modality,
                   image_tmpl="frame_{:04d}.jpg",
                   test_mode=True,
                   transform=torchvision.transforms.Compose([
                       cropping,
                       Stack(roll=(args.arch in ['BNInception','InceptionV3'])),
                       ToTorchFormatTensor(div=(args.arch not in ['BNInception','InceptionV3'])),
                       GroupNormalize(net.input_mean, net.input_std),
                   ])),
        batch_size=16, shuffle=False,
        num_workers=args.workers * 2, pin_memory=True)

if args.gpus is not None:
    devices = [args.gpus[i] for i in range(args.workers)]
else:
    devices = list(range(args.workers))


#net = torch.nn.DataParallel(net.cuda(devices[0]), device_ids=devices)
net = torch.nn.DataParallel(net.cuda())
net.eval()

data_gen = enumerate(data_loader)

total_num = len(data_loader.dataset)
output = []


def eval_video(video_data):
    i, data, label = video_data
    num_crop = args.test_crops

    if args.modality == 'RGB':
        length = 3
    elif args.modality == 'Flow':
        length = 10
    elif args.modality == 'RGBDiff':
        length = 18
    else:
        raise ValueError("Unknown modality "+args.modality)

    input_var = torch.autograd.Variable(data.view(-1, length, data.size(2), data.size(3)),
                                        volatile=True)
    rst = net(input_var)
    if args.softmax==1:
        # take the softmax to normalize the output to probability
        rst = F.softmax(rst)

    rst = rst.data.cpu().numpy().copy()

    if args.crop_fusion_type in ['TRN','TRNmultiscale']:
        rst = rst.reshape(-1, 1, num_class)
    else:
        rst = rst.reshape((num_crop, args.test_segments, num_class)).mean(axis=0).reshape((args.test_segments, 1, num_class))

    return i, rst, label[0]


#proc_start_time = time.time()
#max_num = args.max_num if args.max_num > 0 else len(data_loader.dataset)

#top1 = AverageMeter()
#top5 = AverageMeter()
def validate(val_loader, model, logger=None):
    #batch_time = AverageMeter()
    #losses = AverageMeter()
    #top1 = AverageMeter()
    #top5 = AverageMeter()

    # switch to evaluate mode
    model.eval().float()
    #losses = 0
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(non_blocking=True)
        input_var = torch.autograd.Variable(input, volatile=True).float()
        target_var = torch.autograd.Variable(target, volatile=True).float()
        # compute output
        output = model(input_var).float()
        #print("outpue.shape:",output.shape)
        #loss = criterion(output, target_var)
        #losses += loss.item()
        print(i, len(val_loader))
        if i == 0:
            output_mtx = output.data.cpu().numpy()
        else:
            output_mtx = np.concatenate((output_mtx, output.data.cpu().numpy()), axis=0)

        label_path = '/home/jzwang/code/Video_3D/movienet/data/movie/movie_test.npy'
        #label_path = '/home/jzwang/code/RGB-FLOW/MovieNet/data/new/ceshi_val.npy'
        if i == (len(val_loader)-1):
            np.save("output_mtx.npy", output_mtx)
            labels = np.load(label_path)
            print("output_mtx.shape:", output_mtx.shape)
            print("labels.shape:", labels.shape)
            mAP = get_map(output_mtx, labels)
    return mAP
val_loader = data_loader
result = validate(val_loader, net)
print("Validation_result", result)

"""
for i, (data, label) in data_gen:
    if i >= max_num:
        break
    rst = eval_video((i, data, label))
    output.append(rst[1:])
    cnt_time = time.time() - proc_start_time
    prec1, prec5 = accuracy(torch.from_numpy(np.mean(rst[1], axis=0)), label, topk=(1, 5))
    top1.update(prec1[0], 1)
    top5.update(prec5[0], 1)
    print('video {} done, total {}/{}, average {:.3f} sec/video, moving Prec@1 {:.3f} Prec@5 {:.3f}'.format(i, i+1,
                                                                    total_num,
                                                                    float(cnt_time) / (i+1), top1.avg, top5.avg))

video_pred = [np.argmax(np.mean(x[0], axis=0)) for x in output]

video_labels = [x[1] for x in output]


cf = confusion_matrix(video_labels, video_pred).astype(float)

cls_cnt = cf.sum(axis=1)
cls_hit = np.diag(cf)

cls_acc = cls_hit / cls_cnt

print('-----Evaluation is finished------')
print('Class Accuracy {:.02f}%'.format(np.mean(cls_acc) * 100))
print('Overall Prec@1 {:.02f}% Prec@5 {:.02f}%'.format(top1.avg, top5.avg))

if args.save_scores is not None:

    # reorder before saving
    name_list = [x.strip().split()[0] for x in open(args.val_list)]
    order_dict = {e:i for i, e in enumerate(sorted(name_list))}
    reorder_output = [None] * len(output)
    reorder_label = [None] * len(output)
    reorder_pred = [None] * len(output)
    output_csv = []
    for i in range(len(output)):
        idx = order_dict[name_list[i]]
        reorder_output[idx] = output[i]
        reorder_label[idx] = video_labels[i]
        reorder_pred[idx] = video_pred[i]
        output_csv.append('%s;%s'%(name_list[i], categories[video_pred[i]]))

    np.savez(args.save_scores, scores=reorder_output, labels=reorder_label, predictions=reorder_pred, cf=cf)

    with open(args.save_scores.replace('npz','csv'),'w') as f:
        f.write('\n'.join(output_csv))
"""
