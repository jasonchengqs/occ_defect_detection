import os
import argparse
import time
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from datasets.mvtec import MVTecDataset
from datasets.emdc import EMDCDataset
from models.resnet_backbone import modified_resnet18
from utils.util import  time_string, convert_secs2time, AverageMeter
from utils.functions import cal_anomaly_maps, cal_loss
from utils.visualization import plt_fig
from utils.gen_mask import gen_mask
from models.unet import UNet

class STPM():
    def __init__(self, args):
        self.device = args.device
        self.data_path = args.data_path
        self.test_data_path = args.test_data_path
        self.obj = args.obj
        self.img_size = args.img_size
        self.img_cropsize = args.img_cropsize
        self.validation_ratio = args.validation_ratio
        self.num_epochs = args.num_epochs
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.vis = args.vis
        self.model_dir = args.model_dir
        self.img_dir = args.img_dir
        self.threshold = args.threshold
        self.kd_tasks = args.kd_tasks
        
        self.load_checkpoints()
        self.load_model()
        self.load_dataset()

        # self.criterion = torch.nn.MSELoss(reduction='sum')
        self.params = []
        for task, pair in self.kd_models.items():
            _params = list(pair['s'].parameters())
            self.params.extend(_params)
        self.optimizer = torch.optim.SGD(self.params, lr=self.lr, momentum=0.9, weight_decay=0.0001)

    def load_checkpoints(self):
        self.t_ckpts = {}
        self.s_ckpts = {}
        self.t_ckpts['recon'] = args.recon_t_ckpt
        self.t_ckpts['sr'] = args.sr_t_ckpt
        self.s_ckpts['cls'] = args.cls_s_ckpt
        self.s_ckpts['recon'] = args.recon_s_ckpt
        self.s_ckpts['sr'] = args.sr_s_ckpt

    def load_dataset(self):
        kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
        train_dataset = EMDCDataset(args.data_path, phase='train', sample_size=5000, resize=args.img_size, cropsize=args.img_size)
        val_dataset = EMDCDataset(args.data_path, phase='val', sample_size=5000, resize=args.img_size, cropsize=args.img_size)
        # img_nums = len(train_dataset)
        # valid_num = int(img_nums * self.validation_ratio)
        # train_num = img_nums - valid_num
        # train_data, val_data = torch.utils.data.random_split(train_dataset, [train_num, valid_num])
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, **kwargs)
        self.val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, **kwargs)

    def _load_by_task(self, task, load_student_status=False):
        if task == 'cls':
            model_t = modified_resnet18().to(self.device)
            model_s = modified_resnet18(pretrained=False).to(self.device)
        elif task == 'recon':
            model_t = UNet(return_feats=True).to(self.device)
            model_t.load_state_dict(torch.load(self.t_ckpts['recon'])['model'])
            model_s = UNet(return_feats=True).to(self.device)
        elif task == 'sr':
            pass
        model_t.eval()
        if load_student_status:
            model_s.load_state_dict(torch.load(self.s_ckpts[task])['model'])
        return model_t, model_s

    def load_model(self, load_student_status=False):
        self.kd_models = {}
        for task in self.kd_tasks:
            model_t, model_s = self._load_by_task(task, load_student_status)
            for param in model_t.parameters():
                param.requires_grad = False
            self.kd_models[task] = {'t': model_t, 's': model_s}

    def _set_student_status(self, is_train=True):
        for _, pair in self.kd_models.items():
            if is_train:
                pair['s'].train()
            else:
                pair['s'].eval()

    def _forward(self, data, task, pair):
        model_t = pair['t']
        model_s = pair['s']
        feat_t = model_t(data)
        feat_s = model_s(data)
        y = None
        if task != 'cls':
            y = feat_t[0]
            feat_t = feat_t[1:]
            feat_s = feat_s[1:]
        return y, feat_t, feat_s

    def train(self):
        self._set_student_status(is_train=True)
        best_score = None
        start_time = time.time()
        epoch_time = AverageMeter()
        for epoch in range(1, self.num_epochs+1):
            need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * ((self.num_epochs+1) - epoch))
            need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)
            print('{:3d}/{:3d} ----- [{:s}] {:s}'.format(epoch, self.num_epochs, time_string(), need_time))
            losses = AverageMeter()
            for (data, _) in tqdm(self.train_loader):
                data = data.to(self.device)
                self.optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    loss = 0
                    for task, pair in self.kd_models.items():
                        y, features_t, features_s = self._forward(data, task, pair)
                        loss += cal_loss(features_s, features_t)
                    losses.update(loss.sum().item(), data.size(0))
                    loss.backward()
                    self.optimizer.step()
            print('Train Epoch: {} loss: {:.6f}'.format(epoch, losses.avg))

            val_loss = self.val(epoch)
            if best_score is None:
                best_score = val_loss
                self.save_checkpoint()
            elif val_loss < best_score:
                best_score = val_loss
                self.save_checkpoint()
            
            epoch_time.update(time.time() - start_time)
            start_time = time.time()

        print('Training end.')
    
    def val(self, epoch):
        self._set_student_status(is_train=False)
        losses = AverageMeter()
        for (data, _) in tqdm(self.val_loader):
            data = data.to(self.device)
            with torch.set_grad_enabled(False):
                loss = 0
                for task, pair in self.kd_models.items():
                    y, features_t, features_s = self._forward(data, task, pair)
                    loss += cal_loss(features_s, features_t)
                losses.update(loss.item(), data.size(0))
        print('Val Epoch: {} loss: {:.6f}'.format(epoch, losses.avg))

        return losses.avg

    def get_state(self):
        state = {}
        for task, pair in self.kd_models.items():
            state[task] = pair['s'].state_dict()
        return state

    def save_checkpoint(self):
        print('Save model !!!')
        state = self.get_state()
        torch.save(state, os.path.join(self.model_dir, 'model_s.pth'))

    def test(self):
        # try:
        #     checkpoint = torch.load(os.path.join(self.model_dir, 'model_s.pth'))
        # except:
        #     raise Exception('Check saved model path.')
        # self.model_s.load_state_dict(checkpoint['model'])
        self.load_model(load_student_status=True)
        self._set_student_status(is_train=False)

        kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
        # test_dataset = MVTecDataset(self.data_path, class_name=self.obj, is_train=False, resize=self.img_resize, cropsize=self.img_cropsize)
        test_dataset = EMDCDataset(args.test_data_path, phase='test', sample_size=1000, resize=args.img_size, cropsize=args.img_size)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, **kwargs)

        scores = []
        test_imgs = []
        gt_list = []
        gt_f_list = []
        print('Testing')
        for (data, label, f) in tqdm(test_loader):
            test_imgs.extend(data.cpu().numpy())
            gt_list.extend(label.cpu().numpy())
            gt_f_list.extend(f)

            data = data.to(self.device)
            with torch.set_grad_enabled(False):
                feat_t_list = []
                feat_s_list = []
                for task, pair in self.kd_models.items():
                    y, features_t, features_s = self._forward(data, task, pair)
                    feat_t_list.extend(features_t)
                    feat_s_list.extend(features_s)
                score = cal_anomaly_maps(feat_s_list, feat_t_list, self.img_cropsize)
            scores.extend(score)

        scores = np.asarray(scores)
        max_anomaly_score = scores.max()
        min_anomaly_score = scores.min()
        scores = (scores - min_anomaly_score) / (max_anomaly_score - min_anomaly_score)

        img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
        gt_list = np.asarray(gt_list)
        img_roc_auc = roc_auc_score(gt_list, img_scores)
        print('image ROCAUC: %.3f' % (img_roc_auc))

        precision, recall, thresholds = precision_recall_curve(gt_list.flatten(), img_scores.flatten())
        a = 2 * precision * recall
        b = precision + recall
        f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        cls_threshold = thresholds[np.argmax(f1)]

        # gt_mask = np.asarray(gt_mask_list)
        # precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), scores.flatten())
        # a = 2 * precision * recall
        # b = precision + recall
        # f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        # seg_threshold = thresholds[np.argmax(f1)]

        # per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores.flatten())
        # print('pixel ROCAUC: %.3f' % (per_pixel_rocauc))

        if self.vis:
            seg_threshold = self.threshold
            plt_fig(test_imgs, scores, img_scores, gt_f_list, seg_threshold, cls_threshold, 
                    self.img_dir, self.obj)


def get_args():
    parser = argparse.ArgumentParser(description='STPM anomaly detection')
    parser.add_argument('--phase', default='train')
    parser.add_argument("--data_path", type=str, default="D:/dataset/mvtec_anomaly_detection")
    parser.add_argument("--test_data_path", type=str, default="D:/dataset/mvtec_anomaly_detection")
    parser.add_argument('--obj', type=str, default='cls')
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--img_cropsize', type=int, default=256)
    parser.add_argument('--validation_ratio', type=float, default=0.2)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--vis', type=eval, choices=[True, False], default=False)
    parser.add_argument("--save_path", type=str, default="./mvtec_results")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument('--kd_tasks', type=str, nargs='+', default=['cls', 'recon', 'sr'])
    parser.add_argument('--recon_t_ckpt', type=str, default='')
    parser.add_argument('--sr_t_ckpt', type=str, default='')
    parser.add_argument('--cls_s_ckpt', type=str, default='')
    parser.add_argument('--recon_s_ckpt', type=str, default='')
    parser.add_argument('--sr_s_ckpt', type=str, default='')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    
    print ('Available devices ', torch.cuda.device_count())
    print ('Current cuda device ', torch.cuda.current_device())

    args = get_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    args.model_dir = args.save_path + '/models' + '/' + args.obj
    args.img_dir = args.save_path + '/imgs' + '/' + args.obj
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.img_dir, exist_ok=True)

    stpm = STPM(args)
    if args.phase == 'train':
        stpm.train()
        # stpm.test()
    elif args.phase == 'test':
        stpm.test()
    else:
        print('Phase argument must be train or test.')







    

