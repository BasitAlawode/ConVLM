from tqdm import tqdm
import torch
import numpy as np
from scipy.special import softmax
import torch.nn.functional as F
from helpers import adjust_keep_rate

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




def train(args,model, data_loader, train_attrbs, optimizer, use_cuda,epoch,lamb_1=1.0):
    it = epoch * len(data_loader)
    ITERS_PER_EPOCH = len(data_loader)
    base_rate = args.base_keep_rate
    """returns trained model"""    
    # initialize variables to monitor training and validation loss
    loss_meter = AverageMeter()
    """ train the model  """
    model.train()
    tk = tqdm(data_loader, total=int(len(data_loader)))
    for batch_idx, (data, label) in enumerate(tk):
        # keep_rate = adjust_keep_rate(it, epoch, warmup_epochs=args.shrink_start_epoch,
        #                                  total_epochs=args.shrink_start_epoch + args.shrink_epochs,
        #                                  ITERS_PER_EPOCH=ITERS_PER_EPOCH, base_keep_rate=base_rate)

        keep_rate = args.keep_rate

        # move to GPU
        if use_cuda:
            data,  label = data.to(args.device,non_blocking=True), label.to(args.device,non_blocking=True)
        optimizer.zero_grad()
        
        x_g,x_aux,aux,token_cls_reg = model.vit(data,keep_rate,label=label,att=train_attrbs) 

        
        feat_g = model.mlp_g(x_g)
        logit_g = feat_g @ train_attrbs.T
        loss1 = lamb_1 * F.cross_entropy(logit_g, label)
        loss3 = args.loss_global_alignment * (F.l1_loss(aux[0],x_g)+F.l1_loss(aux[1],x_g)+F.l1_loss(aux[2],x_g))
        loss4  = args.loss_sr * (F.l1_loss(token_cls_reg[0],train_attrbs[label])+F.l1_loss(token_cls_reg[1],train_attrbs[label])+F.l1_loss(token_cls_reg[2],train_attrbs[label]))
        loss = loss1 + loss3 + loss4
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item(), label.shape[0])
        tk.set_postfix({"loss": loss_meter.avg})
        
    # print training/validation statistics 
    print('Train: Average loss: {:.4f}'.format(loss_meter.avg))
    return loss_meter.avg
    

def get_reprs(model, data_loader, use_cuda):
    model.eval()
    reprs = []
    for _, (data, _) in enumerate(data_loader):
        if use_cuda:
            data = data.cuda()
        with torch.no_grad():
            # only take the global feature
            feat,_,_,_= model.vit(data,keep_rate=(1,1,1,0.9,1,1,0.9,1,1,0.9,1,1))
            feat = model.mlp_g(feat)
        reprs.append(feat.cpu().data.numpy())
    reprs = np.concatenate(reprs, 0)
    return reprs

def compute_accuracy(pred_labels, true_labels, labels):
    acc_per_class = np.zeros(labels.shape[0])
    for i in range(labels.shape[0]):
        idx = (true_labels == labels[i])
        acc_per_class[i] = np.sum(pred_labels[idx] == true_labels[idx]) / np.sum(idx)
    return np.mean(acc_per_class)

def validation(model, unseen_loader, unseen_labels, attrs_mat, use_cuda, gamma=None):
    # Representation
    with torch.no_grad():
        unseen_reprs = get_reprs(model, unseen_loader, use_cuda)

    # Labels
    uniq_labels = np.unique(unseen_labels))
    updated_unseen_labels = np.searchsorted(uniq_labels, unseen_labels)
    uniq_updated_unseen_labels = np.unique(updated_unseen_labels)
    uniq_updated_labels = np.unique(np.concatenate([updated_unseen_labels]))

    # truncate the attribute matrix
    trunc_attrs_mat = attrs_mat[uniq_labels]
  
    #### ConVLM ####
    convlm_unseen_sim = unseen_reprs @ trunc_attrs_mat[uniq_updated_unseen_labels].T
    pred_labels = np.argmax(convlm_unseen_sim , axis=1)
    convlm_unseen_predict_labels = uniq_updated_unseen_labels[pred_labels]
    convlm_unseen_acc = compute_accuracy(convlm_unseen_predict_labels , updated_unseen_labels, uniq_updated_unseen_labels)
    
    
    

def test(model, test_unseen_loader, test_unseen_labels, attrs_mat, use_cuda, gamma):
    # Representation
    with torch.no_grad():
        unseen_reprs = get_reprs(model, test_unseen_loader, use_cuda)
    # Labels
    uniq_test_unseen_labels = np.unique(test_unseen_labels)

    # ConVLM
    convlm_unseen_sim = unseen_reprs @ attrs_mat[uniq_test_unseen_labels].T
    predict_labels = np.argmax(convlm_unseen_sim , axis=1)
    convlm_unseen_predict_labels = uniq_test_unseen_labels[predict_labels]
    convlm_unseen_acc = compute_accuracy(convlm_unseen_predict_labels , test_unseen_labels, uniq_test_unseen_labels)

    # Calibrated stacking
    Cs_mat = np.zeros(attrs_mat.shape[0])
    Cs_mat[uniq_test_seen_labels] = gamma

       
    return convlm_unseen_acc 