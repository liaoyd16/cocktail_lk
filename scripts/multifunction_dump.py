import __init__
from __init__ import *
import matplotlib
# matplotlib.use("agg")
import matplotlib.pyplot as plt
import Meta
import json
import torch
import math


# 返回所有的 attention分布 - probe层 的pair
import copy
cumu = True ##
topdown_scan = True ##
def attentions_and_probe():
    conditions = []
    n_attentions = Meta.model_meta['attend_layers']

    layer_attentions = [0 for _ in range(n_attentions)]
    conditions.append((copy.copy(layer_attentions), ''))
    for iatt in range(n_attentions):
        if topdown_scan:
            layer_attentions[n_attentions - 1 - iatt] = 1
            if cumu == False:
                layer_attentions[(n_attentions - iatt)%n_attentions] = 0 # cancel previous layer attention, this works even when iatt==0
        else:
            layer_attentions[iatt] = 1
            if cumu == False:
                layer_attentions[iatt - 1] = 0 # cancel previous layer attention, this works even when iatt==0
        
        conditions.append((copy.copy(layer_attentions), ''))

    return conditions

Type = ['activ', 'spec_ami', 'activ_ami', 'spec'][3] ##


def _make_FloatTensor(array):
    if torch.cuda.is_available(): return torch.cuda.FloatTensor(array,device=Meta.DEVICE_ID)
    else:                         return torch.Tensor(array)

def _load_features_of_speakers():
    ans = []
    for speaker in Meta.data_meta['speakers']:
        ans.append(json.load(open("../pickled/features_of_speakers/speaker_{}_feature.json".format(speaker))))
    return _make_FloatTensor(ans)

def Corr(a,b):
    a = a - np.mean(a)
    b = b - np.mean(b)
    r = (a*b).sum() / math.sqrt((a*a).sum() * (b*b).sum())
    return r

def AMI(sp1, sp2, re1, re2):
    return Corr(sp1, re1) + Corr(sp2, re2) - Corr(sp1, re2) - Corr(sp2, re1)

def _load_batch(b, speakers, bs):
    # clips: [b * bs : (b+1) * bs]
    # time : [b * bs : (b+1) * bs] * 0.5s
    # 
    block_no = (b * bs) // 120 + 27
    offset   = (b * bs) %  120
    Fs_2 = 22050 // 2
    batch = [[], []]
    for i_speaker in range(2):
        _, x = wavfile.read(Meta.audio_name_(speakers[i_speaker], block_no))
        for s in range(bs):
            batch[i_speaker].append( x[(offset + s) * Fs_2 : (offset + s + 1) * Fs_2] )
    return batch

def do_dump(nlyr, en, de, aNet, aeNet, two_speakers, conditions):
    print("n = {}, en = {}, de = {}".format(nlyr, en, de))
    feature_vectors = _load_features_of_speakers()[two_speakers]

    for icondition, condition in zip(np.arange(len(conditions)), conditions):
        print("\t", condition[0])
        if Type == "activ":
            results = [[], []]
        elif Type == "spec":
            results = [[], [], [], []]
        else:
            results = []

        Meta.model_meta['layer_attentions'] = condition[0]
        probe_layer = condition[1]
        for b in range(BATCHES):
            print("\t\tfeed batch #{}".format(b))
            batch = _load_batch(b, speakers=two_speakers, bs=BS) # speakers x BS - sized
            sp1   = _make_FloatTensor([spectrogram(batch[0][b]) for b in range(BS)])
            sp2   = _make_FloatTensor([spectrogram(batch[1][b]) for b in range(BS)])
            mixed = _make_FloatTensor([spectrogram(batch[0][b] + batch[1][b]) for b in range(BS)])

            batch_results = []
            for i in range(2):
                attentions = aNet.forward(\
                            torch.cat([feature_vectors[i].view(1, 256) for _ in range(BS)], dim=0))
                spec = aeNet.forward(mixed, attentions).cpu().detach().numpy()
                del attentions
                
                if Type == "activ":
                    batch_results.append(getattr(aeNet, probe_layer).cpu().detach().numpy().reshape(BS, -1))
                elif Type == "spec_ami" or Type == "spec":
                    batch_results.append(spec)
                elif Type == "activ_ami":
                    batch_results.append(getattr(aeNet, probe_layer).cpu().detach().numpy().reshape(BS, -1))

            if Type == "activ":
                # print(batch_results[0].shape, batch_results[1].shape)
                results[0].append(batch_results[0])
                results[1].append(batch_results[1])
            elif Type == "spec":
                results[0].append(batch_results[0])
                results[1].append(batch_results[1])
                results[2].append(sp1.cpu().detach().numpy())
                results[3].append(sp2.cpu().detach().numpy())
            elif Type == "spec_ami":
                batch_results = [AMI(sp1[i].cpu().detach().numpy(), sp2[i].cpu().detach().numpy(), \
                                     batch_results[0][i], batch_results[1][i]) for i in range(BS)]
                results.append(batch_results)
            elif Type == "activ_ami":
                attentions_none = [1 for _ in range(6)]
                aeNet.forward(sp1, attentions_none)
                sp1_repr = getattr(aeNet, probe_layer).cpu().detach().numpy().reshape(BS, -1)
                aeNet.forward(sp2, attentions_none)
                sp2_repr = getattr(aeNet, probe_layer).cpu().detach().numpy().reshape(BS, -1)
                batch_results = [AMI(sp1_repr[i], sp2_repr[i], \
                                     batch_results[0][i], batch_results[1][i]) for i in range(BS)]
                results.append(batch_results)

        if Type == "activ":
            for elem in results[0]: print(elem.shape)
            part1 = np.concatenate(results[0], axis=0).tolist()
            part2 = np.concatenate(results[1], axis=0).tolist()
            results = [part1, part2]
        elif Type == "spec":
            p1 = np.concatenate(results[0], axis=0).tolist()
            p2 = np.concatenate(results[1], axis=0).tolist()
            p3 = np.concatenate(results[2], axis=0).tolist()
            p4 = np.concatenate(results[3], axis=0).tolist()
            results = [p1, p2, p3, p4]
        else:
            results = np.concatenate(results, axis=0).tolist()

        if not os.path.isdir("../results/dumped/n={}_en={}_resblock_3x3conv_5shortcut/".format(nlyr, en)):
            os.mkdir("../results/dumped/n={}_en={}_resblock_3x3conv_5shortcut/".format(nlyr, en))
        if not os.path.isdir("../results/dumped/n={}_en={}_resblock_3x3conv_5shortcut/de={}/".format(nlyr, en, de)):
            os.mkdir("../results/dumped/n={}_en={}_resblock_3x3conv_5shortcut/de={}/".format(nlyr, en, de))

        cumu_str = {True:"cumu", False:"single"}[cumu]
        topdown_str = {True:"tpdwn", False:"botup"}[topdown_scan]
        json.dump(results, open("../results/dumped/n={}_en={}_resblock_3x3conv_5shortcut/de={}/condition_{}_{}_{}.json".format(\
                                                    nlyr, en, de, cumu_str+"-"+topdown_str, icondition, Type), 'w'))


if __name__ == '__main__':
#     two_speakers = [1,3]
    two_speakers = Meta.data_meta['using_speakers'][:2]
    BS = 20
    BATCHES = 60 * 2 * 2 // BS # audio 27,28, each 60s, 0.5s/slice

    nlyr = Meta.model_meta['attend_layers'] + 1
    en = 3 ##
    de = 'en'##

    aeNet = torch.load("../pickled/n={}_en={}_resblock_3x3conv_5shortcut/de={}/aeNet_3.pickle".format(nlyr, en, de), map_location=torch.device(Meta.DEVICE_ID))
    aNet = torch.load("../pickled/n={}_en={}_resblock_3x3conv_5shortcut/de={}/aNet.pickle".format(nlyr, en, de), map_location=torch.device(Meta.DEVICE_ID))
    aeNet.to(Meta.device)
    aNet.to(Meta.device)

    conditions = attentions_and_probe()
    do_dump(nlyr, en, de, aNet, aeNet, two_speakers, conditions)
