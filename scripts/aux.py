
import __init__
from __init__ import *

import Meta
import pdb
import math

sampling_meta = {
    'samples_per_batch' : 5,
}


def _make_FloatTensor(array):
    if torch.cuda.is_available(): return torch.cuda.FloatTensor(array,device=Meta.DEVICE_ID)
    else:                         return torch.Tensor(array)

def _make_LongTensor(array):
    if torch.cuda.is_available(): return torch.cuda.LongTensor(array,device=Meta.DEVICE_ID)
    else:                         return torch.LongTensor(array)


class SamplesStorage_recover:
    def __init__(self, batch_size, using_speakers):
        self.using_speakers = using_speakers
        self.slices_speakers = self._init_slices_speakers()
        self._batch_size = batch_size

    def add(self, slic, speaker):
        self.slices_speakers[speaker].append(slic)

    def _init_slices_speakers(self):
        ans = {}
        for speaker in self.using_speakers:
            ans[speaker] = []
        return ans

    @property
    def batch_complete(self):
        # if every specs in specs_speakers has length _batch_size
        return all([self._batch_size == len(self.slices_speakers[speaker]) \
            for speaker in self.using_speakers])

    def dump(self):
        ans = self.slices_speakers
        self.slices_speakers = self._init_slices_speakers()
        return ans


class ModelUpdater_autoencode:
    def __init__(self, model, lossF, optimizer, sparse):
        self.model = model
        self.lossF = lossF
        self.optimizer = optimizer
        self._loss = 0
        self._train = False
        self.attentions = [torch.ones(1,1,1,1
                        ).to(Meta.device) for l in range(Meta.model_meta['attend_layers'] - 1)]
        self.attentions.append(torch.ones(1, 1).to(Meta.device))

        self.total_tops = []
        self.sparse = sparse

    def mode(self, train):
        self._train = train
        if train:
            self.model.train()
        else:
            self.model.eval()

    def update(self, batch):
        using_speakers = batch.keys()
        sp_i_s = list(itertools.product( using_speakers, range(len(batch[Meta.data_meta['using_speakers'][0]])) ))
        coch = [spectrogram(batch[sp_i[0]][sp_i[1]]) for sp_i in sp_i_s]
        coch = _make_FloatTensor(coch).view(-1, Meta.data_meta['specgram_size'][0], Meta.data_meta['specgram_size'][1])

        outputs = self.model(coch, self.attentions).view(-1, \
                            Meta.data_meta['specgram_size'][0],\
                            Meta.data_meta['specgram_size'][1])

        loss = self.lossF(outputs, coch)
        self._loss = loss.item()
        # for sampling outputs & inputs
        self.ground_truth = coch[0 : sampling_meta['samples_per_batch']].cpu().detach().numpy()
        self.recover = outputs[0 : sampling_meta['samples_per_batch']].cpu().detach().numpy()

        if self._train:
            # loss = loss \
            #      + self.sparse * torch.mean(F.relu(self.model.x7), dim=(0,1))\
            #      + self.sparse * torch.mean(F.relu(self.model.x5), dim=(0,1,2,3))\
            #      + self.sparse * torch.mean(F.relu(self.model.x4), dim=(0,1,2,3))
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

    def loss(self, mean):
        return self._loss

    def sample(self, batch_count):
        train_or_test = {True:"train", False:"test"}[self._train]
        spb = sampling_meta['samples_per_batch']
        for k in range(spb):
            path_to_sample = os.path.join("../results/phase2/{}".format(train_or_test), "sample_{}".format(batch_count * spb + k))
            utils.dir_utils.mkdir_in( \
                "../results/phase2/{}".format(train_or_test), \
                "sample_{}".format(batch_count * spb + k), \
            )
            json.dump(self.ground_truth[k].tolist(),
                      open(os.path.join(path_to_sample, "ground_truth.json"), "w")
            )
            json.dump(self.recover[k].tolist(),
                      open(os.path.join(path_to_sample, "recover.json"), "w")
            )

def _load_features_of_speakers():
    ans = []
    for speaker in Meta.data_meta['speakers']:
        ans.append(json.load(open("../pickled/features_of_speakers/speaker_{}_feature.json".format(speaker))))

    return _make_FloatTensor(ans)

class ModelUpdater_denoise:
    def __init__(self, anet, aenet, lossF, optimizer):
        self.anet = anet
        self.aenet = aenet
        self.aenet.eval()

        self.lossF = lossF
        self.optimizer = optimizer

        self.speakers_features = _load_features_of_speakers()
        self.attentions_none = [_make_FloatTensor([1]).to(Meta.device) for l in range(Meta.model_meta['attend_layers'])]

        self._train = False

        self.sp1_specs = [None for _ in range(len(Meta.data_meta['using_speakers']) // 2)]
        self.sp2_specs = [None for _ in range(len(Meta.data_meta['using_speakers']) // 2)]
        self.mixed = [None for _ in range(len(Meta.data_meta['using_speakers']) // 2)]
        self.recover_12 = [None for _ in range(len(Meta.data_meta['using_speakers']) // 2)]
        self.recover_21 = [None for _ in range(len(Meta.data_meta['using_speakers']) // 2)]
        self.recover_none = [None for _ in range(len(Meta.data_meta['using_speakers']) // 2)]

    def mode(self, train):
        self._train = train
        if train:
            self.anet.train()
        else:
            self.anet.eval()

    def update(self, batch):
        batch_size = len(batch[Meta.data_meta['using_speakers'][0]])
        # now it's 2-sized batch
        losses = [[],[]]
        for i_pair in range(len(Meta.data_meta['using_speakers']) // 2):
            sp1 = Meta.data_meta['using_speakers'][i_pair]
            sp2 = Meta.data_meta['using_speakers'][len(Meta.data_meta['using_speakers']) - 1 - i_pair]

            features_1 = self.speakers_features[[sp1 for _ in range(batch_size)]]
            attentions_1 = self.anet(features_1)
            del features_1
            features_2 = self.speakers_features[[sp2 for _ in range(batch_size)]]
            attentions_2 = self.anet(features_2)
            del features_2

            sp1_specs = [spectrogram(batch[sp1][i])\
                         for i in range(len(batch[Meta.data_meta['using_speakers'][0]]))]

            sp2_specs = [spectrogram(batch[sp2][i])\
                         for i in range(len(batch[Meta.data_meta['using_speakers'][0]]))]

            mixed = [spectrogram(batch[sp1][i] + batch[sp2][i])\
                     for i in range(len(batch[Meta.data_meta['using_speakers'][0]]))]

            sp1_specs = _make_FloatTensor(sp1_specs)
            sp2_specs = _make_FloatTensor(sp2_specs)
            mixed = _make_FloatTensor(mixed)

            top_12 = self.aenet.upward(mixed, attentions_1)
            recover_12 = self.aenet.downward(top_12).view(-1, 256, 128)
            top_21 = self.aenet.upward(mixed, attentions_2)
            recover_21 = self.aenet.downward(top_21).view(-1, 256, 128)

            spb = sampling_meta['samples_per_batch']
            if not self._train:
                top_none = self.aenet.upward(mixed, self.attentions_none)
                recover_none = self.aenet.downward(top_none).view(-1, 256, 128)
                self.recover_none[i_pair] = recover_none[0 : spb].cpu().detach().numpy()

            loss_12 = self.lossF(recover_12, sp1_specs)
            loss_21 = self.lossF(recover_21, sp2_specs)
            loss = (loss_12 + loss_21) * 0.5
            if self._train:
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            losses[0].append(loss_12.item())
            losses[1].append(loss_21.item())

            # save things for sample() method
            self.sp1_specs[i_pair] = sp1_specs.cpu().detach().numpy()
            self.sp2_specs[i_pair] = sp2_specs.cpu().detach().numpy()
            self.mixed[i_pair] = mixed.cpu().detach().numpy()
            self.recover_12[i_pair] = recover_12.cpu().detach().numpy()
            self.recover_21[i_pair] = recover_21.cpu().detach().numpy()
            if not self._train:
                self.recover_none[i_pair] = recover_none.cpu().detach().numpy()

        self._loss = [np.mean(losses[0]),np.mean(losses[1])]

    def loss(self, mean):
        if mean:
            return np.mean(self._loss)
        else:
            return self._loss

    def sample(self, batch_count):
        # json dump self.ground_truth[k], self.masker[k], self.mixed[k], self.recover[k]
        train_or_test = {True:"train", False:"test"}[self._train]

        spb = len(self.sp1_specs[0])
        for k in range(spb):
            utils.dir_utils.mkdir_in("../results/phase3/{}".format(train_or_test), "sample_{}".format(batch_count * spb + k))
            for i_pair in range(len(Meta.data_meta['using_speakers']) // 2):
                path_to_pair = os.path.join("../results/phase3/{}/sample_{}".format(train_or_test, batch_count * spb + k), "pair_{}".format(i_pair))
                utils.dir_utils.mkdir_in( \
                    "../results/phase3/{}/sample_{}".format(train_or_test, batch_count * spb + k), \
                    "pair_{}".format(i_pair) \
                )
                json.dump(self.sp1_specs[i_pair][k].tolist(), \
                          open(os.path.join(path_to_pair, "sp1.json") , "w") \
                )
                json.dump(self.sp2_specs[i_pair][k].tolist(), \
                          open(os.path.join(path_to_pair, "sp2.json") , "w") \
                )
                json.dump(self.mixed[i_pair][k].tolist(), \
                          open(os.path.join(path_to_pair, "mix.json") , "w") \
                )
                json.dump(self.recover_12[i_pair][k].tolist(), \
                          open(os.path.join(path_to_pair, "re1.json") , "w") \
                )
                json.dump(self.recover_21[i_pair][k].tolist(), \
                          open(os.path.join(path_to_pair, "re2.json") , "w") \
                )
                json.dump(self.recover_none[i_pair][k].tolist(), \
                          open(os.path.join(path_to_pair, "re_none.json") , "w") \
                )

class ModelTopDumper:
    def __init__(self, anet, aenet):
        self.anet = anet
        self.aenet = aenet
        self.anet.eval()
        self.aenet.eval()

        self.speakers_features = _load_features_of_speakers()
        self.attentions_none = [torch.ones(\
                            1, \
                            1, \
                            1, \
                            1
                        ).to(Meta.device) for l in range(Meta.model_meta['attend_layers'] - 1)]
        self.attentions_none.append(torch.ones(\
                            1, \
                            1, \
                        ).to(Meta.device))

        total_pairs = len(Meta.data_meta['using_speakers']) // 2
        self.tops = [{'-1': None, \
                      Meta.data_meta['using_speakers'][ipair]: None,\
                      Meta.data_meta['using_speakers'][total_pairs - 1 - ipair]: None}\
                    for ipair in range(total_pairs)]

        pairs = len(Meta.data_meta['using_speakers']) // 2
        if not os.path.isdir("../results/tops/"): os.mkdir("../results/tops/")
        for ipair in range(pairs):
            sp1 = Meta.data_meta['using_speakers'][ipair]
            sp2 = Meta.data_meta['using_speakers'][pairs*2 - 1 - ipair]
            dirname = os.path.join("../results/", "tops/sp{}-sp{}/".format(sp1, sp2))
            if not os.path.isdir(dirname):
                os.mkdir(dirname)
            else:
                os.system("rm {}".format(os.path.join(dirname, "*")))

    def mode(self, train):
        pass

    def update(self, batch):
        batch_size = len(batch[Meta.data_meta['using_speakers'][0]])
        pairs = len(Meta.data_meta['using_speakers']) // 2
        for i_pair in range(pairs):
            sp1 = Meta.data_meta['using_speakers'][i_pair]
            sp2 = Meta.data_meta['using_speakers'][2*pairs - 1 - i_pair]
            sp1_mix = sp1
            sp2_mix = sp2

            # spectrograms
            sp1_spec = _make_FloatTensor([spectrogram(batch[sp1_mix][i])\
                                       for i in range(batch_size)])
            sp2_spec = _make_FloatTensor([spectrogram(batch[sp2_mix][i])\
                                       for i in range(batch_size)])
            # mixed = _make_FloatTensor([spectrogram(batch[sp1_mix][i] + batch[sp2_mix][i])\
            #                            for i in range(batch_size)])

            # attentions
            features_1 = self.speakers_features[[sp1_mix for _ in range(batch_size)]]
            attentions_1 = self.anet(features_1)

            features_2 = self.speakers_features[[sp2_mix for _ in range(batch_size)]]
            attentions_2 = self.anet(features_2)

            # top distribution
            self.aenet.forward(sp1_spec, attentions_1)
            top_11 = getattr(self.aenet, Meta.dump_meta['dump_layer'])\
                        .cpu().detach().numpy()
            self.aenet.forward(sp1_spec, attentions_2)
            top_12 = getattr(self.aenet, Meta.dump_meta['dump_layer'])\
                        .cpu().detach().numpy()
            self.aenet.forward(sp1_spec, self.attentions_none)
            top_1none = getattr(self.aenet, Meta.dump_meta['dump_layer'])\
                        .cpu().detach().numpy()

            self.aenet.forward(sp2_spec, attentions_1)
            top_21 = getattr(self.aenet, Meta.dump_meta['dump_layer'])\
                        .cpu().detach().numpy()
            self.aenet.forward(sp2_spec, attentions_2)
            top_22 = getattr(self.aenet, Meta.dump_meta['dump_layer'])\
                        .cpu().detach().numpy()
            self.aenet.forward(sp2_spec, self.attentions_none)
            top_2none = getattr(self.aenet, Meta.dump_meta['dump_layer'])\
                        .cpu().detach().numpy()

            self.tops[i_pair][sp1_mix] = [np.mean(top_11, axis=0), np.mean(top_21, axis=0)]
            self.tops[i_pair][sp2_mix] = [np.mean(top_12, axis=0), np.mean(top_22, axis=0)]
            self.tops[i_pair]['-1'] = [np.mean(top_1none, axis=0), np.mean(top_2none, axis=0)]

    def loss(self, mean):
        return -1

    def sample(self, batch_count):
        pairs = len(Meta.data_meta['using_speakers']) // 2

        for ipair in range(pairs):
            sp1 = Meta.data_meta['using_speakers'][ipair]
            sp2 = Meta.data_meta['using_speakers'][2*pairs - 1 - ipair]
            
            json.dump(\
                self.tops[ipair][sp1][0].tolist(), \
                open( os.path.join("../results/", \
                    "tops/sp{}-sp{}/attend{}{}_batch_count{}.json".format(sp1, sp2, sp1, sp1, batch_count)), "w" ) \
            )
            json.dump(\
                self.tops[ipair][sp1][1].tolist(), \
                open( os.path.join("../results/", \
                    "tops/sp{}-sp{}/attend{}{}_batch_count{}.json".format(sp1, sp2, sp2, sp1, batch_count)), "w" ) \
            )

            json.dump(\
                self.tops[ipair][sp2][0].tolist(), \
                open( os.path.join("../results/", \
                    "tops/sp{}-sp{}/attend{}{}_batch_count{}.json".format(sp1, sp2, sp1, sp2, batch_count)), "w" ) \
            )
            json.dump(\
                self.tops[ipair][sp2][1].tolist(), \
                open( os.path.join("../results/", \
                    "tops/sp{}-sp{}/attend{}{}_batch_count{}.json".format(sp1, sp2, sp2, sp2, batch_count)), "w" ) \
            )

            json.dump(\
                self.tops[ipair]['-1'][0].tolist(), \
                open( os.path.join("../results/", \
                    "tops/sp{}-sp{}/attend{}{}_batch_count{}.json".format(sp1, sp2, sp1, "none", batch_count)), "w" )\
            )
            json.dump(\
                self.tops[ipair]['-1'][1].tolist(), \
                open( os.path.join("../results/", \
                    "tops/sp{}-sp{}/attend{}{}_batch_count{}.json".format(sp1, sp2, sp2, "none", batch_count)), "w" )\
            )


def combine_tops_data():
    pairs = len(Meta.data_meta['using_speakers'])//2
    for i_pair in range(pairs):
        sp1 = Meta.data_meta['using_speakers'][i_pair]
        sp2 = Meta.data_meta['using_speakers'][2*pairs - 1 - i_pair]
        sp_pair_dir = os.path.join("../results/tops/sp{}-sp{}/".format(sp1, sp2))
        jsons = os.listdir("../results/tops/sp{}-sp{}/".format(sp1, sp2))

        ## in: sp1
        # attend: sp1
        tops_sp1 = []
        for js in jsons:
            if "attend{}{}".format(sp1, sp1) in js:
                tops = json.load(open(os.path.join(sp_pair_dir, js)))
                tops_sp1.append(tops)
        tops_sp1 = np.mean(tops_sp1, axis=0).tolist()
        json.dump(tops_sp1, open(os.path.join(sp_pair_dir, "tops_{}_{}.json".format(sp1, sp1)), "w"))

        # attend: sp2
        tops_sp2 = []
        for js in jsons:
            if "attend{}{}".format(sp1, sp2) in js:
                tops = json.load(open(os.path.join(sp_pair_dir, js)))
                tops_sp2.append(tops)
        tops_sp2 = np.mean(tops_sp2, axis=0).tolist()
        json.dump(tops_sp2, open(os.path.join(sp_pair_dir, "tops_{}_{}.json".format(sp1, sp2)), "w"))

        # attend: none
        tops_none = []
        for js in jsons:
            if "attend{}none".format(sp1) in js:
                tops = json.load(open(os.path.join(sp_pair_dir, js)))
                tops_none.append(tops)
        tops_none = np.mean(tops_none, axis=0).tolist()
        json.dump(tops_none, open(os.path.join(sp_pair_dir, "tops_{}_none.json".format(sp1)), "w"))

        ## in: sp2
        # attend: sp1
        tops_sp1 = []
        for js in jsons:
            if "attend{}{}".format(sp2, sp1) in js:
                tops = json.load(open(os.path.join(sp_pair_dir, js)))
                tops_sp1.append(tops)
        tops_sp1 = np.mean(tops_sp1, axis=0).tolist()
        json.dump(tops_sp1, open(os.path.join(sp_pair_dir, "tops_{}_{}.json".format(sp2, sp1)), "w"))

        # attend: sp2
        tops_sp2 = []
        for js in jsons:
            if "attend{}{}".format(sp2, sp2) in js:
                tops = json.load(open(os.path.join(sp_pair_dir, js)))
                tops_sp2.append(tops)
        tops_sp2 = np.mean(tops_sp2, axis=0).tolist()
        json.dump(tops_sp2, open(os.path.join(sp_pair_dir, "tops_{}_{}.json".format(sp2, sp2)), "w"))

        # attend: none
        tops_none = []
        for js in jsons:
            if "attend{}none".format(sp2) in js:
                tops = json.load(open(os.path.join(sp_pair_dir, js)))
                tops_none.append(tops)
        tops_none = np.mean(tops_none, axis=0).tolist()
        json.dump(tops_none, open(os.path.join(sp_pair_dir, "tops_{}_none.json".format(sp2)), "w"))


def Corr(a,b):
    a = a - np.mean(a)
    b = b - np.mean(b)
    r = (a*b).sum() / math.sqrt((a*a).sum() * (b*b).sum())
    return r

class LayerAMIDumper:
    def __init__(self, anet, aenet):
        self.anet = anet
        self.aenet = aenet
        self.anet.eval()
        self.aenet.eval()

        self.speakers_features = _load_features_of_speakers()

        total_pairs = len(Meta.data_meta['using_speakers']) // 2

        pairs = len(Meta.data_meta['using_speakers']) // 2
        if not os.path.isdir("../results/tops/"): os.mkdir("../results/tops/")
        for ipair in range(pairs):
            sp1 = Meta.data_meta['using_speakers'][ipair]
            sp2 = Meta.data_meta['using_speakers'][pairs*2 - 1 - ipair]
            dirname = os.path.join("../results/", "tops/sp{}-sp{}/".format(sp1, sp2))
            if not os.path.isdir(dirname):
                os.mkdir(dirname)
            else:
                os.system("rm {}".format(os.path.join(dirname, "amis.json")))

        self.amis_layers = [{'x2':[], 'x3':[], 'x4':[], 'x5':[], 'x6':[], 'x7':[]} \
                            for _ in range(pairs)]

    def mode(self, train):
        pass

    def update(self, batch):
        batch_size = len(batch[Meta.data_meta['using_speakers'][0]])
        pairs = len(Meta.data_meta['using_speakers']) // 2
        attentions_speakers = self.anet(self.speakers_features)
        for i_pair in range(pairs):
            sp1 = Meta.data_meta['using_speakers'][i_pair]
            sp2 = Meta.data_meta['using_speakers'][2*pairs - 1 - i_pair]

            # spectrograms
            sp1_specs = _make_FloatTensor([spectrogram(batch[sp1][i])\
                                     for i in range(batch_size)])
            sp2_specs = _make_FloatTensor([spectrogram(batch[sp2][i])\
                                     for i in range(batch_size)])
            mixed     = _make_FloatTensor([spectrogram(batch[sp1][i] + batch[sp2][i])\
                                     for i in range(batch_size)])

            # attentions
            features_1 = self.speakers_features[[sp1 for _ in range(batch_size)]]
            attentions_1 = self.anet(features_1)
            del features_1

            features_2 = self.speakers_features[[sp2 for _ in range(batch_size)]]
            attentions_2 = self.anet(features_2)
            del features_2

            # layer distribution
            for layer in ['x2','x3','x4','x5','x6','x7']:
                self.aenet.upward(sp1_specs, attentions_1)
                layer_clean_1 = getattr(self.aenet, layer).cpu().detach().numpy()
                self.aenet.upward(sp2_specs, attentions_2)
                layer_clean_2 = getattr(self.aenet, layer).cpu().detach().numpy()
                self.aenet.upward(mixed, attentions_1)
                layer_mixed_1 = getattr(self.aenet, layer).cpu().detach().numpy()
                self.aenet.upward(mixed, attentions_2)
                layer_mixed_2 = getattr(self.aenet, layer).cpu().detach().numpy()

                amis = [Corr(layer_clean_1[batch], layer_mixed_1[batch]) + Corr(layer_clean_2[batch], layer_mixed_2[batch])\
                      - Corr(layer_clean_1[batch], layer_mixed_2[batch]) - Corr(layer_clean_2[batch], layer_mixed_1[batch]) for batch in range(batch_size)]

                self.amis_layers[i_pair][layer].extend(amis)

    def loss(self, mean):
        return -1

    def sample(self, batch_count):
        pairs = len(Meta.data_meta['using_speakers']) // 2

        for ipair in range(pairs):
            sp1 = Meta.data_meta['using_speakers'][ipair]
            sp2 = Meta.data_meta['using_speakers'][2*pairs - 1 - ipair]
            json.dump(\
                self.amis_layers[ipair], \
                open( os.path.join("../results/", \
                    "tops/sp{}-sp{}/amis.json".format(sp1, sp2)), "w" )\
            )
