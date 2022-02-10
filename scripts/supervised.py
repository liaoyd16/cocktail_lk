import __init__
from __init__ import *

import Meta
from models.SpeechEmbedder import SpeechEmbedder
if Meta.model_meta['attend_layers'] + 1 == 5:
    from models.AttentionNet5 import AttentionNet
    from models.ResAE5 import ResAE
elif Meta.model_meta['attend_layers'] + 1 == 7:
    from models.AttentionNet import AttentionNet
    from models.ResAE import ResAE
elif Meta.model_meta['attend_layers'] + 1 == 10:
    from models.AttentionNet10 import AttentionNet
    from models.ResAE10 import ResAE

from utils.logger import Logger
from utils.skip import *

from aux import *

train_meta = {
    'model': {
        'keepsize_num_en': 2,
        'keepsize_num_de': 2,
        'shortcut': {5:  [True, True, True, True], 
                     7:  [True, True, True, True, True, True],
                     10: [True, True, True, True, True, True, True, True, True]},
    },
    're_dump_embeddings': False,
    'do_sample': False,
    '2' : {
        'lr': 1e-4,
        'reuse': 0,
        'test_before_continue': False,
        'epochs':5,
        'batch': {
            'train': 10,
            'test': 10,
        },
        'lossF': nn.MSELoss(),
        'sparse': 0,
        'batch_count': 0,
        'test_batch_count': 0,
        'loss_test': 0,
        'optimizer': 'adam',
    },
    '3' : {
        'lr': 1e-4,
        'reuse': 0,
        'test_before_continue': False,
        'epochs':5,
        'batch': {
            'train': 5,
            'test': 5,
        },
        'lossF': nn.MSELoss(),
        'image_batch_count': 0,
        'batch_count': 0,
        'test_batch_count': 0,
        'loss_test': 0,
        'optimizer': 'adam',
    },
}

def _iterate_and_update_(model_updater, samples_storage, logger, phase_no, train, using_speakers):
    if skip(): return

    if train:
        blocks = copy.copy(Meta.data_meta['train_blocks_per_speaker'])
        np.random.shuffle(blocks)
    else:     blocks = Meta.data_meta['test_blocks_per_speaker']

    model_updater.mode(train)
    if not train:
        loss_test = []

    for block_no in blocks:
        xs_speaker = []
        for speaker in Meta.data_meta['speakers']:
            _, x = wavfile.read(Meta.audio_name_(speaker, block_no))
            xs_speaker.append(x)

        slice_len = Meta.data_meta['slice_len']
        for i_slice in range(Meta.data_meta['slices_per_block']):
            for speaker in using_speakers:
                slic = xs_speaker[speaker][i_slice*slice_len : (i_slice+1)*slice_len]

                samples_storage.add(slic, speaker)

                if samples_storage.batch_complete:
                    model_updater.update(samples_storage.dump())
                    if train:
                        logger.summary('loss{}_train'.format(phase_no), model_updater.loss(mean=True), train_meta[phase_no]['batch_count'])
                        print("\tstep #{}, {} = {}".format(train_meta[phase_no]['batch_count'],\
                                                           'loss{}_train'.format(phase_no), \
                                                           model_updater.loss(mean=False)))
                        train_meta[phase_no]['batch_count'] += 1
                    else: # test
                        print("test batch #{}".format(train_meta[phase_no]['test_batch_count']))
                        loss_test.append(model_updater.loss(mean=True))
                        if train_meta['do_sample']:
                            model_updater.sample(train_meta[phase_no]['test_batch_count'])
                        train_meta[phase_no]['test_batch_count'] += 1

                    if skip(): return

    if not train:
        train_meta[phase_no]['loss_test'] = np.mean(loss_test)
        print("test loss = {}".format(train_meta[phase_no]['loss_test']))
        train_meta[phase_no]['image_batch_count'] = 0
        train_meta[phase_no]['test_batch_count']  = 0

def dump_features(fake):
    if fake:
        features_dim_256 = gen_feature_vectors(num=len(Meta.data_meta['speakers']), dim=256)
        for sp in Meta.data_meta['speakers']:
            json.dump(features_dim_256[sp].tolist(), \
                      open(os.path.join(Meta.PROJ_ROOT,\
                                        "pickled/features_of_speakers/",\
                                        "speaker_{}_feature.json".format(sp)),\
                      "w"))
    else:
        fnet = SpeechEmbedder()
        fnet.load_state_dict(torch.load("../pickled/fnet.model"))
        slices = []
        for sp in Meta.data_meta['speakers']:
            _, x = wavfile.read(Meta.audio_name_(sp, 29))
            slices.append(x)

        embeddings = fnet.embed(slices)

        for sp in Meta.data_meta['speakers']:
            json.dump(embeddings[sp].tolist(), \
                      open(os.path.join(Meta.PROJ_ROOT,\
                                        "pickled/features_of_speakers/",\
                                        "speaker_{}_feature.json".format(sp)),\
                      "w"))

if __name__ == '__main__':
    NUM_LAYERS = Meta.model_meta['attend_layers'] + 1

    # load model
    aeNet = ResAE(train_meta['model']['keepsize_num_en'], \
                  train_meta['model']['keepsize_num_de'], \
                  train_meta['model']['shortcut'][NUM_LAYERS])
    aNet = AttentionNet()

    hyper_param = "n={}_en={}_resblock_3x3conv_5shortcut".format(Meta.model_meta['attend_layers']+1,\
                                      train_meta['model']['keepsize_num_en'])
    if not os.path.isdir("../pickled/{}/".format(hyper_param)): os.mkdir("../pickled/{}/".format(hyper_param))

    if train_meta['model']['keepsize_num_de']==0: hyper_param += "/de=0"
    else: hyper_param += "/de=en"

    if not os.path.isdir("../pickled/{}/".format(hyper_param)): os.mkdir("../pickled/{}/".format(hyper_param))    

    if train_meta['3']['reuse'] > 0:
        aeNet = torch.load("../pickled/{}/aeNet_3.pickle".format(hyper_param), map_location=torch.device(Meta.DEVICE_ID))
        aNet = torch.load("../pickled/{}/aNet.pickle".format(hyper_param), map_location=torch.device(Meta.DEVICE_ID))
    elif train_meta['2']['reuse'] > 0:
        aeNet = torch.load("../pickled/{}/aeNet_2.pickle".format(hyper_param), map_location=torch.device(Meta.DEVICE_ID))
    assert(aeNet.shortcut == train_meta['model']['shortcut'][NUM_LAYERS])

    aeNet.to(Meta.device)
    aNet.to(Meta.device)

    # directory stuff
    logger = Logger("./log")

    all_speakers = np.array(Meta.data_meta['speakers'])

    # fake features
    clean_log_dir()
    if  train_meta['re_dump_embeddings'] \
     and "speaker_0_feature.json" not in os.listdir("../pickled/features_of_speakers/"):
        dump_features(Meta.model_meta['fake_features'])

    # phase 2: autoencode
    if train_meta['2']['optimizer']=='adam':
        optimizer = torch.optim.Adam(aeNet.parameters(), lr=train_meta['2']['lr'])
    else:
        optimizer = torch.optim.SGD(aeNet.parameters(), lr=train_meta['2']['lr'], momentum=0.9)
    if train_meta['2']['reuse'] < 2:
        samples_storage = SamplesStorage_recover(train_meta['2']['batch']['train'], Meta.data_meta['speakers'])
        model_updater = ModelUpdater_autoencode(aeNet, train_meta['2']['lossF'], optimizer, train_meta['2']['sparse'])
        clean_results_in_(2, "test")
        clean_results_in_(2, "train")
        for epo in range(train_meta['2']['epochs']):
            print("phase 2, epoch {}".format(epo))
            _iterate_and_update_(model_updater, samples_storage, logger, phase_no='2', train=True, using_speakers=all_speakers)
            torch.cuda.empty_cache()
            _iterate_and_update_(model_updater, samples_storage, logger, phase_no='2', train=False, using_speakers=all_speakers)
            if skip():
                reset_skip()
                break
        torch.save(aeNet, "../pickled/{}/aeNet_2.pickle".format(hyper_param))
    elif train_meta['2']['test_before_continue']:
        samples_storage = SamplesStorage_recover(train_meta['2']['batch']['test'], Meta.data_meta['speakers'])
        model_updater = ModelUpdater_autoencode(aeNet, train_meta['2']['lossF'], optimizer, train_meta['2']['sparse'])
        _iterate_and_update_(model_updater, samples_storage, logger, phase_no='2', train=False, using_speakers=all_speakers)


    # phase 3: attended denoise
    parameters = list(aNet.parameters())
    parameters.extend(aeNet.parameters())
    if train_meta['3']['optimizer']=='adam':
        optimizer = torch.optim.Adam(parameters, lr=train_meta['3']['lr'])
    else:
        optimizer = torch.optim.SGD(parameters, lr=train_meta['3']['lr'], momentum=0.9)

    if train_meta['3']['reuse'] < 2:
        samples_storage = SamplesStorage_recover(train_meta['3']['batch']['train'], Meta.data_meta['using_speakers'])
        model_updater = ModelUpdater_denoise(aNet, aeNet, train_meta['3']['lossF'], optimizer)
        clean_results_in_(3, "test")
        clean_results_in_(3, "train")
        for epo in range(train_meta['3']['epochs']):
            print("phase 3, epoch {}".format(epo))
            _iterate_and_update_(model_updater, samples_storage, logger, phase_no='3', train=True, using_speakers=Meta.data_meta['using_speakers'])
            torch.cuda.empty_cache()
            _iterate_and_update_(model_updater, samples_storage, logger, phase_no='3', train=False, using_speakers=Meta.data_meta['using_speakers'])
            if skip():
                reset_skip()
                break
        torch.save(aNet, "../pickled/{}/aNet.pickle".format(hyper_param))
        torch.save(aeNet, "../pickled/{}/aeNet_3.pickle".format(hyper_param))
    elif train_meta['3']['test_before_continue']:
        samples_storage = SamplesStorage_recover(train_meta['3']['batch']['test'], Meta.data_meta['using_speakers'])
        model_updater = ModelUpdater_denoise(aNet, aeNet, train_meta['3']['lossF'], optimizer)
        _iterate_and_update_(model_updater, samples_storage, logger, phase_no='3', train=False, using_speakers=Meta.data_meta['using_speakers'])
