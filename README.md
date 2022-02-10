### Readme

---

#### Project description

This project is code directory for our work on cocktail party problem. The project trains and tests an auto-encoder model that is able to separate one spectrogram component from a 'cocktail party' mixture (this is called cocktail party effect). Structurally, the model contains a ResNet-based autoencoder, with multi-layered attention signal projecting on it.

For our project, we are using data from pairs of speakers in LibriVox dataset. See links below: .

To prepare dataset, please download from the links, then cut out 30 min long audios from each person's recording. Then, cut 1 min long segments and name them `dataset/sp?/?.wav`. Or, you could download from google drive: https://drive.google.com/file/d/1Zv7F5OaqLI_ADC5ez5-TQEAmJnUiVBat/view?usp=sharing, unzip and place the unzipped directory named `dataset` in project home directory.

#### Running tips

Training code are all integrated in `scripts/supervised.py`. To train everything from scratch, just `cd scripts/` and `python supervised.py`.

If you would like to train autoencoder (ResAE) and denoising (AttentionNet) separately, read the following.

##### SpeechEmbedder

For SpeechEmbedder, we trained the model described in https://github.com/HarryVolek/PyTorch_Speaker_Verification. If you would like to re-train this speaker embedder model, go change the key `re_dump_embeddings` to `True` and delete all entries in `pickled/features_of_speakers/`, and come back to run `scripts/supervised.py`.

Or, you could directly use embedding dumps provided already in our project. 

##### Autoencoder

Configuration of training plan is hard-coded in `scripts/supervised.py`. To train an autoencoder, you would like open  `scripts/supervised.py`, then focus on section `2` in configuration variable `train_meta (type 'dict')`.

Key `'reuse'` has three legitimate values:` 0`,`1` and `2`. `0` tells the program to train everything from scratch, `1` tells the program to load checkpoint and continue training, and `2` skips this training session.

Key `'test_before_continue'` tells the program whether test checkpoint model before training begins or not.

##### Cocktail party

Similarly, training plan associated with `AttentionNet` training is specified in section `3`. if you would like to skip this section, change key `'reuse'` to `2`. To do everything from scratch, change `'reuse'` to `0`. To train from previous checkpoint, change it to `1`.

##### Changing model meta-parameters

All model meta-parameter information are configured in `Meta.py`. Two parts of configuration are coded here: `data_meta` and `model_meta`.

`data_meta`: Choose your interested pair in `'using_speakers'`.

`model_meta`: Here, you have the opportunity to disable attentions at layers. This is useful when plotting figures for the model under different attending scenarios. Specifically, edit `attend_layers` for different model depths (`4`, `6`, `9`) and `layer_attentions` (a sequence of 0 and 1, specifying whether or not the program projects attention signal onto each level in encoder).

##### Figure 2

Obtain figure 2 by running `scripts/supervised.py`. Find the "speaker1, speaker2, mixture, mixture-recovered, speaker1-recovered, speaker2-recovered" hexa-tuple in directory `results/phase3/test/sample_?/` (for cocktail party task). Check up `results/phase2/test/sample_?/` for spectrogram task performance.

##### Figure 3

###### AMI scatter and histogram

To run this part, first run `scripts/multifunction_dump.py`. Be sure to first set variables `cumu` to `True`, `topdown_scan` to `True` and `Type` to `'spec'`.

Go to file `scripts/scatter.py` and uncomment `figure3` part. Then switch to `scripts/` and run `python scatter.py cumu`. This will give you `results/figures/ami_hist_6.png` and `results/figures/ami_scatter_6_mark_legend.png`.

###### AMI barplot

To run this part, first run `scripts/multifunction_dump.py`. Be sure to first set variables `cumu` to `False`, `topdown_scan` to `True` and `Type` to `'spec'`.

Go to file `scripts/barplot.py` and uncomment `figure3` part. Then run `python barplot.py`. This will give you `results/bar.png`.

##### Figure 4

###### AMI scatters

To run this part, first run `scripts/multifunction_dump.py`. Be sure to first set variables `cumu` to `True`, `topdown_scan` to `True` and `Type` to `'spec'`.

Go to file `scripts/scatter.py` and uncomment `figure4/5` part. Then switch to `scripts/` and run `python scatter.py cumu`. This will give you `ami_scatter_*.png` under directory `results/figures/[model metaparameters]/` and `ami_scatter_*_legend.png` under directory `results/figures/[model metaparamters]`.

###### AMI barplots

To run this part, first run `scripts/multifunction_dump.py`. Be sure to first set variables `cumu` to `True`, `topdown_scan` to `True` and `Type` to `'spec'`.

Go to file `scripts/barplot.py` and uncomment `figure4,5` part. Edit `cumu_single` to `'cumu'`. Then switch to `scripts/` and run `python barplot.py`. This will give you `results/barplot_1x_cumu.png` and `results/barplot_2x_cumu.png`.

##### Figure 5

###### AMI scatters

To run this part, first run `scripts/multifunction_dump.py`. Be sure to first set variables `cumu` to `False`, `topdown_scan` to `True` and `Type` to `'spec'`.

Go to file `scripts/scatter.py` and uncomment `figure4/5` part. Then switch to `scripts/` and run `python scatter.py cumu`. This will give you `ami_scatter_*.png` under directory `results/figures/[model metaparameters]/` and `ami_scatter_*_legend.png` under directory `results/figures/[model metaparamters]`.

###### AMI barplots

To run this part, first run `scripts/multifunction_dump.py`. Be sure to first set variables `cumu` to `False`, `topdown_scan` to `True` and `Type` to `'spec'`.

Go to file `scripts/barplot.py` and uncomment `figure4,5` part. Edit `cumu_single` to `'single'`. Then switch to `scripts/` and run `python barplot.py`. This will give you `results/barplot_1x_single.png` and `results/barplot_2x_single.png`.

###### Figure 6

- cumulative projection

- single layer projection



#### Required libraries

- python 3.7.1
- librosa 0.7.1
- numpy 1.21.4
- matplotlib 3.1.1
- pytorch 1.0.1.post2
- scipy 1.7.3
