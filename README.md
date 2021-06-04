# DiffSinger - PyTorch Implementation

PyTorch implementation of [DiffSinger: Diffusion Acoustic Model for Singing Voice Synthesis](https://arxiv.org/abs/2105.02446) (TTS Extension).

<p align="center">
    <img src="img/model_1.png" width="80%">
</p>

<p align="center">
    <img src="img/model_2.png" width="80%">
</p>

# Status (2021.06.03)
- [x] Naive Version of DiffSinger
- [ ] Shallow Diffusion Mechanism: Training boundary predictor by leveraging pre-trained auxiliary decoder + Training denoiser using `k` as a maximum time step

# Quickstart

## Dependencies
You can install the Python dependencies with
```
pip3 install -r requirements.txt
```

## Inference

You have to download the [pretrained models](https://drive.google.com/drive/folders/1BBuaoSlInwFoUt1PKLxo0Sjl5qWCq945?usp=sharing) and put them in ``output/ckpt/LJSpeech/``.

For English single-speaker TTS, run
```
python3 synthesize.py --text "YOUR_DESIRED_TEXT" --restore_step 160000 --mode single -p config/LJSpeech/preprocess.yaml -m config/LJSpeech/model.yaml -t config/LJSpeech/train.yaml
```
The generated utterances will be put in ``output/result/``.


## Batch Inference
Batch inference is also supported, try

```
python3 synthesize.py --source preprocessed_data/LJSpeech/val.txt --restore_step 160000 --mode batch -p config/LJSpeech/preprocess.yaml -m config/LJSpeech/model.yaml -t config/LJSpeech/train.yaml
```
to synthesize all utterances in ``preprocessed_data/LJSpeech/val.txt``

## Controllability
The pitch/volume/speaking rate of the synthesized utterances can be controlled by specifying the desired pitch/energy/duration ratios.
For example, one can increase the speaking rate by 20 % and decrease the volume by 20 % by

```
python3 synthesize.py --text "YOUR_DESIRED_TEXT" --restore_step 160000 --mode single -p config/LJSpeech/preprocess.yaml -m config/LJSpeech/model.yaml -t config/LJSpeech/train.yaml --duration_control 0.8 --energy_control 0.8
```

# Training

## Datasets

The supported datasets are

- [LJSpeech](https://keithito.com/LJ-Speech-Dataset/): a single-speaker English dataset consists of 13100 short audio clips of a female speaker reading passages from 7 non-fiction books, approximately 24 hours in total.
- (will be added more)

## Preprocessing
 
First, run 
```
python3 prepare_align.py config/LJSpeech/preprocess.yaml
```
for some preparations.

As described in the paper, [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/en/latest/) (MFA) is used to obtain the alignments between the utterances and the phoneme sequences.
Alignments for the LJSpeech datasets are provided [here](https://drive.google.com/drive/folders/1DBRkALpPd6FL9gjHMmMEdHODmkgNIIK4?usp=sharing) from [ming024's FastSpeech2](https://github.com/ming024/FastSpeech2).
You have to unzip the files in ``preprocessed_data/LJSpeech/TextGrid/``.

After that, run the preprocessing script by
```
python3 preprocess.py config/LJSpeech/preprocess.yaml
```

Alternately, you can align the corpus by yourself. 
Download the official MFA package and run
```
./montreal-forced-aligner/bin/mfa_align raw_data/LJSpeech/ lexicon/librispeech-lexicon.txt english preprocessed_data/LJSpeech
```
or
```
./montreal-forced-aligner/bin/mfa_train_and_align raw_data/LJSpeech/ lexicon/librispeech-lexicon.txt preprocessed_data/LJSpeech
```

to align the corpus and then run the preprocessing script.
```
python3 preprocess.py config/LJSpeech/preprocess.yaml
```

## Training

Train your model with
```
python3 train.py -p config/LJSpeech/preprocess.yaml -m config/LJSpeech/model.yaml -t config/LJSpeech/train.yaml
```

# TensorBoard

Use
```
tensorboard --logdir output/log/LJSpeech
```

to serve TensorBoard on your localhost.
The loss curves, synthesized mel-spectrograms, and audios are shown.

![](./img/tensorboard_loss.png)
![](./img/tensorboard_spec.png)
![](./img/tensorboard_audio.png)

# Implementation Issues

1. **Pitch extractor comparison (on LJ001-0006.wav)**

    <p align="center">
        <img src="img/pitch_extractor_comparison.png" width="100%">
    </p>

    **pyworld** is used to extract f0 (fundamental frequency) as pitch information in this implementation. Empirically, however, I found that all three methods were equally acceptable for clean datasets (e.g., LJSpeech) as above figures. Note that **pysptk** would work better for noisy datasets (as described in [STYLER](https://github.com/keonlee9420/STYLER)).

2. Stack two layers of `FFTBlock` for the lyrics encoder (text encoder).
3. (Naive version) The number of learnable parameters is `34.337M`, which is larger than the original paper (`26.744M`). The `diffusion` module takes a significant portion of whole parameters.
4. I did not remove the energy prediction of FastSpeech2 since it is not critical to the model training or performance (as described in [LightSpeech](https://arxiv.org/abs/2102.04040)). It should be easily removed without any performance degradation.
5. Use **HiFi-GAN** instead of **Parallel WaveGAN (PWG)** for vocoding.

# Citation

```
@misc{lee2021diffsinger,
  author = {Lee, Keon},
  title = {DiffSinger},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/keonlee9420/DiffSinger}}
}
```

# References
- Authors' codebase
- [ming024's FastSpeech2](https://github.com/ming024/FastSpeech2) (Later than 2021.02.26 ver.)
- [hojonathanho's diffusion](https://github.com/hojonathanho/diffusion)
- [lmnt-com's diffwave](https://github.com/lmnt-com/diffwave)
