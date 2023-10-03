import os
import random
from typing import Tuple
from pathlib import Path

import torch
import numpy as np
from torchvision.transforms import Compose
from torchvision import transforms as tv_transforms
from torchaudio.datasets.speechcommands import SPEECHCOMMANDS, HASH_DIVIDER, EXCEPT_FOLDER, _load_list
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchaudio import transforms, load
from torchaudio.compliance.kaldi import fbank

from collections import defaultdict
import flwr #! DON'T REMOVE -- bad things happen
import pdb


# [1]: https://pytorch.org/tutorials/intermediate/speech_command_classification_with_torchaudio_tutorial.html
# [2] HelloEdge: Keyword Spotting on Microcontrollers (https://arxiv.org/abs/1711.07128)
# [3] Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition (https://arxiv.org/abs/1804.03209)

BKG_NOISE = ["doing_the_dishes.wav", "dude_miaowing.wav", "exercise_bike.wav", "pink_noise.wav", "running_tap.wav", "white_noise.wav"]

CLASSES_12 = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'unknown', 'silence']

def pad_sequence(batch): #! borrowed from [1]
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)

def label_to_index(word, labels): #! borrowed from [1]
    # Return the position of the word in labels
    if word in labels:
        return torch.tensor(labels.index(word))
    else:
        return torch.tensor(10) # higlight as `unknown`


class PartitionedSPEECHCOMMANDS(SPEECHCOMMANDS):
    def __init__(self, data_path: Path, subset: str, transforms: list, classes: str = 'all', wav2fbank: bool=False):
        '''classes:
        # v1: either 30 (all) or 12 (10 + unknown + silence)
        # v2: either 35 (all) or 12 (10 + unknown + silence)
        '''

        super().__init__(data_path, url="",
                         folder_in_archive="",
                         subset=subset, download=False)

        self.subset = subset
        self.transforms = transforms
        self.device = 'cpu'
        self.wav2fbank = wav2fbank

        cls = [Path(f.path).name for f in os.scandir(self._path) if f.is_dir() and f.path != str(Path(self._path)/EXCEPT_FOLDER)]
        if "federated" in cls:
            cls.remove("federated") # if data_path points to the whole dataset (i.e. not inside /federated), we'll hit this

        self.classes_to_use = cls if classes=='all' else CLASSES_12
        # self.collate_fn = get_collate(self.classes_to_use, self.transforms)

        # let's pre-load all background audio clips. This should help when
        # blending keyword audio with bckg noise
        self.background_sounds = []
        for noise in BKG_NOISE:
            path = data_path/EXCEPT_FOLDER/noise
            path = os.readlink(path) if os.path.islink(path) else path
            waveform, sample_rate = load(path)
            self.background_sounds.append([waveform, sample_rate])

        # now let's assume we have 10% more data representing `silence`.
        #! Hack alert: we artificially add paths (that do not exist) to the _walker.
        #! When this path is chosen via __getitem__, it will be detected as w/ label "silence"and the file itself wont' be loaded. Instead a silence clip (i.e. all zeros) will be returned
        #! Silence support is done in self._load_speechcommands_item_with_silence_support()
        if 'silence' in self.classes_to_use:
            # append silences to walker in dataset object
            for _ in range(int(len(self._walker)*0.1)):
                self._walker.append(data_path/"silence/sdfsfdsf.wav")

        # print(f"Dataset contains {len(self._walker)} audio files")


    def _collate_fn(self, batch): #! ~borrowed from [1]
        # A data tuple has the form:
        # waveform, sample_rate, label, speaker_id, utterance_number

        tensors, targets = [], []

        # Gather in lists, and encode labels as indices
        for waveform, sr, label, *_ in batch:
            if self.wav2fbank:
                tensors += [self._wav2fbank(waveform, sr)]
                # print(tensors[-1].shape)
            else:
                tensors += [waveform]
            targets += [label_to_index(label, self.classes_to_use)]

        # Group the list of tensors into a batched tensor
        tensors = pad_sequence(tensors)
        targets = torch.stack(targets)

        # tensors = tensors.to(self.device)
        tensor_t = self.transforms(tensors)
        if self.wav2fbank:
            tensor_t = tensor_t.unsqueeze(1)

        return tensor_t, targets

    def _get_labels_histogram(self):
        """returns histogram of labels"""
        hist = [0] * len(self.classes_to_use)
        for p in self._walker:
            path = os.readlink(p) if os.path.islink(p) else p
            label = Path(path).parent.name
            hist[label_to_index(label, self.classes_to_use)] += 1
        return hist

    def get_balanced_sampler(self):
        """This construct a [1,N] array w/ N the number of datapoints in the datasets. Each
        gets assigned a probabily of being added to a batch of data. This will be passed to a initialise
        a WeightedRandomSampler and return it."""

        hist = self._get_labels_histogram()
        weight_per_class = [len(self._walker)/float(count) if count>0 else 0 for count in hist]
        w = [0] * len(self._walker)
        
        lls = []
        lls_idx = []

        for i, p in enumerate(self._walker):
            path = os.readlink(p) if os.path.islink(p) else p
            label = Path(path).parent.name
            label_idx = label_to_index(label, self.classes_to_use)
            lls.append(label)
            lls_idx.append(label_idx)
            w[i] = weight_per_class[label_idx]

        sampler = WeightedRandomSampler(w, len(w))
        return sampler, hist

    def _decode_classes(self, labels: torch.tensor):
        return [self.classes_to_use[i] for i in labels]


    def _extract_from_waveform(self, waveform, sample_rate):
        """Returns a waveform of `sample_rate` samples of the
        inputed `waveform`. If `sample_rate` is that of the `waveform`
        then the returned waveform will be 1s long."""
        min_t = 0
        max_t = waveform.shape[1] - sample_rate
        off_set = random.randint(min_t, max_t)
        return waveform[:, off_set:off_set+sample_rate]


    def _load_speechcommands_item_with_silence_support(self, filepath:str, path: str):
        """if loading `silence` we extract a 1s random clip from the background audio
        files in SpeechCommands dataset (this is how the `silence` category should be
        constructed according to the SpeechCommands paper). Else, the
        behaviour is the same as in the default SPECHCOMMANDS dataset"""
        # relpath = os.path.relpath(filepath, path)
        # print('------')
        # print(type(filepath), filepath)
        # print(path)
        
        relpath = '/'.join(str(filepath).split('/')[-2:])
        label, filename = os.path.split(relpath)

        # print(relpath)
        # print(label)
        # print(filename)
        # print('------')
        if label == 'silence':
            # construct path to a random .wav in background_noise dir
            # filepath = path + '/' + EXCEPT_FOLDER + "/" + random.sample(BKG_NOISE,1)[0]
            # picking one random pre-loaded background sound
            waveform, sample_rate = random.sample(self.background_sounds,1)[0]

            # let's extact a 1s sequence
            waveform = self._extract_from_waveform(waveform, sample_rate)
            utterance_number = -1
            speaker_id = -1
        else:

            speaker, _ = os.path.splitext(filename)
            speaker, _ = os.path.splitext(speaker)

            speaker_id, utterance_number = speaker.split(HASH_DIVIDER)
            utterance_number = int(utterance_number)

            # Load audio
            # print(f"loading: {filepath}")
            waveform, sample_rate = load(filepath)

        return waveform, sample_rate, label, speaker_id, utterance_number


    def _apply_time_shift(self, waveform, sample_rate):
        """Applies time shifting (positive or negative). Hardcoded
        to apply rand(-100ms, +100ms)."""

        #TODO: should we be doing this as a torchaudio.transform ?

        # apply random time shift of [-100ms, 100ms]
        shift_ammount = sample_rate/10 # this will give us a 10th of a 1s signal
        shift = random.randint(-shift_ammount, shift_ammount)
        if shift < 0:
            waveform = waveform[:, abs(shift):] # will be padded with zeros later on in collate_fn
        else:
            waveform_ = torch.zeros_like(waveform)
            waveform_[:, shift:] = waveform[:, :waveform.shape[1]-shift]
            waveform = waveform_

        return waveform


    def _blend_with_background(self, waveform):

        #TODO: should we be doing this as a torchaudio.transform ?
        background_volume = 0.1 #  the default in [2] #! this seems to limit acc -- maybe lower is better?
        background_frequency = 0.8 # ratio of samples that will get background added in (as in [2])

        if random.uniform(0.0, 1.0) < background_frequency:
            volume = random.uniform(0.0, background_volume) # as in [2]
            noise, _ = random.sample(self.background_sounds,1)[0]
            noise = self._extract_from_waveform(noise, waveform.shape[1])
            return (1.0 - volume)*waveform + volume*noise
        else:
            return waveform

    def _wav2fbank(self, waveform, sr, mel_bins=128, target_length=128):
        # eavily borrowing from `make_features()` in: https://colab.research.google.com/github/YuanGongND/ast/blob/master/Audio_Spectrogram_Transformer_Inference_Demo.ipynb#scrollTo=sapXfOwbhrzG
        f_bank = fbank(
            waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
            window_type='hanning', num_mel_bins=mel_bins, dither=0.0,
            frame_shift=10)

        n_frames = f_bank.shape[0]

        p = target_length - n_frames
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            f_bank = m(f_bank)
        elif p < 0:
            f_bank = f_bank[0:target_length, :]

        f_bank = (f_bank - (-4.2677393)) / (4.5689974 * 2)

        return f_bank

    def __getitem__(self, n: int) -> Tuple[torch.Tensor, int, str, str, int]:
        fileid = self._walker[n]
        path = self._path

        if os.path.islink(fileid):
            fileid = os.readlink(fileid)
            path = Path(self._path).parent.parent

        wvfrm, sr, label, speaker_id, utt_num = self._load_speechcommands_item_with_silence_support(fileid, path)

        if self.subset == "training":
            wvfrm = self._apply_time_shift(wvfrm, sr)
            wvfrm = self._blend_with_background(wvfrm) # TODO: this seems to impact accuracy quite a bit (~3% lower when activated)

        return wvfrm, sr, label, speaker_id, utt_num


def get_speechcommands_and_partition_it(destination_path: Path, version: int, max_train=250, max_unseen=50):
    """Downloads SpeechCommands dataset if not found and partitions it by
    `session ID` (which is a randomly generated alphanumeric sequence prefixing
    each audio file and can be use as speaker identifier -- according to [3]).

    max_train refers to the top k training speakers w the most data
    max_unseen refers to the top k validation+testing speakers w the most data
    Dataset statistics:
        v1: 64721 .wav files from 1881 speakers (1503 for training)
        v2: 105829 .wav files from 2618 speakers (2112 for training)
    """

    assert version in [1,2], f"Only version `1` or `2` are understood. You chose: {version}"

    path = Path(destination_path)
    path.mkdir(exist_ok=True)
    url = f"speech_commands_v0.0{version}"
    folder_in_archive = "SpeechCommands"
    whole_dataset = SPEECHCOMMANDS(path, url=url, folder_in_archive=folder_in_archive, subset=None, download=True)

    # get class all classes names
    cls_names = [Path(f.path).name for f in os.scandir(whole_dataset._path) if f.is_dir() and f.path != str(Path(whole_dataset._path)/EXCEPT_FOLDER)]

    if "federated" in cls_names:
        cls_names.remove("federated")

    # now we generate the `federated` directory
    fed_dir = Path(whole_dataset._path)/"federated"
    if not fed_dir.exists():
        print(f"{len(cls_names)} (total) classes found")
        print(f"Dataset has: {len(whole_dataset._walker)} .wav files")

        # Get speakers IDs
        unique_ids = []
        for wav in whole_dataset._walker:
            wav = Path(wav)
            session_id = wav.stem[:wav.stem.find(HASH_DIVIDER)]
            if session_id not in unique_ids:
                unique_ids.append(session_id)

        print(f"Unique speaker IDs found: {len(unique_ids)}")

        # From all the IDs, some are **excluselively** in the test set, others exclusively in
        # the validation set and the rest form the training set. Now we identify which
        # belongs to which split.

        val_list = _load_list(whole_dataset._path, "validation_list.txt")
        test_list = _load_list(whole_dataset._path, "testing_list.txt")
        train_ids = []
        val_ids = []
        test_ids = []
        for i, id in enumerate(unique_ids):
            for wav in whole_dataset._walker:
                if id in wav:
                    if wav in val_list:
                        val_ids.append(id)
                    elif wav in test_list:
                        test_ids.append(id)
                    else:
                        train_ids.append(id)
                    break

        print(f"Clients for training ({len(train_ids)}), validation ({len(val_ids)}), testing ({len(test_ids)})")

        assert len(train_ids)+len(val_ids)+len(test_ids) == len(unique_ids), "This shouldn't happen"

        # picking the top k from training speakers to form training clients and picking top k from val + test speakers to form unseen clients
        

        val_test_files = set(_load_list(whole_dataset._path, "validation_list.txt", "testing_list.txt"))
        walker = sorted(str(p) for p in Path(whole_dataset._path).glob("*/*.wav"))
        train_files = [
                w
                for w in walker
                if HASH_DIVIDER in w and EXCEPT_FOLDER not in w and os.path.normpath(w) not in val_test_files
            ]

        train_ids_files = defaultdict(list)
        unseen_ids_files = defaultdict(list)

        for id in train_ids:
            for file in train_files:
                if id in file: # if file belongs to this speaker ID
                    train_ids_files[id].append(file)

        for id in val_ids + test_ids:
            for file in val_test_files:
                if id in file:
                    unseen_ids_files[id].append(file)

        train_ids_files = sorted(train_ids_files.items(), key=lambda x:len(x[1]), reverse=True)
        unseen_ids_files = sorted(unseen_ids_files.items(), key=lambda x:len(x[1]), reverse=True)

        def setup_dir(dir_path, files, val_files, test_files):
            dir_path.mkdir(parents=True,exist_ok=True)

            # ensure all classes have a directory (this will be relevant for PartitionedSPEECHCOMMANDS as it will
            # be required to figureout the classes in the dataset)
            for _cls in cls_names:
                (dir_path/str(_cls)).mkdir()

            # create empyt `testing_list.txt` and `validation_list.txt`
            (dir_path/"validation_list.txt").touch()
            val_str = '\n'.join(['/'.join(v_file.split('/')[-2:]) for v_file in val_files])
            (dir_path/"validation_list.txt").write_text(val_str)
            (dir_path/"testing_list.txt").touch()
            test_str = '\n'.join(['/'.join(t_file.split('/')[-2:]) for t_file in test_files])
            (dir_path/"testing_list.txt").write_text(test_str)

            for _file in files:
                _cls = Path(_file).parent.stem
                os.symlink(_file, dir_path/_cls/Path(_file).name)

            # symlink also background sounds
            (dir_path/EXCEPT_FOLDER).mkdir()
            for each_file in (Path(whole_dataset._path)/EXCEPT_FOLDER).glob('*.wav*'):
                os.symlink(each_file, dir_path/EXCEPT_FOLDER/each_file.name)

        def train_val_test_split(idx, files):
            # sort files in classes
            cls_files = defaultdict(list) # class --> [files]
            for _file in files:
                cls_file = '/'.join(_file.split('/')[-2])
                cls_files[cls_file].append(_file)
            
            # for each class, split into train/val/test
            val_files = []
            test_files = []
            for v in cls_files.values():
                _train_val_size = int(0.8 * len(v))
                test_size = len(v) - _train_val_size
                # test_size = int(0.2 * len(v))
                _train_size = int(0.8 * (len(v) - test_size))
                val_size = len(v) - test_size - _train_size
                rng = np.random.default_rng()
                rng.shuffle(v)
                cls_test_files, _v = v[:test_size], v[test_size:]
                cls_val_files, cls_train_files = _v[:val_size], _v[val_size:]
                assert len(cls_train_files) + len(cls_val_files) + len(cls_test_files) == len(v)
                val_files += cls_val_files
                test_files += cls_test_files
            
            assert len(val_files) > 0, f'client {idx} has no val files'
            assert len(test_files) > 0, f'client {idx} has no test files' 
            return val_files, test_files

        fed_dir.mkdir()
        folders = ['train', 'val', 'test']

        ## Partition training. federated/training/{client_id}/{train/val/test}
        global_training_all = []
        global_training_val = []
        global_training_test = []
        
        for i, (_, files) in zip(range(max_train), train_ids_files):
            val_files, test_files = train_val_test_split(i, files)

            global_training_all += files
            global_training_val += val_files
            global_training_test += test_files

            setup_dir(Path(fed_dir/'training'/str(i)), files=files, val_files=val_files, test_files=test_files)
                
        
        setup_dir(Path(fed_dir/'training'), files=global_training_all, val_files=global_training_val, test_files=global_training_test)

        ## Partition val+test (unseen). federated/unseen/{client_id}/{train/val/test}
        global_unseen_all = []
        global_unseen_val = []
        global_unseen_test = []
        
        for i, (_, files) in zip(range(max_unseen), unseen_ids_files):
            val_files, test_files = train_val_test_split(i, files)

            global_unseen_all += files
            global_unseen_val += val_files
            global_unseen_test += test_files

            setup_dir(Path(fed_dir/'unseen'/str(i)), files=files, val_files=val_files, test_files=test_files)


        setup_dir(Path(fed_dir/'unseen'), files=global_unseen_all, val_files=global_unseen_val, test_files=global_unseen_test)

        print("Done")

    return fed_dir


#! this is the typical transformations for a KWS setup.
def raw_audio_to_mfcc_transforms():
    # values from [1], [2]: Here we transform the raw audio wave into MFCC features
    # which encode each audio clip into a 2D matrix.
    # This allows us to treat audio signals as images
    ss = 8000 # 8KHz
    n_mfcc = 40
    window_width = 40e-3 # length of window in seconds
    stride = 20e-3 # stride between windows
    n_fft = 400
    T = Compose([transforms.Resample(16000, ss),
                 transforms.MFCC(sample_rate=ss,
                                 n_mfcc=n_mfcc,
                                 melkwargs={'win_length': int(ss*window_width),
                                 'hop_length': int(ss*stride),
                                 'n_fft': n_fft}
                                 )
                ])
    return T


def raw_audio_to_AST_spectrogram():
    #! Convertion to Fbank spectrogram is done by passing wav2fbank=True, during dataset object construction
    # upscale to 224x1224 so ViT doesn't break
    #! THe reason why we dont do the FBank as a standard transform is because torchaudio.compliance.kaldi.fbank doesn't operate on batched data
    T = Compose([tv_transforms.Resize(size=(224,224))])
    return T


# def test(fed_dir: str, client_id: int, device: str, test_whole_dataset: bool = False):
#     '''Loads dataset for clien with id=client_id'''

#     if test_whole_dataset:
#         data_path = Path(fed_dir)
#     else:
#         data_path = Path(fed_dir)/str(client_id)
    
#     train_dataset = PartitionedSPEECHCOMMANDS(data_path, "training", transforms=raw_audio_to_AST_spectrogram(), wav2fbank=True, classes=12)
#     val_dataset = PartitionedSPEECHCOMMANDS(data_path, "validation", transforms=raw_audio_to_AST_spectrogram(), wav2fbank=True, classes=12)
#     test_dataset = PartitionedSPEECHCOMMANDS(data_path, "testing", transforms=raw_audio_to_AST_spectrogram(), wav2fbank=True, classes=12)

#     # if not using the whole labels provided (i.e. classes="all"), SpeechCommands is very inbalanced since many labels are collapsed under the "unknown" label
#     sampler, hist = train_dataset.get_balanced_sampler()
#     print("Train Histogram of labels:")
#     print(hist)

#     _, hist = val_dataset.get_balanced_sampler()
#     print("Val Histogram of labels:")
#     print(hist)
    
#     _, hist = test_dataset.get_balanced_sampler()
#     print("Test Histogram of labels:")
#     print(hist)

# if __name__ == "__main__":

#     version = 2
#     fed_dir = get_speechcommands_and_partition_it('../../datasets', version=version)
#     fed_dir = fed_dir / 'training'
#     print(f"{fed_dir}")

#     for client_id in range(245,250):
#         test(fed_dir, client_id=client_id, device='cuda', test_whole_dataset=False)

    # test(fed_dir, client_id=client_id, device='cuda', test_whole_dataset=True)