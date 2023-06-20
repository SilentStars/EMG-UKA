import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from functools import lru_cache
import random
from copy import copy


from absl import app, flags
FLAGS = flags.FLAGS

speaker_list = ['501', '551', '552', '553', '554', '556', '557', '601']

class EMG_dataset(Dataset):
    def __init__(self, root_dir):
        self.emg_dir = root_dir + '/emg'
        self.audio_dir = root_dir + '/audio'
        
        self.emg_speaker_subset_path = []
        self.audio_speaker_subset_path = []
        # architecture: Speaker, Sessions, paired EMG & audio files.
        self.paired_files = dict.fromkeys(speaker_list, [])
        
        self.full_data_filename = []
        
        for i in range(0, len(speaker_list)):
            emg_speaker_dir = os.path.join(self.emg_dir, speaker_list[i])
            audio_speaker_dir = os.path.join(self.audio_dir, speaker_list[i])
            
            self.emg_speaker_subset_path.append(emg_speaker_dir)
            self.audio_speaker_subset_path.append(audio_speaker_dir)
            
            single_speaker_record = []
            for sessions in os.listdir(emg_speaker_dir):
                single_session_emg_records = []
                single_session_audio_records = []
                
                for single_emg_record in os.listdir(os.path.join(emg_speaker_dir, sessions)):
                    if single_emg_record.endswith('.adc'):
                        if os.path.exists(os.path.join(audio_speaker_dir, sessions, 'a_' + speaker_list[i] + '_' + single_emg_record[-8:-4] + '.wav')):
                            single_session_audio_records.append(os.path.join(audio_speaker_dir, sessions, 'a_' + speaker_list[i] + '_' + single_emg_record[-8:-4] + '.wav'))
                            single_session_emg_records.append(os.path.join(emg_speaker_dir, sessions, single_emg_record))
                            for k in range(0, len(single_session_audio_records)):
                                # Triple tuple (speaker_id, wave_filename, emg_filename)
                                self.full_data_filename.append((str(i), single_session_audio_records[k], single_session_emg_records[k]))
                assert len(single_session_emg_records) == len(single_session_audio_records)
                single_speaker_record.append((sorted(single_session_emg_records), sorted(single_session_audio_records)))
            self.paired_files[speaker_list[i]] = single_speaker_record
        random.shuffle(self.full_data_filename)


    def certain_speaker_records(self, speaker_id):
        res = copy(self)
        res.full_data_filename = [k for k in self.full_data_filename if str(speaker_id) == k[0]]
        return res
        
    @lru_cache(maxsize=None)
    def __getitem__(self, index):
        def _read_adc(adc_file_path):
            # Read the adc file.
            # Shape should be (channels, lengths), for example, (7, 3842)
            def normalize(x):
                x = np.asarray(x)
                arr = (x - x.min()) / (x.max() - x.min())
                return np.subtract(np.multiply(arr, 2), 1)
            f = np.fromfile(adc_file_path, dtype='<i2').astype('int32')
            f = np.reshape(f, (int(len(f) / 7), 7)).T
            f = normalize(f)
            return f
        
        
        return super().__getitem__(index)
        
    def __len__(self):
        return len(self.full_data_filename)
        
        
def test_dataset(root_dir):
    dataset = EMG_dataset(root_dir).certain_speaker_records('1')
    print(dataset.__len__())
    
if __name__ == '__main__':
    test_dataset('EMG-UKA-Full-Corpus')
    