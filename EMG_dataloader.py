import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os


from absl import app, flags
FLAGS = flags.FLAGS

speaker_list = ['501', '551', '552', '553', '554', '556', '557', '601']

class EMG_dataset(Dataset):
    def __init__(self, root_dir):
        self.emg_dir = root_dir + '/emg'
        self.audio_dir = root_dir + '/audio'
        
        self.emg_speaker_subset_path = []
        self.audio_speaker_subset_path = []
        # Sessions.
        emg_files = dict.fromkeys(speaker_list, [])
        
        for i in range(0, len(speaker_list)):
            emg_speaker_dir = os.path.join(self.emg_dir, speaker_list[i])
            self.emg_speaker_subset_path.append(emg_speaker_dir)
            single_speaker_record = []
            for sessions in os.listdir(emg_speaker_dir):
                single_session_records = []
                for single_emg_record in os.listdir(os.path.join(emg_speaker_dir, sessions)):
                    if single_emg_record.endswith('.adc'):
                        single_session_records.append(os.path.join(emg_speaker_dir, sessions, single_emg_record))
                single_speaker_record.append(single_session_records)
            emg_files[speaker_list[i]] = single_speaker_record
            self.audio_speaker_subset_path.append(os.path.join(self.audio_dir,speaker_list[i]))
    
    def __getitem__(self, index):
        def _read_adc(adc_file_path):
            # Read the adc file.
            # Shape should be (channels, lengths), for example, (7, 3842)
            def normalize(x):
                x = np.asarray(x)
                arr = (x - x.min()) / (int(x.max()) - int(x.min()))
                return np.subtract(np.multiply(arr, 2), 1)
            f = np.fromfile(adc_file_path, dtype='<i2').astype('int32')
            f = np.reshape(f, (int(len(f) / 7), 7)).T
            f = normalize(f)
            return f
        return super().__getitem__(index)
        
    def __len__(self):
        return 0
        
        
def test_dataset(root_dir):
    dataset = EMG_dataset(root_dir)
    
if __name__ == '__main__':
    test_dataset('EMG-UKA-Full-Corpus')