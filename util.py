import os
import numpy as np
from scipy import signal
from pydub import AudioSegment
import matplotlib.pyplot as plt

def split_audio(path_to_audio, time_step):
    '''
        Description:
            The split_audio() function splits a determinate audio into several parts, according to 
            the stipulated time inverval.

        Use:
            split_audio('/path/to/pac_audio', 1000)

        Parameters:
            path_to_audio:
                Path to the folder containing the audio file or the audio itself.
            step:
                Step to the audio will be divided, givin in milliseconds. 
                E.g: 1000 equals 1000 ms, which is 1 seconds.

        Return:
            A audios list containing in each position a part from original audio.
            NOTE: IN CASE THE AUDIO INEXISTENT, it'll return a empty list.
    '''

    if(not os.path.isfile(path_to_audio)):
        files = os.listdir(path_to_audio)
        if('qv001.wav' in files):
            path_to_audio = os.path.join(path_to_audio, 'qv001.wav')
        elif('qv012.wav' in files):
            path_to_audio = os.path.join(path_to_audio, 'qv012.wav')
        else:
            return []

    audio = AudioSegment.from_wav(path_to_audio)
    audios_list = [audio[start:start+time_step] for start in range(0, len(audio), time_step)]
    return audios_list


def wav_to_spectrogram(audio_file, path_to_save):
    '''
        Description:
            The wav2spectrogram function generates a spectrogram from an audio and saves it on disc.

        Use:
            wav2spectrogram(audio_file, '/path/to/save/image.png')

        Return:
            audio_file:
                Audio file of type pydub.AudioSegment.
            path_to_save:
                Path to the where the image will be save, along with its name.
    '''

    frequencies, times, spectrogram = signal.spectrogram(np.array(audio_file.get_array_of_samples()), audio_file.frame_rate)
    plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
    plt.box(False)
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    plt.pcolormesh(times, frequencies, np.log(spectrogram)) # 20.*np.log10(np.abs(spectrogram)/10e-6) decibel
    # plt.gcf().set_size_inches(2.24, 2.24)
    plt.gcf().set_frameon(False)
    plt.savefig(path_to_save)
    plt.clf()
