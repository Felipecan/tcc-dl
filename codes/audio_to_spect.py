import matplotlib.pyplot as plt
from scipy import signal
from pydub import AudioSegment
import numpy as np

def split_audio(time):

    audio = AudioSegment.from_wav('qv002.wav')
    audios_list = [audio[start:start+1000] for start in range(0, len(audio), 1000)]
    # print(len(audios_list))
    # print(len(audios_list[12]))
    frequencies, times, spectrogram = signal.spectrogram(np.array(audio.get_array_of_samples()), audio.frame_rate)
    
    plt.pcolormesh(times, frequencies, np.log(spectrogram))    
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')

    plt.show()


def wav2spectrogram(path_to_file):
    
    audio = AudioSegment.from_wav(path_to_file)

    frequencies, times, spectrogram = signal.spectrogram(np.array(audio.get_array_of_samples()), audio.frame_rate)
    # 20.*np.log10(np.abs(spectrogram)/10e-6) decibel
    # 10.*np.log10(spectrogram)
    plt.pcolormesh(times, frequencies, np.log(spectrogram))
    plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
    plt.box(False)    
    plt.savefig('./teste.png', bbox_inches='tight', pad_inches=0)

