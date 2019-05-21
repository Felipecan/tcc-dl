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

        Parameters:
            audio_file:
                Audio file of type pydub.AudioSegment.
            path_to_save:
                Path to the where the image will be save, along with its name.
    '''

    frequencies, times, spectrogram = signal.spectrogram(np.array(audio_file.get_array_of_samples()), audio_file.frame_rate)
    # fmin = 0000 # Hz
    # fmax = 4000 # Hz
    # freq_slice = np.where((frequencies >= fmin) & (frequencies <= fmax))
    # # keep only frequencies of interest
    # frequencies = frequencies[freq_slice]
    # spectrogram = spectrogram[freq_slice,:][0]

    plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
    plt.box(False)
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    plt.pcolormesh(times, frequencies, np.log(spectrogram)) # 20.*np.log10(np.abs(spectrogram)/10e-6) decibel
    # plt.gcf().set_size_inches(2.24, 2.24)
    plt.gcf().set_frameon(False)
    plt.savefig(path_to_save)
    plt.clf()


def stacked_bar(data, series_labels, category_labels=None, 
                show_values=False, value_format="{}", y_label=None, 
                grid=True, reverse=False):
    """Plots a stacked bar chart with the data and labels provided.

    Keyword arguments:
    data            -- 2-dimensional numpy array or nested list
                       containing data for each series in rows
    series_labels   -- list of series labels (these appear in
                       the legend)
    category_labels -- list of category labels (these appear
                       on the x-axis)
    show_values     -- If True then numeric value labels will 
                       be shown on each bar
    value_format    -- Format string for numeric value labels
                       (default is "{}")
    y_label         -- Label for y-axis (str)
    grid            -- If True display grid
    reverse         -- If True reverse the order that the
                       series are displayed (left-to-right
                       or right-to-left)
    """
    plt.tick_params(top=True, bottom=True, left=True, right=True, labelleft=True, labelbottom=True)
    plt.box(True)
    plt.tight_layout()
    
    ny = len(data[0])
    ind = list(range(ny))

    axes = []
    cum_size = np.zeros(ny)

    data = np.array(data)

    if reverse:
        data = np.flip(data, axis=1)
        category_labels = reversed(category_labels)

    for i, row_data in enumerate(data):
        axes.append(plt.bar(ind, row_data, bottom=cum_size, 
                            label=series_labels[i]))
        cum_size += row_data

    if category_labels:
        plt.xticks(ind, category_labels)
        
    # to y ticks
#     m = max(data[0]+data[1])
#     plt.yticks(list(range(m)), list(range(m)))

    if y_label:
        plt.ylabel(y_label)

    plt.legend()

#     if grid:
#         plt.grid()

    if show_values:
        for axis in axes:
            for bar in axis:
                w, h = bar.get_width(), bar.get_height()
                plt.text(bar.get_x() + w/2, bar.get_y() + h/2, 
                         value_format.format(h), ha="center", 
                         va="center")
                
    plt.gcf().set_frameon(True)