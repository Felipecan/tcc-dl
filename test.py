from util import *
from pydub import AudioSegment

audio = AudioSegment.from_wav('./tcc_files/patients_audios/pac006/qv002.wav')
print(audio.frame_rate)
print(len(audio))