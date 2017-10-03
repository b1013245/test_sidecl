# Calc first stroke time by using onset
# coding:utf-8

from __future__ import print_function
import librosa
import numpy
from matplotlib import pyplot as plt
import os
import sys
import re
import csv

class getOnset(object):
    # 1. Get the file path to the included audio example
    def init(self, argv):
        dir = os.listdir(argv[1])
        self.files = [f for f in dir if not len(re.findall('.wav', f)) == 0 ]
        self.dir_path = argv[1]

    # 2. Load the audio as a waveform `y`
    #    Store the sampling rate as `sr`
    def load_wav(self):
        onset_times = {}
        for file in self.files:
            name = file[8:13]
            angle = self.get_angle(file[13])

            y, sr = librosa.load(self.dir_path + file)
            time = self.onset_detect(y,sr)
            if name in onset_times.keys():
                onset_times[name].update({angle:time})
            else:
                onset_times[name] = {angle:time}
        if len(onset_times.items()) > 2:
            return onset_times
    
    def get_angle(self, angle):
        if angle == 'F':
            return 'Front'
        elif angle == 'L':
            return 'Left'
        elif angle == 'R':
            return 'Right'

    def onset_detect(self, y, sr):
        env   = librosa.onset.onset_strength(y, sr=sr)
        count = 0
        for i in env:
            count +=1
            if i > 0:
                # 4. 時刻に戻す
                t = librosa.frames_to_time(count, sr=sr)
                return t[0]

    def write_csv(self, times):
        with open(self.dir_path + 'onset_time.csv', 'w') as f:
            header = ['Name', 'Front', 'Left', 'Right']
            w = csv.DictWriter(f, header)
            w.writeheader()

            for key, value in times.items():
                row = {}
                row['Name'] = key
                for k, v in value.items():
                    row[k] = v
                w.writerow(row)


    # Plot onset times for debug
    def plot_onset(self, file):
        y, sr = librosa.load(file)
        env   = librosa.onset.onset_strength(y, sr=sr)
        times = librosa.frames_to_time(numpy.arange(len(env)), sr=sr)
        plt.vlines(times, 0, env, color='r', alpha=0.9, linestyle='-', label='Onsets')
        plt.show()


if __name__ == '__main__':
    onset = getOnset()
    onset.init(sys.argv)
    times = onset.load_wav()
    onset.write_csv(times)


    # test