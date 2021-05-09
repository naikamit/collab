import numpy as np #test
import bisect
# import soundfile as sf
# For threading and queueing of songs
import threading
import timestretching
import timestretching2
import stretchspleet
import multiprocessing
from multiprocessing import Process, Queue, Event
import ctypes
from time import sleep
# For live playback of songs
import pyaudio
import wave
from array import array
import tracklister as tl
import songtransitions


import logging
logger = logging.getLogger('colorlogger')
# import librosa
import time, os, sys
# import csv
# import wave, array
# from essentia.standard import MonoWriter
import numpy as np
# import scipy.interpolate as interp
import random

class DjController:

    def __init__(self, tracklister): # inits two audio threads/processes, inits a bunch of object variables to be used later. no class methods called from below
        self.tracklister = tracklister
        # two processes run in parallel: the DJ process and the sound output to the crowd
        self.audio_thread = None
        self.dj_thread = None
        self.playEvent = multiprocessing.Event() # playEvent used in play()
        self.isPlaying = multiprocessing.Value('b', True) #If lock is True (the default) then a new recursive lock object is created to synchronize access to the value. If lock is a Lock or RLock object then that will be used to synchronize access to the value. If lock is False then access to the returned object will not be automatically protected by a lock, so it will not necessarily be “process-safe”.
        self.skipFlag = multiprocessing.Value('b', False)
        self.queue = Queue(5)     # A blocking queue to pass at most N audio fragments between audio thread and generation thread aka dj_thread?
        # self.currentMasterString = multiprocessing.Manager().Value(ctypes.c_char_p, '') # what? SOMETHING TO mark a playing song as wrongly annotated

        # init object properties to be used later
        self.pyaudio = None
        self.stream = None
        self.stream_r = None
        self.djloop_calculates_crossfade = False

        self.save_mix = True # was false
        self.save_dir_idx = 0
        self.save_dir = './mix_{}.mp3'
        self.save_dir_tracklist = './mix.txt'
        self.audio_to_save = None
        self.audio_save_queue = Queue(25)
        self.save_tracklist = []
        self.save_audio_mp3 = '../output.wav'
        self.frame = np.array([[0, 0], [0, 0]]);  # np.zeros(2**29)
        self.frame_r = np.array([[0, 0], [0, 0]]);  # np.zeros(2**29)
        # self.frame=[]
        self.time = 0


    def play(self, count,save_mix=True):

        self.playEvent.set() # multiprocessing event. check multiprocessing library for whats going on

        if self.dj_thread is None and self.audio_thread is None: # this if block is for the first time/song that play is called. its elif block is for an exception. both threads were initialized as NULL in dj class INIT
            self.save_mix = save_mix
            self.save_dir_idx = 0 # ++/--1 counter
            self.audio_to_save = []
            self.save_tracklist = []

            # if self.save_mix: # another process started to save the audio file. this process runs _flush_save_audio_buffer() method. total 3 processes I guess. also, unlike the dj and audio threads, this save file thread is started and init'd in the same line
            #     Process(target=self._flush_save_audio_buffer, args=(self.audio_save_queue,)).start()

            self.dj_thread = Process(group=None, target=self._dj_loop, args=(self.isPlaying,)) # process for dj controller/loop. this is just assignment, the thread starts a few lines down with thread.start()
            self.audio_thread = Process(target=self._audio_play_loop, args=(self.playEvent, self.isPlaying))

            self.isPlaying.value = True
            self.dj_thread.start() # here execution forks to _dj_loop() as a new process

            while self.queue.empty():
                sleep(0.1)

            self.audio_thread.start()  # here execution forks to _audio_play_loop() as a new process

        elif self.dj_thread is None or self.audio_thread is None:
            raise Exception('dj_thread and audio_thread are not both Null!')

    def pause(self):
        if self.isPlaying.value is False:
            self.isPlaying.value = True
        else:
            self.isPlaying.value = False
            return
        # self.playEvent.clear()


    def _audio_play_loop(self, playEvent, isPlaying):
        if self.pyaudio is None:

            null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
            save = os.dup(1), os.dup(2)
            os.dup2(null_fds[0], 1) # https://www.geeksforgeeks.org/python-os-dup2-method/
            os.dup2(null_fds[1], 2)

            # Open the audio
            self.pyaudio = pyaudio.PyAudio()

            # Reset stderr, stdout
            os.dup2(save[0], 1)
            os.dup2(save[1], 2)
            os.close(null_fds[0])
            os.close(null_fds[1])

        if self.stream is None:
            self.stream_r = self.pyaudio.open(format=pyaudio.paFloat32,
                                              channels=2,
                                              rate=44100,
                                              input=False,
                                              output=True)
            self.stream = self.pyaudio.open(format=pyaudio.paFloat32,
                                            channels=1,
                                            rate=44100,
                                            input=True,
                                            output=True)

        while isPlaying.value:
            toPlay, toPlayStr, masterTitle = self.queue.get()

            FRAME_LEN = 100
            last_frame_start_idx = int(len(toPlay) / FRAME_LEN) * FRAME_LEN
            for cur_idx in range(0, last_frame_start_idx + 1, FRAME_LEN):
                # playEvent.wait()
                # if not self.isPlaying.value:
                #     return
                # if self.skipFlag.value:
                #     self.skipFlag.value = False
                #     break
                if cur_idx == last_frame_start_idx:
                    end_idx = len(toPlay)
                else:
                    end_idx = cur_idx + FRAME_LEN
                toPlayNow = toPlay[cur_idx:end_idx,:]
                if toPlayNow.dtype != 'float32':
                    toPlayNow = toPlayNow.astype('float32')

                self.stream_r.write(toPlayNow, num_frames=len(toPlayNow), exception_on_underflow=False);

    def _dj_loop(self, isPlaying):

        unplayed = self.tracklister.getUnplayed()
        random.shuffle(unplayed)
        for s in unplayed:
            print(s.title)

        current_song = unplayed.pop()

        trackNo = 6
        # current_song = unplayed[trackNo]

        current_song.open()
        current_song.openAudio()
        next_song = unplayed.pop()
        next_song.open()
        next_song.openAudio()

        cues = current_song.downbeats
        cuesB = next_song.downbeats
        # stretchedBar = current_song.audio[int(44100 * cues[0]):int(44100 * cues[10])]
        # toPlayTuple = (stretchedBar, current_song.title, current_song.title)
        # self.queue.put(toPlayTuple, isPlaying.value)
        bar = 0

        while 1:
            originalBPM = current_song.downbeats[bar+8] - current_song.downbeats[bar]
            targetBPM = next_song.downbeats[8] - next_song.downbeats[0]
            f = targetBPM / originalBPM
            # print(f, abs(1-f),(len(cues)-30) )
            fSTEP = abs(1-f)/(len(cues))
            if fSTEP == 0:
                stepsToStretch = 0
            else:
                stepsToStretch = abs(1 - f) / fSTEP

            print("Now Playing:",current_song.title)
##########################

            if f > 1 + fSTEP: #current song needs to be slowed down by elongating
                ff = 1
                while bar < len(cues)-9:
                    ff = ff + fSTEP
                    stretchedBar = timestretching2.stretch(current_song, bar, cues, ff, "audio")
                    toPlayTuple = (stretchedBar, current_song.title, current_song.title)
                    self.queue.put(toPlayTuple, isPlaying.value)
                    bar = bar + 1
                    print(bar, " slow ", "BPM:", round(44100*240/len(stretchedBar),2), "  ProgressBar%:", round((bar)/(len(cues)) *100,0))

            if f < 1 - fSTEP:  #current song needs to be sped up by shortening
                ff = 1
                while bar < len(cues)-9:
                    ff = ff - fSTEP
                    stretchedBar = timestretching2.stretch(current_song, bar, cues, ff, "audio")
                    toPlayTuple = (stretchedBar, current_song.title, current_song.title)
                    self.queue.put(toPlayTuple, isPlaying.value)
                    bar = bar + 1
                    print(bar, " fast ", "BPM:", round(44100*240/len(stretchedBar),2), "  ProgressBar%:", round((bar)/(len(cues)) *100,0))
    # #CF
    #         originalBPM = current_song.downbeats[bar + 8] - current_song.downbeats[bar]
    #         targetBPM = next_song.downbeats[8] - next_song.downbeats[0]
    #         f = originalBPM / targetBPM
    #         print(f)


            count = 0
            allstepA = 1
            allstepB = 0
            step8 = 0.125

            while (count < 8): # source sweep

                barAall = timestretching2.stretch(current_song,bar, cues, f,"audio")
                barBall = next_song.audio[int(44100 * cuesB[count]):int(44100 * cuesB[count + 1])]
                # print("before crops: ", len(barAall), len(barBall))

                barAall = barAall[:len(barBall), :]
                barBall = barBall[:len(barAall), :]

                print(bar, " mixx ", "BPM:", round(44100*240/len(stretchedBar),2) , "  ProgressBar%:", round((bar)/(len(cues)) *100,0))

                allstepA = allstepA - step8
                allstepB = allstepB + step8

                barCF = barAall * allstepA + barBall * allstepB

                toPlayTuple = (barCF, current_song.title, current_song.title)
                self.queue.put(toPlayTuple, isPlaying.value)
                bar = bar + 1
                count = count + 1
                # print(bar, "sweep")
                # print(bar, "sweep", round(allstepA,1) , round(vocalstepB,1) , round(otherstepB,1) , round(drumstepB,1) , round(bassstepB,1))


            #A fin
            current_song = next_song
            next_song = unplayed.pop()
            next_song.open()
            next_song.openAudio()
            cues = current_song.downbeats
            cuesB = next_song.downbeats
            bar = count

        print ("while fin")
