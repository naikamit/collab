import numpy as np #test
import bisect
import soundfile as sf
# For threading and queueing of songs
import threading
import timestretching2
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
import librosa
import time, os, sys
import csv
import wave, array
from essentia.standard import MonoWriter
import numpy as np
import scipy.interpolate as interp

class DjController:

    def __init__(self, tracklister): # inits two audio threads/processes, inits a bunch of object variables to be used later. no class methods called from below
        self.tracklister = tracklister
        # two processes run in parallel: the DJ process and the sound output to the crowd
        self.audio_thread = None
        self.dj_thread = None
        self.playEvent = multiprocessing.Event() # playEvent used in play()
        self.isPlaying = multiprocessing.Value('b', True) #If lock is True (the default) then a new recursive lock object is created to synchronize access to the value. If lock is a Lock or RLock object then that will be used to synchronize access to the value. If lock is False then access to the returned object will not be automatically protected by a lock, so it will not necessarily be “process-safe”.
        self.skipFlag = multiprocessing.Value('b', False)
        self.queue = Queue(20)     # A blocking queue to pass at most N audio fragments between audio thread and generation thread aka dj_thread?
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
        self.audio_save_queue = Queue(20)
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

    # def save_audio_to_disk(self, audio, song_title):
    #     self.audio_to_save.extend(audio)
    #     self.save_tracklist.append(song_title)
    #
    #     if len(self.audio_to_save) > 44100 * 60 * 15:  # TODO test
    #         self.flush_audio_to_queue()
    #
    #
    # def _flush_save_audio_buffer(self, queue):
    #     while True:
    #         filename, audio, tracklist = queue.get() #think this line never gets executed . fatoes
    #         if not (filename is None):
    #             logger.debug('Saving {} to disk'.format(filename))
    #             writer = MonoWriter(filename=filename, bitrate=320, format='mp3')
    #             writer(np.array(audio, dtype='single'))
    #             # Save tracklist
    #             with open(self.save_dir_tracklist, 'a+') as csvfile:
    #                 writer = csv.writer(csvfile)
    #                 for line in tracklist:
    #                     writer.writerow([line])
    #         else:
    #             logger.debug('Stopping audio saving thread!')
    #             return
    #
    # def skipToNextSegment(self):
    #     if not self.queue.empty():
    #         self.skipFlag.value = True
    #     else:
    #         self.skipFlag.value = False
    #         logger.warning('Cannot skip to next segment, no audio in queue!')

    # def markCurrentMaster(self): # this method never really gets called. can be called from main though to deal with the wrong annotation later. CLI command mark current master
    #     with open('markfile.csv', 'a+') as csvfile:
    #         writer = csv.writer(csvfile)
    #         writer.writerow([self.currentMasterString.value])
    #     logger.debug('{:20s} has been marked for manual annotation.'.format(self.currentMasterString.value))

    # def pause(self):
    #     if self.audio_thread is None:
    #         return
    #     self.playEvent.clear()

    # def stop(self):
    #     # If paused, then continue playing (deadlock prevention)
    #     try:
    #         self.playEvent.set()
    #     except Exception as e:
    #         logger.debug(e)
    #     # Notify the threads to stop working
    #     self.isPlaying.value = False
    #     # Empty the queue so the dj thread can terminate
    #     while not self.queue.empty():
    #         self.queue.get_nowait()
    #     if not self.dj_thread is None:
    #         self.dj_thread.terminate()
    #     # Reset the threads
    #     self.queue = Queue(6)
    #     self.audio_thread = None
    #     self.dj_thread = None
    #     # Reset pyaudio resources
    #     if not self.stream is None:
    #         self.stream.stop_stream()
    #         self.stream.close()
    #     if not self.pyaudio is None:
    #         self.pyaudio.terminate()
    #     self.pyaudio = None



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
                                              output=True)  # for underrun, may try stream_callback=callback. this is by not using blocking IO
            self.stream = self.pyaudio.open(format=pyaudio.paFloat32,
                                            channels=1,
                                            rate=44100,
                                            input=True,
                                            output=True)
        # file_h = wave.open(self.save_audio_mp3, 'wb')
        # channel = 1
        # file_h.setnchannels(channel)
        # file_h.setsampwidth(self.pyaudio.get_sample_size(pyaudio.paInt32))
        # file_h.setframerate(44100)

        while isPlaying.value:
            toPlay, toPlayStr, masterTitle = self.queue.get()
            logger.info(toPlayStr)

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
        current_song = unplayed.pop()
        current_song.open()
        current_song.openAudio()
        next_song = unplayed.pop()
        next_song.open()
        next_song.openAudio()

        cues = current_song.downbeats
        print(cues[0])
        print(cues[1])
        case = "single"
        bar = 0
        FRAME_LEN = 100
        fSTEP = 0.011 # CONSTANT. absolute BPM change allowed per bar

        while 1:

            if case == "single":
                toPlay = current_song.audio[int(44100*cues[bar]):int(44100*cues[bar+1])]
                toPlayTuple = (toPlay, current_song.title, current_song.title)
                self.queue.put(toPlayTuple, isPlaying.value)
                print(' Bar:', bar)

            elif case == "stretch":
                downbeats = current_song.downbeats
                originalBPM = current_song.tempo
                targetBPM = 80 #next_song.tempo
                f = originalBPM / targetBPM

                if f > 1: #current song needs to be slowed down by elongating
                    ff = 1
                    while ff < f:
                        ff = ff + fSTEP
                        stretchedBar = timestretching2.stretch(current_song, bar, cues, ff)
                        toPlayTuple = (stretchedBar, current_song.title, current_song.title)
                        self.queue.put(toPlayTuple, isPlaying.value)
                        print("stretched bar: ", bar)
                        bar = bar + 1



                case = "transition"

            elif case == "transition":
                print ("transition case ran")

            bar = bar + 1

            if bar < (len(current_song.downbeats) - 195):
                print ("case: single")
            elif bar > (len(current_song.downbeats) - 195):
                case = "stretch"
                print("case: stretch")
            elif case == "stretch":
                case = "transition"
                print ("case: stretch")
            elif case == "transition":
                case = "single"
                print ("case: transition")



        print ("while fin")
