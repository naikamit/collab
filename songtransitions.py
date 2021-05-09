import numpy as np
import itertools
import yodel.filter  # For low pass and high pass bench filters
import scipy.signal
import tracklister



import time

import logging

logger = logging.getLogger('colorlogger')


def piecewise_fade_volume(audio, volume_profile, fade_in_len):
    output_audio = np.zeros(audio.shape)
    # fade_in_len_samples = output_audio.size
    fade_in_len_samples = len(output_audio)
    for j in range(len(volume_profile) - 1):
        start_dbeat, start_volume = volume_profile[j]
        end_dbeat, end_volume = volume_profile[j + 1]
        start_idx = int(fade_in_len_samples * float(start_dbeat) / fade_in_len)
        end_idx = int(fade_in_len_samples * float(end_dbeat) / fade_in_len)
        audio_to_fade = audio[start_idx:end_idx, :]
        output_audio[start_idx:end_idx, :] = linear_fade_volume(audio_to_fade, start_volume=start_volume,
                                                                end_volume=end_volume)

    return output_audio


def piecewise_lin_fade_filter(audio, filter_type, profile, fade_in_len):
    output_audio = np.zeros(audio.shape)
    fade_in_len_samples = audio.size

    for j in range(len(profile) - 1):
        start_dbeat, start_volume = profile[j]
        end_dbeat, end_volume = profile[j + 1]
        start_idx = int(fade_in_len_samples * float(start_dbeat) / fade_in_len)
        end_idx = int(fade_in_len_samples * float(end_dbeat) / fade_in_len)
        audio_to_fade = audio[start_idx:end_idx]
        output_audio[start_idx:end_idx] = linear_fade_filter(audio_to_fade, filter_type, start_volume=start_volume,
                                                             end_volume=end_volume)

    return output_audio


def linear_fade_volume(audio, start_volume=0.0, end_volume=1.0):
    if start_volume == end_volume == 1.0:
        return audio

    # length = audio.size
    # profile = np.sqrt(np.linspace(start_volume, end_volume, length))
    length = len(audio);
    profile_1 = np.sqrt(np.linspace(start_volume, end_volume, length))
    profile_2 = np.sqrt(np.linspace(start_volume, end_volume, length));
    profile = np.column_stack((profile_1, profile_2));  # logger.info(np.shape(profile));
    return audio * profile


def linear_fade_filter(audio, filter_type, start_volume=0.0, end_volume=1.0):
    if start_volume == end_volume == 1.0:
        return audio

    SAMPLE_RATE = 44100
    LOW_CUTOFF = 32
    MID_CENTER_1 = 64
    MID_CENTER_2 = 125
    MID_CENTER_3 = 250
    MID_CENTER_4 = 500
    MID_CENTER_5 = 1000
    MID_CENTER_6 = 2000
    MID_CENTER_7 = 4000
    MID_CENTER_8 = 8000
    HIGH_CUTOFF = 16000
    Q = 1.0 / np.sqrt(2)
    NUM_STEPS = 20 if start_volume != end_volume else 1

    bquad_filter = yodel.filter.Biquad()
    length = audio.size  # Assumes mono audio

    profile = np.linspace(start_volume, end_volume, NUM_STEPS)
    output_audio = np.zeros(audio.shape)

    for i in range(NUM_STEPS):
        start_idx = int((i / float(NUM_STEPS)) * length)
        end_idx = int(((i + 1) / float(NUM_STEPS)) * length)
        if filter_type == 'low_shelf':
            bquad_filter.low_shelf(SAMPLE_RATE, LOW_CUTOFF, Q, -int(26 * (1.0 - profile[i])))
        elif filter_type == 'mid_center1_shelf':
            bquad_filter.high_shelf(SAMPLE_RATE, LOW_CUTOFF, Q, -int(26 * (1.0 - profile[i])))
            bquad_filter.low_shelf(SAMPLE_RATE, MID_CENTER_1, Q, -int(26 * (1.0 - profile[i])))
        elif filter_type == 'mid_center2_shelf':
            bquad_filter.high_shelf(SAMPLE_RATE, MID_CENTER_1, Q, -int(26 * (1.0 - profile[i])))
            bquad_filter.low_shelf(SAMPLE_RATE, MID_CENTER_2, Q, -int(26 * (1.0 - profile[i])))
        elif filter_type == 'mid_center3_shelf':
            bquad_filter.high_shelf(SAMPLE_RATE, MID_CENTER_2, Q, -int(26 * (1.0 - profile[i])))
            bquad_filter.low_shelf(SAMPLE_RATE, MID_CENTER_3, Q, -int(26 * (1.0 - profile[i])))
        elif filter_type == 'mid_center4_shelf':
            bquad_filter.high_shelf(SAMPLE_RATE, MID_CENTER_3, Q, -int(26 * (1.0 - profile[i])))
            bquad_filter.low_shelf(SAMPLE_RATE, MID_CENTER_4, Q, -int(26 * (1.0 - profile[i])))
        elif filter_type == 'mid_center5_shelf':
            bquad_filter.high_shelf(SAMPLE_RATE, MID_CENTER_4, Q, -int(26 * (1.0 - profile[i])))
            bquad_filter.low_shelf(SAMPLE_RATE, MID_CENTER_5, Q, -int(26 * (1.0 - profile[i])))
        elif filter_type == 'mid_center6_shelf':
            bquad_filter.high_shelf(SAMPLE_RATE, MID_CENTER_5, Q, -int(26 * (1.0 - profile[i])))
            bquad_filter.low_shelf(SAMPLE_RATE, MID_CENTER_6, Q, -int(26 * (1.0 - profile[i])))
        elif filter_type == 'mid_center7_shelf':
            bquad_filter.high_shelf(SAMPLE_RATE, MID_CENTER_6, Q, -int(26 * (1.0 - profile[i])))
            bquad_filter.low_shelf(SAMPLE_RATE, MID_CENTER_7, Q, -int(26 * (1.0 - profile[i])))
        elif filter_type == 'mid_center8_shelf':
            bquad_filter.high_shelf(SAMPLE_RATE, MID_CENTER_7, Q, -int(26 * (1.0 - profile[i])))
            bquad_filter.low_shelf(SAMPLE_RATE, MID_CENTER_8, Q, -int(26 * (1.0 - profile[i])))
        elif filter_type == 'high_shelf':
            bquad_filter.high_shelf(SAMPLE_RATE, HIGH_CUTOFF, Q, -int(26 * (1.0 - profile[i])))
        else:
            raise Exception('Unknown filter type: ' + filter_type)
        # ~ bquad_filter.process(audio[start_idx : end_idx], output_audio[start_idx : end_idx]) # This was too slow, code beneath is faster!
        b = bquad_filter._b_coeffs
        a = bquad_filter._a_coeffs
        a[
            0] = 1.0  # Normalizing the coefficients is already done in the yodel object, but a[0] is never reset to 1.0 after division!
        output_audio[start_idx: end_idx] = scipy.signal.lfilter(b, a, audio[start_idx: end_idx]).astype('float32')

    return output_audio


class TransitionProfile:

    def __init__(self, len_dbeats, volume_profile, low_profile, mid_center1_profile, mid_center2_profile,
                 mid_center3_profile, mid_center4_profile, mid_center5_profile, mid_center6_profile,
                 mid_center7_profile, mid_center8_profile, high_profile):
        '''
            This class represents a transition profile during a fade.
            It takes three profiles as input. A profile is a sequence of tuples in the following format:
            [(queue_dbeat_1,volume_fraction_1),(queue_dbeat_2,volume_fraction_2),...]
            For example:
            [(0,0.0), (1,0.5), (7, 0.5), (8, 1.0)] is a transition like this:
                         ____
                        /
                 -------
            ____/

            The first downbeat must always be 0 and the last must always be length_dbeats
            The profile must also be non-decreasing in downbeat number
            The fractions must be between 0.0 and 1.0

            :len_dbeats		The length of the transition in downbeats
            :volume_profile	The profile
        '''
        if not ((volume_profile[0][0] == low_profile[0][0] == mid_center1_profile[0][0] == mid_center2_profile[0][0] ==
                 mid_center3_profile[0][0] == mid_center4_profile[0][0] == mid_center5_profile[0][0] ==
                 mid_center6_profile[0][0] == mid_center7_profile[0][0] == mid_center8_profile[0][0] == high_profile[0][
                     0] == 0) \
                and (volume_profile[len(volume_profile) - 1][0] == low_profile[len(low_profile) - 1][0] ==
                     mid_center1_profile[len(mid_center1_profile) - 1][0] ==
                     mid_center2_profile[len(mid_center2_profile) - 1][0] ==
                     mid_center3_profile[len(mid_center3_profile) - 1][0] ==
                     mid_center4_profile[len(mid_center4_profile) - 1][0] ==
                     mid_center5_profile[len(mid_center5_profile) - 1][0] ==
                     mid_center6_profile[len(mid_center6_profile) - 1][0] ==
                     mid_center7_profile[len(mid_center7_profile) - 1][0] ==
                     mid_center8_profile[len(mid_center8_profile) - 1][0] == high_profile[len(high_profile) - 1][
                         0] == len_dbeats)):
            raise Exception('Profiles must start at downbeat 0 and end at downbeat len_dbeats')
        i_prev = -1
        for i, v in volume_profile:
            if not i_prev <= i:
                logger.debug(volume_profile)
                raise Exception('Profiles must be increasing in downbeat indices')
            if not (v >= 0.0 and v <= 1.0):
                raise Exception('Profile values must be between 0.0 and 1.0')
            i_prev = i
        i_prev = -1
        for i, v in low_profile:
            if not i_prev <= i:
                raise Exception('Profiles must be increasing in downbeat indices')
            if not (v >= -1.0 and v <= 1.0):
                raise Exception('Profile values must be between 0.0 and 1.0')
            i_prev = i
        i_prev = -1
        for i, v in mid_center1_profile:
            if not i_prev <= i:
                raise Exception('Profiles must be increasing in downbeat indices')
            if not (v >= -1.0 and v <= 1.0):
                raise Exception('Profile values must be between 0.0 and 1.0')
            i_prev = i
        i_prev = -1
        for i, v in mid_center2_profile:
            if not i_prev <= i:
                raise Exception('Profiles must be increasing in downbeat indices')
            if not (v >= -1.0 and v <= 1.0):
                raise Exception('Profile values must be between 0.0 and 1.0')
            i_prev = i
        i_prev = -1
        for i, v in mid_center3_profile:
            if not i_prev <= i:
                raise Exception('Profiles must be increasing in downbeat indices')
            if not (v >= -1.0 and v <= 1.0):
                raise Exception('Profile values must be between 0.0 and 1.0')
            i_prev = i
        i_prev = -1
        for i, v in mid_center4_profile:
            if not i_prev <= i:
                raise Exception('Profiles must be increasing in downbeat indices')
            if not (v >= -1.0 and v <= 1.0):
                raise Exception('Profile values must be between 0.0 and 1.0')
            i_prev = i
        i_prev = -1
        for i, v in mid_center5_profile:
            if not i_prev <= i:
                raise Exception('Profiles must be increasing in downbeat indices')
            if not (v >= -1.0 and v <= 1.0):
                raise Exception('Profile values must be between 0.0 and 1.0')
            i_prev = i
        i_prev = -1
        for i, v in mid_center6_profile:
            if not i_prev <= i:
                raise Exception('Profiles must be increasing in downbeat indices')
            if not (v >= -1.0 and v <= 1.0):
                raise Exception('Profile values must be between 0.0 and 1.0')
            i_prev = i
        i_prev = -1
        for i, v in mid_center7_profile:
            if not i_prev <= i:
                raise Exception('Profiles must be increasing in downbeat indices')
            if not (v >= -1.0 and v <= 1.0):
                raise Exception('Profile values must be between 0.0 and 1.0')
            i_prev = i
        i_prev = -1
        for i, v in mid_center8_profile:
            if not i_prev <= i:
                raise Exception('Profiles must be increasing in downbeat indices')
            if not (v >= -1.0 and v <= 1.0):
                raise Exception('Profile values must be between 0.0 and 1.0')
            i_prev = i
        i_prev = -1
        for i, v in high_profile:
            if not i_prev <= i:
                raise Exception('Profiles must be increasing in downbeat indices')
            if not (v >= -1.0 and v <= 1.0):
                raise Exception('Profile values must be between 0.0 and 1.0')
            i_prev = i
        self.len_dbeats = len_dbeats
        self.volume_profile = volume_profile
        self.low_profile = low_profile
        self.mid_center1_profile = mid_center1_profile
        self.mid_center2_profile = mid_center2_profile
        self.mid_center3_profile = mid_center3_profile
        self.mid_center4_profile = mid_center4_profile
        self.mid_center5_profile = mid_center5_profile
        self.mid_center6_profile = mid_center6_profile
        self.mid_center7_profile = mid_center7_profile
        self.mid_center8_profile = mid_center8_profile
        self.high_profile = high_profile

    def apply(self, audio):

        output_audio = np.copy(audio)
        fade_in_len = self.len_dbeats

        low_profile = self.low_profile
        output_audio = piecewise_lin_fade_filter(output_audio, 'low_shelf', low_profile, fade_in_len)

        mid_center1_profile = self.mid_center1_profile
        output_audio = piecewise_lin_fade_filter(output_audio, 'mid_center1_shelf', mid_center1_profile, fade_in_len)

        mid_center2_profile = self.mid_center2_profile
        output_audio = piecewise_lin_fade_filter(output_audio, 'mid_center2_shelf', mid_center2_profile, fade_in_len)

        mid_center3_profile = self.mid_center3_profile
        output_audio = piecewise_lin_fade_filter(output_audio, 'mid_center3_shelf', mid_center3_profile, fade_in_len)

        mid_center4_profile = self.mid_center4_profile
        output_audio = piecewise_lin_fade_filter(output_audio, 'mid_center4_shelf', mid_center4_profile, fade_in_len)

        mid_center5_profile = self.mid_center5_profile
        output_audio = piecewise_lin_fade_filter(output_audio, 'mid_center5_shelf', mid_center5_profile, fade_in_len)

        mid_center6_profile = self.mid_center6_profile
        output_audio = piecewise_lin_fade_filter(output_audio, 'mid_center6_shelf', mid_center6_profile, fade_in_len)

        mid_center7_profile = self.mid_center7_profile
        output_audio = piecewise_lin_fade_filter(output_audio, 'mid_center7_shelf', mid_center7_profile, fade_in_len)

        mid_center8_profile = self.mid_center8_profile
        output_audio = piecewise_lin_fade_filter(output_audio, 'mid_center8_shelf', mid_center8_profile, fade_in_len)

        high_profile = self.high_profile
        output_audio = piecewise_lin_fade_filter(output_audio, 'high_shelf', high_profile, fade_in_len)

        volume_profile = self.volume_profile
        output_audio = piecewise_fade_volume(output_audio, volume_profile, fade_in_len)

        return output_audio


class CrossFade:
    '''
        Represents a crossfade where both the master and the slave can be adjusted at the same time
    '''

    def __init__(self, queue_1, queue_2_options, len_dbeats, crossover_point, fade_type, switch_master=True):
        '''
            :queue_1		the queue point in the first song in nr of downbeats
            :queue_2		the queue point in the second song in nr of downbeats
            :fade_in_len	the length of the fade in section (after fade_in: switch master)
            :fade_out_len	the length of the fade out section
            :switch_master	false if the master before and master after the fade are the same song (song 1)
        '''
        self.queue_1 = queue_1
        self.queue_2_options = queue_2_options
        self.len_dbeats = len_dbeats
        self.crossover_point = crossover_point

        P = self.crossover_point
        L = len_dbeats
        # if P < 2 or L - P < 2:
        #     # This is an atypical cross-fade:fade-in and/or fade-out is very short!
        #     if P < 0:
        #         # This is a bugfix, sometimes the crossover point is 0 apparently!
        #         logger.warning('Crossover point is negative!')
        #         P = L / 2
        #
        #     time_points = [0, P, P, L]
            # master_vol_profile = list(zip(time_points, [1.0, 1.0, 0.8, 0.0]))
            # master_low_profile = list(zip(time_points, [1.0, 1.0, -1.0, -1.0]))
            # master_mid_center1 = list(zip(time_points, [1.0, 1.0, -1.0, -1.0]))
            # master_mid_center2 = list(zip(time_points, [1.0, 1.0, -1.0, -1.0]))
            # master_mid_center3 = list(zip(time_points, [1.0, 1.0, -1.0, -1.0]))
            # master_mid_center4 = list(zip(time_points, [1.0, 1.0, -1.0, -1.0]))
            # master_mid_center5 = list(zip(time_points, [1.0, 1.0, -1.0, -1.0]))
            # master_mid_center6 = list(zip(time_points, [1.0, 1.0, -1.0, -1.0]))
            # master_mid_center7 = list(zip(time_points, [1.0, 1.0, -1.0, -1.0]))
            # master_mid_center8 = list(zip(time_points, [1.0, 1.0, -1.0, -1.0]))
            # master_high_profile = list(zip(time_points, [1.0, 1.0, -1.0, -1.0]))
            # slave_vol_profile = list(zip(time_points, [0.0, 0.8, 1.0, 1.0]))
            # slave_low_profile = list(zip(time_points, [-1.0, -1.0, 1.0, 1.0]))
            # slave_mid_center1 = list(zip(time_points, [-1.0, -1.0, 1.0, 1.0]))
            # slave_mid_center2 = list(zip(time_points, [-1.0, -1.0, 1.0, 1.0]))
            # slave_mid_center3 = list(zip(time_points, [-1.0, -1.0, 1.0, 1.0]))
            # slave_mid_center4 = list(zip(time_points, [-1.0, -1.0, 1.0, 1.0]))
            # slave_mid_center5 = list(zip(time_points, [-1.0, -1.0, 1.0, 1.0]))
            # slave_mid_center6 = list(zip(time_points, [-1.0, -1.0, 1.0, 1.0]))
            # slave_mid_center7 = list(zip(time_points, [-1.0, -1.0, 1.0, 1.0]))
            # slave_mid_center8 = list(zip(time_points, [-1.0, -1.0, 1.0, 1.0]))
            # slave_high_profile = list(zip(time_points, [-1.0, -1.0, 1.0, 1.0]))
        #
        # else:
        #     if fade_type == tracklister.TYPE_ROLLING or fade_type == tracklister.TYPE_DOUBLE_DROP:
        #         time_points = [0, 1, P / 2, P, P + 1, P + (L - P) / 2, L]
                # master_vol_profile = list(zip(time_points, [1.0, 1.0, 1.0, 1.0, 0.8, 0.8, 0.0]))
                # master_low_profile = list(zip(time_points, [1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0]))
                # master_mid_center1 = list(zip(time_points, [1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0]))
                # master_mid_center2 = list(zip(time_points, [1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0]))
                # master_mid_center3 = list(zip(time_points, [1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0]))
                # master_mid_center4 = list(zip(time_points, [1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0]))
                # master_mid_center5 = list(zip(time_points, [1.0, 1.0, 1.0, 1.0, 0.2, -1.0, -1.0]))
                # master_mid_center6 = list(zip(time_points, [1.0, 1.0, 1.0, 1.0, 0.2, -1.0, -1.0]))
                # master_mid_center7 = list(zip(time_points, [1.0, 1.0, 1.0, 1.0, 0.2, -1.0, -1.0]))
                # master_mid_center8 = list(zip(time_points, [1.0, 1.0, 1.0, 1.0, 0.2, -1.0, -1.0]))
                # master_high_profile = list(zip(time_points, [1.0, 1.0, 1.0, 1.0, 0.2, -1.0, -1.0]))
                # slave_vol_profile = list(zip(time_points, [0.0, 0.2, 0.8, 0.8, 1.0, 1.0, 1.0]))
                # slave_low_profile = list(zip(time_points, [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0]))
                # slave_mid_center1 = list(zip(time_points, [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0]))
                # slave_mid_center2 = list(zip(time_points, [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0]))
                # slave_mid_center3 = list(zip(time_points, [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0]))
                # slave_mid_center4 = list(zip(time_points, [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0]))
                # slave_mid_center5 = list(zip(time_points, [-1.0, -1.0, -1.0, 0.2, 1.0, 1.0, 1.0]))
                # slave_mid_center6 = list(zip(time_points, [-1.0, -1.0, -1.0, 0.2, 1.0, 1.0, 1.0]))
                # slave_mid_center7 = list(zip(time_points, [-1.0, -1.0, -1.0, 0.2, 1.0, 1.0, 1.0]))
                # slave_mid_center8 = list(zip(time_points, [-1.0, -1.0, -1.0, 0.2, 1.0, 1.0, 1.0]))
                # slave_high_profile = list(zip(time_points, [-1.0, -1.0, -1.0, 0.2, 1.0, 1.0, 1.0]))
            # else:
        P = self.crossover_point
        L = len_dbeats
        time_points = [0, P / 2, P, P, P + (L - P) / 2, L]

        master_vol_profile = list(zip(time_points, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))

        master_low_profile = list(zip(time_points, [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]))
        master_mid_center1 = list(zip(time_points, [1.0, -1.0, -1.0, -1.0, -1.0, -1.0]))
        master_mid_center2 = list(zip(time_points, [1.0, -1.0, -1.0, -1.0, -1.0, -1.0]))
        master_mid_center3 = list(zip(time_points, [1.0, 1.0, -1.0, -1.0, -1.0, -1.0]))
        master_mid_center4 = list(zip(time_points, [1.0, 1.0, -1.0, -1.0, -1.0, -1.0]))
        master_mid_center5 = list(zip(time_points, [1.0, 1.0, 1.0, -1.0, -1.0, -1.0]))
        master_mid_center6 = list(zip(time_points, [1.0, 1.0, 1.0, 1.0, -1.0, -1.0]))
        master_mid_center7 = list(zip(time_points, [1.0, 1.0, 1.0, 1.0, 1.0, -1.0]))
        master_mid_center8 = list(zip(time_points, [1.0, 1.0, 1.0, 1.0, 1.0, -1.0]))
        master_high_profile = list(zip(time_points,[1.0, 1.0, 1.0, 1.0, 1.0, -1.0]))

        slave_vol_profile = list(zip(time_points,  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))

        slave_low_profile = list(zip(time_points, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
        slave_mid_center1 = list(zip(time_points, [-1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
        slave_mid_center2 = list(zip(time_points, [-1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
        slave_mid_center3 = list(zip(time_points, [-1.0, -1.0, 1.0, 1.0, 1.0, 1.0]))
        slave_mid_center4 = list(zip(time_points, [-1.0, -1.0, 1.0, 1.0, 1.0, 1.0]))
        slave_mid_center5 = list(zip(time_points, [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]))
        slave_mid_center6 = list(zip(time_points, [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]))
        slave_mid_center7 = list(zip(time_points, [-1.0, -1.0, -1.0, -1.0, -1.0, 1.0]))
        slave_mid_center8 = list(zip(time_points, [-1.0, -1.0, -1.0, -1.0, -1.0, 1.0]))
        slave_high_profile = list(zip(time_points, [-1.0, -1.0, -1.0, -1.0, -1.0,-1.0]))

        master_profile = TransitionProfile(self.len_dbeats, master_vol_profile, master_low_profile, master_mid_center1,
                                           master_mid_center2, master_mid_center3, master_mid_center4,
                                           master_mid_center5, master_mid_center6, master_mid_center7,
                                           master_mid_center8, master_high_profile)
        slave_profile = TransitionProfile(self.len_dbeats, slave_vol_profile, slave_low_profile, slave_mid_center1,
                                          slave_mid_center2, slave_mid_center3, slave_mid_center4, slave_mid_center5,
                                          slave_mid_center6, slave_mid_center7, slave_mid_center8, slave_high_profile)

        self.master_profile = master_profile
        self.slave_profile = slave_profile

    def apply(self, master_audio, new_audio, tempo):

        '''
            Applies this transition, i.e. the low, mid and high profiles, to the input audio.
            The master song is faded out, and the new audio is faded in.

            :master_audio 	The master audio, which should be cropped so that the first queue point is the first sample in this buffer
            :new_audio		The new audio to be mixed with the master audio
        '''

        if self.master_profile == None or self.slave_profile == None:
            raise Exception('Master and slave profile must be set. Call optimize(...) before applying!')

        output_audio = master_audio  # shallow copy

        # Calculate the necessary offsets
        fade_len = self.slave_profile.len_dbeats
        fade_len_samples = int(fade_len * (60.0 / tempo) * 4 * 44100)

        # Perform the fade-out of the master audio first (not yet overlapped with rest of new_audio)

        master_audio_fadeout = master_audio[:fade_len_samples, :]
        output_audio[:fade_len_samples, :] = self.master_profile.apply(master_audio_fadeout)
        output_audio[fade_len_samples:, :] = 0

        # Current situation:
        #			|q1		 |q2		|end
        # MASTER:	===========------....
        # SLAVE:	nothing yet

        # Perform the fade-in and add it to the faded out master audio
        new_audio_fadein = new_audio[:fade_len_samples, :];
        new_audio_rest = new_audio[fade_len_samples:, :];
        new_audio_fadein = self.slave_profile.apply(new_audio_fadein);
        crop = min(fade_len_samples,len(output_audio))
        output_audio[:crop  , :] = output_audio[:crop, :] + new_audio_fadein[:crop, :]

        # Current situation:
        #			|q1		 |q2		|end
        # MASTER:	===========------....
        # SLAVE:	.....----============

        output_audio = output_audio[:fade_len_samples, :];
        logger.info("sfdsdfsdf");
        logger.info(np.shape(new_audio_rest));
        output_audio = np.append(output_audio, new_audio_rest, axis=0);
        logger.info(np.shape(output_audio));

        # Current situation:
        #			|q1		 |q2		|end
        # MASTER:	===========------....
        # SLAVE:	.....----==========================

        # Apply (self-invented) loudness balancing
        loudness_balance_profile = np.zeros((fade_len_samples, 2))
        for j in range(len(self.master_profile.volume_profile) - 1):
            # Get the positions in the audio array and the start and end volumes
            start_dbeat, start_volume_master = self.master_profile.volume_profile[j]
            end_dbeat, end_volume_master = self.master_profile.volume_profile[j + 1]
            _, start_volume_slave = self.slave_profile.volume_profile[j]
            _, end_volume_slave = self.slave_profile.volume_profile[j + 1]
            # Select the correct part of the audio corresponding to this segment
            start_idx = int(fade_len_samples * float(start_dbeat) / fade_len)
            end_idx = int(fade_len_samples * float(end_dbeat) / fade_len)
            # Calculate the loudness profile of this part using the formula:
            # beta = sqrt(v1^2 + v2^2)
            master_vol_profile = linear_fade_volume(np.ones((end_idx - start_idx, 2)), start_volume_master,
                                                    end_volume_master);
            logger.info(np.shape(master_vol_profile))
            slave_vol_profile = linear_fade_volume(np.ones((end_idx - start_idx, 2)), start_volume_slave,
                                                   end_volume_slave);
            loudness_balance_profile[start_idx:end_idx, :] = np.sqrt(
                1.0 / (master_vol_profile ** 2 + slave_vol_profile ** 2))

        output_audio[:fade_len_samples, :] *= loudness_balance_profile

        return output_audio
