import numpy as np
import scipy.interpolate as interp

def stretch(current_song, bar, cues, ff, spleet):
	bass = np.array(current_song.bass[int(44100 * cues[bar]):int(44100 * cues[bar + 1])])
	vocals = np.array(current_song.vocals[int(44100 * cues[bar]):int(44100 * cues[bar + 1])])
	drums = np.array(current_song.drums[int(44100 * cues[bar]):int(44100 * cues[bar + 1])])
	other = np.array(current_song.other[int(44100 * cues[bar]):int(44100 * cues[bar + 1])])
	output = bass +  drums + other
	arrL = output[:, 0]
	arrR = output[:, 1]

	if ff < 1:
		# left
		arrRefL = np.array(
			[1] * int(ff * len(current_song.vocals[int(44100 * cues[bar]):int(44100 * cues[bar + 1])])))
		arrL_interp = interp.interp1d(np.arange(arrL.size), arrL)
		arrL_compress = arrL_interp(np.linspace(0, arrL.size - 1, arrRefL.size))
		# right
		arrRefR = np.array(
			[1] * int(ff * len(current_song.vocals[int(44100 * cues[bar]):int(44100 * cues[bar + 1])])))
		arrR_interp = interp.interp1d(np.arange(arrR.size), arrR)
		arrR_compress = arrR_interp(np.linspace(0, arrR.size - 1, arrRefR.size))
		# merge L and R
		output = np.column_stack((arrL_compress, arrR_compress))

	elif ff > 1 or ff == 1:
		# left
		arrRefL = np.array(
			([1, ]) * int(ff * len(current_song.vocals[int(44100 * cues[bar]):int(44100 * cues[bar + 1])])))
		arrL_interp = interp.interp1d(np.arange(arrL.size), arrL)
		arrL_stretch = arrL_interp(
			np.linspace(0, arrL.size - 1, arrRefL.size))  # np.linspace creates a numeric sequence
		# right
		arrRefR = np.array(
			[1] * int(ff * len(current_song.vocals[int(44100 * cues[bar]):int(44100 * cues[bar + 1])])))
		arrR_interp = interp.interp1d(np.arange(arrR.size), arrR)
		arrR_stretch = arrR_interp(np.linspace(0, arrR.size - 1, arrRefR.size))
		# merge L and R
		output = np.column_stack((arrL_stretch, arrR_stretch))


	return output