import numpy as np
import scipy.interpolate as interp

def stretch(current_song, bar, cues, ff, spleet):


	if (spleet == "audio"):
		if ff < 1:
			#Split the audio into channels
			arr = np.array(current_song.audio[int(44100 * cues[bar]):int(44100 * cues[bar + 1])])
			arrL = arr[:,0]
			arrR = arr[:,1]
			# left
			arrRefL = np.array([1] * int(ff * len(current_song.audio[int(44100*cues[bar]):int(44100*cues[bar+1])])))
			arrL_interp = interp.interp1d(np.arange(arrL.size), arrL)
			arrL_compress = arrL_interp(np.linspace(0, arrL.size - 1, arrRefL.size))
			# right
			arrRefR = np.array([1] * int(ff * len(current_song.audio[int(44100*cues[bar]):int(44100*cues[bar+1])])))
			arrR_interp = interp.interp1d(np.arange(arrR.size), arrR)
			arrR_compress = arrR_interp(np.linspace(0, arrR.size - 1, arrRefR.size))
			# merge L and R
			audio = np.column_stack((arrL_compress, arrR_compress))
		elif ff > 1 or ff == 1:
			# Split the audio into channels
			arr = np.array(current_song.audio[int(44100 * cues[bar]):int(44100 * cues[bar + 1])])
			arrL = arr[:,0]
			arrR = arr[:,1]
			#left
			arrRefL = np.array(([1,]) * int( ff *len(current_song.audio[int(44100*cues[bar]):int(44100*cues[bar+1])])))
			arrL_interp = interp.interp1d(np.arange(arrL.size), arrL)
			arrL_stretch = arrL_interp(np.linspace(0, arrL.size - 1, arrRefL.size)) #np.linspace creates a numeric sequence
			# right
			arrRefR = np.array([1] * int( ff *len(current_song.audio[int(44100*cues[bar]):int(44100*cues[bar+1])])))
			arrR_interp = interp.interp1d(np.arange(arrR.size), arrR)
			arrR_stretch = arrR_interp(np.linspace(0, arrR.size - 1, arrRefR.size))
			# merge L and R
			audio = np.column_stack((arrL_stretch, arrR_stretch))

	elif (spleet == "bass"):
		if ff < 1:
			# Split the audio into channels
			arr = np.array(current_song.bass[int(44100 * cues[bar]):int(44100 * cues[bar + 1])])
			arrL = arr[:, 0]
			arrR = arr[:, 1]
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
			audio = np.column_stack((arrL_compress, arrR_compress))
		elif ff > 1 or ff == 1:
			# Split the audio into channels
			arr = np.array(current_song.vocals[int(44100 * cues[bar]):int(44100 * cues[bar + 1])])
			arrL = arr[:, 0]
			arrR = arr[:, 1]
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
			audio = np.column_stack((arrL_stretch, arrR_stretch))

	elif (spleet == "vocals"):
		if ff < 1:
			# Split the audio into channels
			arr = np.array(current_song.vocals[int(44100 * cues[bar]):int(44100 * cues[bar + 1])])
			arrL = arr[:, 0]
			arrR = arr[:, 1]
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
			audio = np.column_stack((arrL_compress, arrR_compress))
		elif ff > 1 or ff == 1:
			# Split the audio into channels
			arr = np.array(current_song.vocals[int(44100 * cues[bar]):int(44100 * cues[bar + 1])])
			arrL = arr[:, 0]
			arrR = arr[:, 1]
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
			audio = np.column_stack((arrL_stretch, arrR_stretch))

	elif (spleet == "drums"):
		if ff < 1:
			# Split the audio into channels
			arr = np.array(current_song.drums[int(44100 * cues[bar]):int(44100 * cues[bar + 1])])
			arrL = arr[:, 0]
			arrR = arr[:, 1]
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
			audio = np.column_stack((arrL_compress, arrR_compress))
		elif ff > 1 or ff == 1:
			# Split the audio into channels
			arr = np.array(current_song.drums[int(44100 * cues[bar]):int(44100 * cues[bar + 1])])
			arrL = arr[:, 0]
			arrR = arr[:, 1]
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
			audio = np.column_stack((arrL_stretch, arrR_stretch))

	elif (spleet == "other"):
		if ff < 1:
			# Split the audio into channels
			arr = np.array(current_song.other[int(44100 * cues[bar]):int(44100 * cues[bar + 1])])
			arrL = arr[:, 0]
			arrR = arr[:, 1]
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
			audio = np.column_stack((arrL_compress, arrR_compress))
		elif ff > 1 or ff == 1:
			# Split the audio into channels
			arr = np.array(current_song.other[int(44100 * cues[bar]):int(44100 * cues[bar + 1])])
			arrL = arr[:, 0]
			arrR = arr[:, 1]
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
			audio = np.column_stack((arrL_stretch, arrR_stretch))

	elif (spleet == "piano"):
		if ff < 1:
			# Split the audio into channels
			arr = np.array(current_song.piano[int(44100 * cues[bar]):int(44100 * cues[bar + 1])])
			arrL = arr[:, 0]
			arrR = arr[:, 1]
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
			audio = np.column_stack((arrL_compress, arrR_compress))
		elif ff > 1 or ff == 1:
			# Split the audio into channels
			arr = np.array(current_song.piano[int(44100 * cues[bar]):int(44100 * cues[bar + 1])])
			arrL = arr[:, 0]
			arrR = arr[:, 1]
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
			audio = np.column_stack((arrL_stretch, arrR_stretch))

	return audio