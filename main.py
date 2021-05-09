#tempo #to ask

from fileinput import filename

from songcollection import SongCollection # to use SongCollection class from songcollection.py
from tracklister import TrackLister # look above
from djcontroller2 import DjController
from djcontroller2 import DjController # look above
# from console import DjController
import essentia # Open-source library and tools for audio and music analysis, description and synthesis
import os # speaks to OS, works with process IDs, works with directories etc

import logging  #  debug(), info(), warning(), error() and critical()....https://docs.python.org/3/library/logging.html
#~ LOG_LEVEL = logging.INFO
LOG_LEVEL = logging.DEBUG
LOGFORMAT = "%(log_color)s%(message)s%(reset)s" # assigning a text value for later use in 3 lines
from colorlog import ColoredFormatter  # `colorlog.ColoredFormatter` is a formatter for use with Python's `logging` module that outputs records using terminal colors.
logging.root.setLevel(LOG_LEVEL)
formatter = ColoredFormatter(LOGFORMAT)
stream = logging.StreamHandler()
stream.setLevel(LOG_LEVEL)
stream.setFormatter(formatter)
logger = logging.getLogger('colorlogger')
logger.setLevel(LOG_LEVEL)
logger.addHandler(stream)
logging.basicConfig(filename='logfile.log',filemode='w') ## sushant
if __name__ == '__main__':
	
	sc = SongCollection() # function call that will call the songcollection constuctor in songcollection.py and return a SC object. this object is basically initialized with a bunch of blank values
	tl = TrackLister(sc)
	dj = DjController(tl)
	
	essentia.log.infoActive = False
	essentia.log.warningActive = False

	# make my life easy
	sc.load_directory('/home/amit/120 bpm')
	# sc.annotate()
	dj.play(30000,save_mix=False)
	# this whole block deals with CLI commands
	while(True):
		try:
			cmd_split = str.split(input('> : '), ' ')
		except KeyboardInterrupt:
			logger.info('Goodbye!')
			break
		cmd = cmd_split[0]
		if cmd == 'loaddir':
			if len(cmd_split) == 1:
				logger.warning('Please provide a directory name to load!')
				continue
			elif not os.path.isdir(cmd_split[1]):
				logger.warning(cmd_split[1] + ' is not a valid directory!')
				continue
			sc.load_directory(cmd_split[1])
			logger.info(str(len(sc.songs)) + ' songs loaded [annotated: ' + str(len(sc.get_annotated())) + ']')

		if cmd == 'load':
			sc.load_directory('/home/amit/trip')
			logger.info(str(len(sc.songs)) + ' songs loaded [annotated: ' + str(len(sc.get_annotated())) + ']')

		elif cmd == 'play':
			if len(sc.get_annotated()) == 0:
				logger.warning('Use the loaddir command to load some songs before playing!')
				continue
			
			if len(cmd_split) > 1 and cmd_split[1] == 'save':
				logger.info('Saving this new mix to disk!')
				save_mix = True
			else:
				save_mix = False ##amit
				
			logger.info('Starting playback!')
			try:
				dj.play(30000,save_mix=save_mix)
			except Exception as e:
				logger.error(e)
		elif cmd == 'p':
			logger.info('Pausing playback!')
			try:
				dj.pause()
			except Exception as e:
				logger.error(e)
		elif cmd == 'skip' or cmd == 's':
			logger.info('Skipping to next segment...')
			try:
				dj.skipToNextSegment()
			except Exception as e:
				logger.error(e)
		elif cmd == 'stop':
			logger.info('Stopping playback!')
			dj.stop()
		elif cmd == 'save':
			logger.info('Saving the next new mix!')
		elif cmd == 'showannotated':
			logger.info('Number of annotated songs ' + str(len(sc.get_annotated())))
			logger.info('Number of unannotated songs ' + str(len(sc.get_unannotated())))
		elif cmd == 'annotate':
			logger.info('Started annotating!')
			sc.annotate()
			logger.info('Done annotating!')
		elif cmd == 'debug':
			LOG_LEVEL = logging.DEBUG
			logging.root.setLevel(LOG_LEVEL)
			stream.setLevel(LOG_LEVEL)
			logger.setLevel(LOG_LEVEL)
			logger.debug('Enabled debug info. Use this command before playing, or it will have no effect.')
		elif cmd == 'mark':
			dj.markCurrentMaster()		
		else:
			logger.info('The command ' + cmd + ' does not exist!')
		
		
	
