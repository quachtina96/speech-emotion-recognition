# Author: Tina Quach (quacht@mit.edu)

# This script manages set of utterances in the IEMOCAP dataset and its 
# corresponding categorical emotion label.

import sys
import os
from collections import defaultdict

def init_dict(path_to_categorical_directory, emotion_is_key=False):
	"""Initialize the map.
	
	Args:
		path_to_categorical_directory: String path to "Categorical" directory in 
		EmoEvaluation
	Returns:
		A dictionary that maps utterance (sentence ID) to list of emotions when  
		emotion_is_key is False. When in emotion_is_key is True, return the 
		opposite: A dictionary that maps emotion to list of utteranceIDs that
		were classified with that emotion by at least one annotator. 
	"""
	utterance_to_emotion_map = defaultdict(list)
	emotion_to_utterance_map = defaultdict(list)
	
	files = os.listdir(path_to_categorical_directory)
	for file in files:
		if file.endswith("cat.txt"):
			f = open(os.path.join(path_to_categorical_directory, file), 'r')
			for line in f:
				line_array = line.split(' :')
				utteranceID = line_array[0]

				emotions = ' '.join(line_array[1:]).split('; ')[:-1]
				if emotion_is_key:
					for emotion in emotions:
						emotion_to_utterance_map[emotion].append(utteranceID)
				else:
					utterance_to_emotion_map[utteranceID].append(emotions)

	f.close()
	if emotion_is_key:
		return emotion_to_utterance_map
	else:
		return utterance_to_emotion_map


class EmoMap():
	def __init__(self):
		self.path = None
		self.utterance_to_emotion_map = None
		self.emotion_to_utterance_map = None
	
	def from_maps(self, utterance_to_emotion_map, emotion_to_utterance_map, opt_path=None):
		if opt_path:
			self.path = opt_path
		self.utterance_to_emotion_map = utterance_to_emotion_map
		self.emotion_to_utterance_map = emotion_to_utterance_map

	def from_categorical_dir(self, path_to_categorical_directory):
		self.path = path_to_categorical_directory
		# Maps utteranceIDs to their respective labels in the form of a list of 
		# lists, where each sub-list corresponds to the emotions recognized by a 
		# single annotator. 
		self.utterance_to_emotion_map = init_dict(path_to_categorical_directory)
		# Maps each emotion to all utterances that were labeled with that emotion
		# at least once
		self.emotion_to_utterance_map = init_dict(path_to_categorical_directory, emotion_is_key=True)

	def get_emotion_to_utterance_map(self):
		return self.map

	def set_emotion_to_utterance_map(self, new_map):
		self.emotion_to_utterance_map = new_map

	def get_utterance_to_emotion_map(self):
		return self.utterance_to_emotion_map

	def set_utterance_to_emotion_map(self, new_map):
		self.utterance_to_emotion_map = new_map

	def get_filtered_map(self, emotion_list):
		# Return map of emotion to utteranceID, including only the emotions in 
		# the emotion list.
		emotion_set = set(emotion_list)
		filtered_map = dict(self.emotion_to_utterance_map)
		for key in self.emotion_to_utterance_map:
			if key not in emotion_set:
				del filtered_map[key]
		return filtered_map

	def get_utterance_ids(self, opt_emotion_list):
		if opt_emotion_list:
			filtered = self.get_filtered_map(opt_emotion_list)
			utteranceIDs = set([val for id_list in filtered.values() for val in id_list])
			return list(utteranceIDs)
		else:
			return self.utterance_to_emotion_map.keys()

def filter_cat_txt(cat_txt_file, emotion_list, path_to_new_file):
	"""Given a path to a "cat.txt" file, save a new "filtered.cat.txt" file 
	listing only sessions that include the emotions desired. Return True if 
	success.
	
	Args: 
		cat_txt_file: Path to the cat.txt file that categorizes the emotion 
			of utterances listed in the file.
		emotion_list: List of emotions to include in the filtered list.

	Raises:
		InputError for files that cannot be opened.
	
	Possible emotions include: Neutral state, Frustration, Anger, Disgust, 
		Sadness, Fear, Excited, Happiness, Surprise """
	try:
		f = open(cat_txt_file, 'r')
	except:
		raise InputError("Could not open path to the cat.txt file.")

	emotion_set = set(emotion_list)

	new_file = open(path_to_new_file, 'a')
	
	for line in f:
		line_array = line.split(' ')
		utteranceID = line_array[0]
		new_file.write(line)
	f.close()
	new_file.close()

	return True

def get_parent_dir(utterance_filename):
	if utterance_filename.find('impro'):
		folder_name = utterance_filename[:len('Ses01F_impro05')]
	elif utterance_filename.find('script'):
		folder_name = utterance_filename[:len('Ses01F_script01_2')]
	return folder_name
	
def save_utterance_id_list(path_to_categorical_directory, path_to_iemocap_sentences_wav, path_to_utterance_id_list):
	emo_map = EmoMap()
	emo_map.from_categorical_dir(path_to_categorical_directory)
	desired_emotions = ['Anger', 'Happiness', 'Sadness', 'Neutral state']
	utterance_ids = emo_map.get_utterance_ids(desired_emotions)
	utterance_ids.sort()
	with open(path_to_utterance_id_list, 'a') as f:
		for utt_id in utterance_ids:
			parent_dir = get_parent_dir(utt_id)
			path = os.path.join(parent_dir, utt_id + '.wav')
			f.write(path + '\n')

if __name__ == "__main__":
	path_to_categorical_directory = "../data/EmoEvaluation/Categorical"
	path_to_utterance_id_list = os.path.join(path_to_categorical_directory, 'utteranceIDs.txt')
	path_to_iemocap_sentences_wav = "../data/sentence_wav_sample/"
	save_utterance_id_list(path_to_categorical_directory, path_to_iemocap_sentences_wav, path_to_utterance_id_list)
