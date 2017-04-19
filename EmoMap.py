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
        A dictionary that maps utterance (sentence ID) to list of emotions
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

    def get_map(self):
        return self.map

    def set_map(self, new_map):
        self.map = new_map

    def get_filtered_map(self, emotion_list):
        # TODO(quacht): How do you determine the classification of an utterance
        # when the utterance has been classified differently by each annotator?
        emotion_set = set(emotion_list)
        filtered_map = dict(self.emotion_to_utterance_map)
        for key in self.emotion_to_utterance_map:
            if key not in emotion_set:
                del filtered_map[key]
        return filtered_map

def filter_cat_txt(self, cat_txt_file, emotion_list, path_to_new_file):
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

        emotions = line_array[1:-1]
        for emo in emotions:
            emotion = emo[1:-1] 
            if emotion in emotion_set:
                new_file.write(line)

    f.close()
    new_file.close()

    return True

def main():
    # parse command-line arguments
    # if len(sys.argv) < 1:
    #     print "you must call program as:  "
    #     print "   python filter_emotions.py <path_to_categorical_directory>"
    #     sys.exit(1)
    # categorical = sys.argv[1]
    categorical = "../data/EmoEvaluation/Categorical"
    emo_map = EmoMap()
    emo_map.from_categorical_dir(categorical)
    for emotion in emo_map.emotion_to_utterance_map:
        print(emotion, len(emo_map.emotion_to_utterance_map[emotion]))
    filtered_map = emo_map.get_filtered_map(['Anger', 'Happiness', 'Sadness', 'Neutral state'])
    print()
    print('FILTERED')
    for emotion in filtered_map:
        print(emotion, len(filtered_map[emotion]))



if __name__ == "__main__":
    main()
