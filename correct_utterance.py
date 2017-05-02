import sys
import os

def get_parent_dir(utterance_filename):
	if utterance_filename.find('impro') != -1:
		folder_name = utterance_filename[:len('Ses01F_impro05')]
	elif utterance_filename.find('script')  != -1:
		folder_name = utterance_filename[:len('Ses01F_script01_2')]
	return folder_name

if __name__ == "__main__":
	if len(sys.argv) == 0:
		print('Usage: python correct_utterance.py <path_to_file_to_fix>')
	else:
		file_in = sys.argv[0]
		file_out = 'testing.txt'
		file_to_fix = open(file_in, 'r')
		file_to_use = open(file_out, 'a')
		for file_path in file_to_fix:
			if file_path.find('impro') != -1:
				file_to_use.write(file_path)
			else:
				# the file was a scripted sample
				utt_id = os.path.split(file_path)[-1]
				parent_dir = get_parent_dir(utt_id)
				path = os.path.join(parent_dir, utt_id)
				file_to_use.write(path)
		file_to_fix.close()
		file_to_use.close()


