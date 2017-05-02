import sys
import os

def get_parent_dir(utterance_filename):
	if utterance_filename.find('impro') != -1:
		folder_name = utterance_filename[:len('Ses01F_impro05')]
		return folder_name
	elif utterance_filename.find('script')  != -1:
		folder_name = utterance_filename[:len('Ses01F_script01_2')]
		return folder_name
	else:
		print('failing: ', utterance_filename)
	

if __name__ == "__main__":
	if len(sys.argv) < 2:
		print('Usage: python correct_utterance.py <path_to_file_to_fix>')
	else:
		file_in = sys.argv[1]
		file_to_fix = open(file_in, 'r')
		file_in_split = os.path.split(file_in)
		outfile = os.path.join(file_in_split[0], 'fixed_' + file_in_split[-1])
		file_to_use = open(os.path.join(file_in_split[0], outfile), 'a+')
		for file_path in file_to_fix:
			if file_path == "\n":
				continue
			if file_path.find('impro') != -1:
				file_to_use.write(file_path)
			else:
				# the file was a scripted sample
				utt_id = os.path.split(file_path)[-1]
				parent_dir = get_parent_dir(utt_id)
				if parent_dir:
					path = os.path.join(parent_dir, utt_id)
					file_to_use.write(path)
				else:
					print("failed at: ", utt_id)
		file_to_fix.close()
		file_to_use.close()
		with open(outfile) as f:
			print(f.read())
		f.close()


