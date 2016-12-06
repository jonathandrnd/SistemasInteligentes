__author__="Jonathan Durand"

import sys
from collections import defaultdict
import math
import re

"""
Usage:
python question6.py ner.counts ner_dev.dat > [output_file]

Question 6: Implementation of the Viterbi algorithm plus
new categories for rare data

Calculate emission e(x|y) and trigram probability based on data 
in ner_counts 

Read ner_dev.dat, output prediction to [output_file]
"""

# Obtain the Count(y) for each type of label, as well as Count(x~>y), 
# to calculate emission; then trigram and bigram frequencies
train_counts = file(sys.argv[1],"r")
count_y = dict([('O', 0), ('I-MISC', 0), ('I-PER', 0), ('I-ORG', 0), ('I-LOC', 0), ('B-MISC', 0), ('B-PER', 0), ('B-ORG', 0), ('B-LOC', 0)])
count_xy = dict()
trigram_counts = dict()
bigram_counts = dict()

line = train_counts.readline()
while line:
	parts = line.strip().split(" ")
	line_type = parts[1]
	# Get Count(y) and Count(x~>y)
	if "WORDTAG" in line_type:
		count = parts[0]
		label = parts[2]
		word = parts[3]
		count_y[label] = count_y[label] + int(float(count))
		if word in count_xy:
			count_xy[word].update({label : count})
		else:
			count_xy[word] = {label : count}
	# Get trigram and bigram counts
	else:
		count = parts[0]
		gram_type = parts[1]
		if "2-" in gram_type:
			y1 = parts[2]
			y2 = parts[3]
			bigram = y1 + " " + y2
			bigram_counts[bigram] = count
		elif "3-" in gram_type:
			y1 = parts[2]
			y2 = parts[3]
			y3 = parts[4]
			trigram = y1 + " " + y2 + " " + y3
			trigram_counts[trigram] = count

	line = train_counts.readline()

# Go through dev data, predict tag & compute probability based on model above
dev_data = file(sys.argv[2],"r")
log_probability = 0
# First round for q(*, *, y_1)
first_round = True
line = dev_data.readline()
while line:
	word = line.strip()
	# Check for end of sentence
	if word == '':
		sys.stdout.write("\n")
		log_probability = 0
		first_round = True
	else:
		# Check if there is an existing label associated to the word
		if word in count_xy:
			max_probability = 0
			for label in list(count_xy[word]):
				# Calculate e(x|y)
				emission = float(count_xy[word][label]) / float(count_y[label])
				# Calculate q(y| y_i-2, y_i-1)
				# Check for first round
				if first_round:
					y_2 = "*"
					y_1 = "*" 
					first_round = False
				bigram = y_2 + " " + y1
				trigram = y_2 + " " + y1 + " " + label
				parameter = 0.0000000001
				if trigram in trigram_counts:
					parameter = float(trigram_counts[trigram])/float(bigram_counts[bigram])
				probability = parameter*emission
				if probability > max_probability:
					max_probability = probability
					arg_max = label	

		# If Count(x~>y) = 0, use _RARE_ 
		else:
			if word.isdigit():
				tag = "_NUMBERS_"
			elif re.match("^[0-9_-]+$", word):
				tag = "_NUMBER_CODE_"
			if re.match("^\d\d\.\w\w\w\.\d\d+$", word):
				tag = "_DATE_"
			elif word.isupper():
				tag = "_ALL_UPPER_"
			elif len(word) > 2 and word[0].isupper() and word[1].islower():
				tag = "_FIRST_UPPER_"
			else:
				tag = "_RARE_"
			for label in list(count_xy[tag]):
				# Calculate e(_RARE_|y)
				probability = 0
				emission = float(count_xy[tag][label]) / float(count_y[label])
				# Calculate q(y| y_i-2, y_i-1)
				# Check for first round
				if first_round:
					y_2 = "*"
					y_1 = "*" 
					first_round = False
				bigram = y_2 + " " + y1
				trigram = y_2 + " " + y1 + " " + label
				parameter = 0.0000000001
				if trigram in trigram_counts:
					parameter = float(trigram_counts[trigram])/float(bigram_counts[bigram])
				probability = parameter*emission
				if probability > max_probability:
					max_probability = probability
					arg_max = label

		log_probability = log_probability + math.log(max_probability)
		sys.stdout.write("{} {} {}\n".format(word,arg_max,log_probability))
		#Arrange next round of y_i-2, y_i-1
		y2 = y1
		y1 = arg_max
	line = dev_data.readline()

train_counts.close()
dev_data.close()