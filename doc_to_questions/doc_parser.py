"""
Copyright (C) Tata Consultancy Services - All Rights Reserved
Unauthorized copying of this file, via any medium is strictly prohibited
Proprietary and confidential
Written by Tushar Nitave <tushar.nitave@tcs.com> and Pushpam Punjabi <email>, 2019
"""

import os
import sys
from docx  import Document
from textblob import TextBlob
from colors import colors
from qg_engine import qg_engine
import time
from unidecode import unidecode
import warnings
warnings.filterwarnings("ignore")

HEADINGS = []
PARAGRAPHS = []
QUESTIONS = []


def get_data(file_name):
	""" loads specified file and extracts headings and paragraphs """
	doc = Document(file_name)


	for i in doc.paragraphs:
		for run in i.runs:
			if run.bold:
				HEADINGS.append(run.text.lower())
			else:
				if i.text:
					PARAGRAPHS.append(i.text.lower())
					break
	print('********************************************************\n')
	print("HEADINGS : ", HEADINGS) 
	print("\n*********len(HEADINGS)***\n",len(HEADINGS))
	print("PARAGRAPHS : ", PARAGRAPHS)               
	print('********************************************************\n')
	print("\n*********len(PARAGRAPHS)***\n",len(PARAGRAPHS))

def generate_question():
	""" generates questions from heading """
	switch = False

	for heading in HEADINGS:

		if heading:
			sentence = TextBlob(heading.lower())

			for tags in sentence.tags:
				# search for plural tags
				if tags[1] == 'NNS':
					switch = True
					break
			if switch:
				gen_que = 'What are' + ' ' + heading.lower() + ' ' + '?'
				QUESTIONS.append(gen_que)
				switch = False
			else:
				gen_que = 'What is' + ' ' + heading.lower() + ' ' + '?'
				QUESTIONS.append(gen_que)

	print("*****************QUESTIONS from headings",QUESTIONS)
	print("\n*********len(questions)***\n",len(QUESTIONS))
	print("*****************Answers for headings",PARAGRAPHS)
	print("\n*********len(PARAGRAPHS)***\n",len(PARAGRAPHS))

	# print('++++++++++++++++++++++++++++++\n',QUESTIONS)



def write_to_file(file_name):
	"""writes question-answer pairs to a csv file"""
	print("***********writing**********\n\n")
	with open("output/"+file_name+"_qa.csv", "w") as file:
			for data in range(len(QUESTIONS)):
				# print("\nwritten\n",QUESTIONS[data])
				file.write(QUESTIONS[data])
				print()
				file.write("\t")
				file.write(PARAGRAPHS[data])
				# print("\nwritten\n",PARAGRAPHS[data])
				file.write("\n")
			file.close()


if __name__ == "__main__":

	start_time = time.time()

	errors = colors.colors.color_codes(None, 1)
	blank = colors.colors.color_codes(None, 0)
	success = colors.colors.color_codes(None, 2)

	try:
		if len(sys.argv) == 1:
			print(errors,"\n[Error]:",blank,"enter file name\n")
			sys.exit()

		# get file from user
		file_path = sys.argv[1]

		# validate if given file is present
		if not os.path.isfile(file_path):
			print(errors,"\n[Error]",blank,"File not found!\n")
			sys.exit()

		# get file extension
		file_name, file_ext = os.path.basename(file_path).split(".")

		# check if file is docx format
		if file_ext != 'docx':
			print("\n\'.{}\' is not a valid format\n".format(file_ext))
			sys.exit()

		# check if file is empty
		if os.stat(file_path).st_size == 0:
			print(errors,"\n[Error]",blank,"File is empty!\n")
			sys.exit()

		get_data(file_path)
		# print("************************GET DATA Succesfull********************************")
		generate_question()
		write_to_file(file_name)

		qg_engine(file_name)

		print(success,"\nQuestion Generation Succesfull !\n", blank)

		print("Time Elapsed: ", int(time.time() - start_time),"s\n")


	except ValueError:
		print("[Error] Not a valid file type")
		sys.exit()


