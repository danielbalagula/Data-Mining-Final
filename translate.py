import sys
import csv
from unidecode import unidecode
from transliterate import translit, get_available_language_codes

new_rows = []
ru_file = str(sys.argv[1])
en_file = "translated_" + ru_file

with open(ru_file, 'rb') as csvfile:
	
	reader = csv.reader(csvfile, delimiter=',')
	writer = csv.writer(csvfile)

	for row in reader:
		new_row = row
		for column in range(1,6):
			new_row[column] = translit(new_row[column].decode('utf-8'), 'ru', reversed=True).encode('ascii','ignore')
		new_rows.append(new_row)

with open(en_file, 'wb') as csvfile:

	writer = csv.writer(csvfile)
	writer.writerows(new_rows)