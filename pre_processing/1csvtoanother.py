import csv
reader = csv.reader(open('./twitDB_sarcasm.csv', 'r'))
writer = csv.writer(open('sarcasm.csv', 'w'))
index=0
for row in reader:
	if index<50000:
		writer.writerow(row)
		index+=1