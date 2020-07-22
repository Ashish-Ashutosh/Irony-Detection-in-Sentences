
'''
to convert from a text file to a CSV File
'''
#
# with open('twitDB_regular.txt', 'r') as in_file:
#     stripped = (line.strip() for line in in_file)
#     lines = (line.split(",") for line in stripped if line)
#     with open('regular.csv', 'w') as out_file:
#         for line in lines:
#             out_file.write(" ".join(line) + "\n")
#     #with open('regular.csv', 'w') as out_file:
#         #writer = csv.writer(out_file)
#         #writer.writerows(lines)
#
#
# with open('twitDB_sarcasm.txt', 'r') as in_file:
#     stripped = (line.strip() for line in in_file)
#     lines = (line.split(",") for line in stripped if line)
#     with open('sarcasm.csv', 'w') as out_file:
#         for line in lines:
#             out_file.write(" ".join(line) + "\n")
#

'''
to convert from CSV to .npy file (numpy array) for feature extraction 
'''
import numpy as np
import numpy
#numpy.set_printoptions(threshold=numpy.nan)
csv = np.genfromtxt('preprocessed_regular.csv', delimiter=",")
first = csv[0:,]
print(first)

csvfile = list(csv.reader(open('sarcasmfull.csv', 'rU'),delimiter='\n'))
#for non sarcastic data file
csvfile = list(csv.reader(open('nonsarcasmfull.csv', 'rU'),delimiter='\n'))
data = preprocess(csvfile)

np.save('sarcpreproc',data)
#for non sarcastic data file
#np.save('sarcpreproc',data)



