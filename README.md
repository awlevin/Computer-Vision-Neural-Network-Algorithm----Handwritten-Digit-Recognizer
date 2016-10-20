# Computer-Vision-Neural-Network-Algorithm----Handwritten-Digit-Recognizer

This is a program that I was required to write during a summer class I took at UW Madison (CS 540 – Introduction to Artificial Intelligence)

The assignment required us to write a neural network that takes an input file of binary data containing a large series of black and white handwritten digits.
The original assignment asked that our program recognize digits between 0-2, and for extra credit we could modify the program to interpret digits 0-9.
This is my (successful) attempt at the extra credit.

The main method of this program is in HW4.java (technically HW4.class within the bin folder). The usage format from terminal is as follows:
usage: java HW4 <numberOfHiddenNode> <learningRate> <maxEpoch> <trainFile> <testFile>

The trainFile is a text file containing binary data of a series of handwritten characters. Each line corresponds to 1 character and 
is a sequence of 256 bits, where each bit is separated by a space, " ", character.. and followed by a classification. Following
each sequence of bits is another series of binary data, where the index of each position corresponds to the number that the data
represents. 

For example:
<characterBits> 0 1 0
would be 1 line in the trainFile. The 0 1 0 means that <characterBits> correspond to the handwritten digit "1". This practice is an example
of trained learning.

The testFile is identical in structure, but the program only uses the identifier to see if it correctly classified each digit.


Increasing the number of <numberOfHiddenNodes> will generally make classifications more accurate (as there will be more possible combinations of 
weights between layers). 

The maxEpoch parameter specifies the number of times to run the training set (on each epoch the training set uses a back-propogation algorithm
to set the weights between layers). The algorithm also uses the sigmoid function as the activation function, as I found this to be more successful
than a step function approach. 

The <learningRate> parameter adjusts the rate at which weights are modified during training. 

Lastly, I'd like to thank the folks at UC Irvine for providing the extensive training set text files that I used in this program. The
files can be reached at https://archive.ics.uci.edu/ml/datasets/Semeion+Handwritten+Digit.
