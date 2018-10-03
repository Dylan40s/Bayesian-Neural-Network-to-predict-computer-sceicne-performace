from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from edward.models import Normal
import os

tf.flags.DEFINE_integer("T", default=1244, help="Total Number of data points.")
tf.flags.DEFINE_integer("D", default=7, help="Number of features.")
tf.flags.DEFINE_integer("O", default=4, help="Number of outputs.")

FLAGS = tf.flags.FLAGS


#used for episdemic uncertainty
#log is base 2 but clamped on 0 and 1 so log works
def predictive_entropy(prob):
	for x in range(len(prob)):
		prob[x] = max(min(prob[x], 1), 0)
	return -np.sum(np.log(prob[prob != 0]) * prob[prob != 0])

#NN part used to multiply
# 5 layer network, 7 input nodes -> h1 -> h2 -> 4 output nodes
def neural_network(X,W_0,W_1,W_2,b_0,b_1,b_2,N):
	h1= tf.nn.tanh(tf.matmul(X, W_0) + b_0)
	h2= tf.nn.tanh(tf.matmul(h1, W_1) + b_1)
	output = tf.nn.sigmoid(tf.matmul(h2, W_2) + b_2)
	return tf.reshape(output, [N,FLAGS.O])

# Used to train
# the 7 input fields and csresults are returned
def create_file_reader_ops(filename_queue):
	reader = tf.TextLineReader(skip_header_lines=1)
	_, csv_row = reader.read(filename_queue)
	# change this to change the number of fields taken in
	record_defaults = [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0],[0.0],[0.0], [""], [0.0]]
	AnonID, UCTScore,Eng, Math,Phy, NBTAL,NBTQL,NBTMath,Course, csresult = tf.decode_csv(csv_row, record_defaults=record_defaults)
	features = tf.stack([ UCTScore,Eng, Math,Phy, NBTAL,NBTQL,NBTMath] )
	return features, csresult

# Used to train, changes the csresult to a pass fail class output
# returns list of features and list of updated y values
def getData(usedFile):
	sess = ed.get_session()
	dir_path = os.path.dirname(os.path.realpath(__file__))
	filename = dir_path + "/" + usedFile
	filename_queue = tf.train.string_input_producer([filename], shuffle=True)
	students, marks = create_file_reader_ops(filename_queue)
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)
	xdata = []
	ydata = []
	# Gets the entries from the dataset and convert to pass or fail class
	for i in range(FLAGS.T):
		x,y = sess.run([students,marks])
		xdata.append(x)
		if y >= 0.75:
		  ydata.append([1.0,0.0,0.0,0.0])
		elif y >=0.60 and y <= 0.74:
		  ydata.append([0.0,1.0,0.0,0.0])
		elif y >= 0.50 and y <= 0.59:
		  ydata.append([0.0,0.0,1.0,0.0])
		else:
		  ydata.append([0.0,0.0,0.0,1.0])
		
	return xdata, ydata
def loadParams():
	returned = []
	hidden = [10,20,50]
	iter1 = [1000,10000]
	params = [0.1, 0.3]
	for i in range(len(hidden)):
		for h2 in range(len(hidden)):
			for para1 in range(len(params)):
				for para2 in range(len(params)):
						for ite in range(len(iter1)):
							name = str(hidden[i]) + "H"+str(hidden[h2]) + "H" + str(params[para1]) +"P1"+ str(params[para2]) + "P2" + str(params[para2]) + "P3" + str(iter1[ite])
							returned.append([name, hidden[i],hidden[h2],params[para1],params[para2], iter1[ite]])
	return returned

def save(arr,xdata,ydata):
	tf.reset_default_graph()

	trainSetNumber = round(FLAGS.T* 0.8)

	x_train = xdata[:trainSetNumber]
	y_train = ydata[:trainSetNumber]
	x_test = xdata[trainSetNumber:]
	y_test = ydata[trainSetNumber:]

	x_train = np.asarray(x_train)
	x_test = np.asarray(x_test)

	x_train = np.asarray(x_train)
	x_test = np.asarray(x_test)
	# print(x_test)
	# print(y_test)
	pos = 0
	name = arr[pos]
	pos +=1
	H1 = int(arr[pos])
	pos+=1
	H2 = int(arr[pos])
	pos+=1
	param1 = float(arr[pos])
	pos += 1
	param2 = float(arr[pos])

	graph1 = tf.Graph()
	with graph1.as_default():
		with tf.name_scope("model"):
			W_0 = Normal(loc=tf.zeros([FLAGS.D, H1]), scale=param1*tf.ones([FLAGS.D,H1 ]),name="W_0")
			W_1 = Normal(loc=tf.zeros([H1, H2]), scale=param2*tf.ones([H1, H2]), name="W_1")
			W_2 = Normal(loc=tf.zeros([H2, FLAGS.O]), scale=param2*tf.ones([H2, FLAGS.O]), name="W_2")
			b_0 = Normal(loc=tf.zeros(H1), scale=param1 *tf.ones(H1), name="b_0")
			b_1 = Normal(loc=tf.zeros(H2), scale=param2* tf.ones(H2), name="b_1")
			b_2 = Normal(loc=tf.zeros(FLAGS.O), scale=param2* tf.ones(FLAGS.O), name="b_2")

			X = tf.placeholder(tf.float32, [trainSetNumber, FLAGS.D], name="X")
			y = Normal(loc=neural_network(x_train,W_0, W_1, W_2, b_0, b_1, b_2, trainSetNumber), scale=0.1*tf.ones([trainSetNumber,FLAGS.O]), name="y")
		
		with tf.variable_scope("posterior",reuse=tf.AUTO_REUSE):
			with tf.variable_scope("qW_0",reuse=tf.AUTO_REUSE):
			    loc = tf.get_variable("loc", [FLAGS.D, H1])
			    scale = param1*tf.nn.softplus(tf.get_variable("scale", [FLAGS.D, H1]))
			    qW_0 = Normal(loc=loc, scale=scale)
			with tf.variable_scope("qW_1",reuse=tf.AUTO_REUSE):
			    loc = tf.get_variable("loc", [H1, H2])
			    scale = param2*tf.nn.softplus(tf.get_variable("scale", [H1, H2]))
			    qW_1 = Normal(loc=loc, scale=scale)
			with tf.variable_scope("qW_2",reuse=tf.AUTO_REUSE):
			    loc = tf.get_variable("loc", [H2, FLAGS.O])
			    scale = param2*tf.nn.softplus(tf.get_variable("scale", [H2, FLAGS.O]))
			    qW_2 = Normal(loc=loc, scale=scale)
			with tf.variable_scope("qb_0",reuse=tf.AUTO_REUSE):
			    loc = tf.get_variable("loc", [H1])
			    scale =param1 * tf.nn.softplus(tf.get_variable("scale", [H1]))
			    qb_0 = Normal(loc=loc, scale=scale)
			with tf.variable_scope("qb_1",reuse=tf.AUTO_REUSE):
			    loc = tf.get_variable("loc", [H2])
			    scale =param2 * tf.nn.softplus(tf.get_variable("scale", [H2]))
			    qb_1 = Normal(loc=loc, scale=scale)
			with tf.variable_scope("qb_2",reuse=tf.AUTO_REUSE):
			    loc = tf.get_variable("loc", [FLAGS.O])
			    scale =param2 * tf.nn.softplus(tf.get_variable("scale", [FLAGS.O]))
			    qb_2 = Normal(loc=loc, scale=scale)
	#inference
	with tf.Session(graph=graph1) as sess:
		# Set up the inference method, mapping the prior to the posterior variables
		inference = ed.KLqp({W_0: qW_0, b_0: qb_0,W_1: qW_1, b_1: qb_1,W_2: qW_2, b_2: qb_2}, data={X: x_train, y: y_train})
		# Set up the adam optimizer
		global_step = tf.Variable(0, trainable=False)
		starter_learning_rate = 0.1
		learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,100, 0.3, staircase=True)
		optimizer = tf.train.AdamOptimizer(learning_rate)

		# Run the inference method
		pos += 1
		iter1 = arr[pos]
		inference.run(n_iter=iter1,optimizer=optimizer ,n_samples=5)

		#Run the test data through the neural network
		infered = neural_network(x_test, qW_0, qW_1, qW_2, qb_0, qb_1, qb_2, len(x_test))
		inferedList = infered.eval()

		#Accuracy checks on the data (The test data)
		# In order to work with PPC and other metrics, it must be a random variables
		# Normal creates this random varaibles by sampling from the poterior with a normal distribution
		NormalTest =Normal(loc=neural_network(x_test, qW_0, qW_1, qW_2, qb_0, qb_1, qb_2,len(x_test)), scale=0.1*tf.ones([len(x_test),FLAGS.O]), name="y_other") 
		NormalTestList = NormalTest.eval()
		
		# Change the graph so that the posterior point to the output
		y_post = ed.copy(NormalTest, {W_0: qW_0, b_0: qb_0,W_1: qW_1, b_1: qb_1,W_2: qW_2, b_2: qb_2})
		X = tf.placeholder(tf.float32, [len(x_test), FLAGS.D], name="X")
		y_test_tensor = tf.convert_to_tensor(y_test)
		MSE = ed.evaluate('mean_squared_error', data={X: x_test, NormalTest: y_test_tensor})
		MAE =ed.evaluate('mean_absolute_error', data={X: x_test, NormalTest: y_test_tensor})
		# PPC calculation
		PPCMean = ed.ppc(lambda xs, zs: tf.reduce_mean(xs[y_post]), data={y_post:  y_test, X:x_test}, latent_vars={W_0: qW_0, b_0: qb_0,W_1: qW_1, b_1: qb_1,W_2: qW_2, b_2: qb_2}, n_samples=5)
		# Change the graph again, this is done to do epistemic uncertainty calculations
		posterior = ed.copy(NormalTest, dict_swap={W_0: qW_0.mean(), b_0: qb_0.mean(),W_1: qW_1.mean(), b_1: qb_1.mean(),W_2: qW_2.mean(), b_2: qb_2.mean()})
		Y_post1 = sess.run(posterior.sample(len(x_test)), feed_dict={X: x_test, posterior: y_test})
		mean_prob_over_samples=np.mean(Y_post1, axis=0) ## prediction means
		prediction_variances = np.apply_along_axis(predictive_entropy, axis=1, arr=mean_prob_over_samples)
		
		# Run analysis on test data, to see how many records were correct
		classes, actualClass, cor, firsts, seconds, thirds, fails, perCorrect = Analysis(inferedList, y_test)
		# Save the model through TF saver
		saver = tf.train.Saver()
		dir_path = os.path.dirname(os.path.realpath(__file__))
		save_path = saver.save(sess, dir_path +"/"+name+"/model.ckpt")
		print("Model saved in path: %s" % save_path)

		file = open(dir_path+"/"+name +"/"+name+".csv",'w')
		file.write("MSE = " + str(MSE))
		file.write("\nMAE = " + str(MAE))
		file.write("\nPPC mean = " + str(PPCMean))
		file.write("; Predicted First;Predicted Second; Predicted Third; Predicted Fail \n")
		classNames = ['First','Second', 'Third', 'Fail']
		for x in range(len(firsts)):
			file.write(classNames[x] + ";" + str(firsts[x]) + ";" + str(seconds[x])+ ";" + str(thirds[x])+ ";" + str(fails[x]) + "\n")
		file.write("Num;Class 1;Class 2;Class 3;Class 4;Epi;Predicted Class;Correct Class\n ")
		for x in range(len(inferedList)):
			line = str(x) 
			for i in range(len(inferedList[x])):
				line += ";" + str(round(inferedList[x][i],2))
			line += ";" + str(round(prediction_variances[x],2)) + ";" + str(classes[x]+1) + ";" + str(actualClass[x]+1) + "\n"
			file.write(line) 
		file.close()

		return perCorrect

# Used to compare the resutls of the NN to the true results and outputing a returns the elements of the confusion matrix
def Analysis(inferedList, y_test):

	firsts = [0,0,0,0]
	seconds = [0,0,0,0]
	thirds = [0,0,0,0]
	fails = [0,0,0,0]

	Correct0_49 = 0
	Correct50_59 = 0
	Correct60_74 = 0
	Correct75_100 = 0
	InCorrect0_49 = 0
	InCorrect50_59 = 0
	InCorrect60_74 = 0
	InCorrect75_100 = 0
	correctPreditions = 0
	incorrectPredictions = 0
	classes = []
	cor = []
	actualClass = []
	for i in range(len(y_test)):
		postList = inferedList[i]
		actualList = y_test[i]
		actualPos = -1
		postPos = -1
		postValue = -1
		actualValue = -1
		for j in range(FLAGS.O):
			if postList[j] > postValue:
				postValue = postList[j]
				postPos = j
			if (actualList[j] ==1):
				actualValue = actualList[j]
				actualPos = j
		classes.append(postPos)
		actualClass.append(actualPos)
		if (actualPos == 0):
			firsts[postPos] += 1
		elif (actualPos == 1):
			seconds[postPos] +=1
		elif (actualPos == 2):
			thirds[postPos] += 1
		else:
			fails[postPos] += 1
		if postPos == actualPos:
			correctPreditions += 1
			cor.append(True)	
		else:
			incorrectPredictions += 1
			cor.append(False)

	# This can be uncommented if a single test is run to show results of hold out data test

	# print("Correct75_100 = " + str(firsts[0]))
	# print("Correct60_74 = " + str(seconds[1]))
	# print("Correct50_59 = " + str(thirds[2]))
	# print("Correct0_49 = " + str(fails[3]))
	# print("InCorrect75_100 = " + str(firsts[1] + firsts[2] + firsts[3]))
	# print("InCorrect60_74 = " + str(seconds[0] + seconds[2] + seconds[3]))
	# print("InCorrect50_59 = " + str(thirds[0] + thirds[1] + thirds[3]))
	# print("InCorrect0_49 = " + str(fails[0] + fails[1] + fails[3]))
	# print("correct Preditions " + str(correctPreditions))
	# print("Incorrect predictions " + str(incorrectPredictions))
	# print("Correct predictions = " + str((correctPreditions)/len(inferedList)))
	perCorrect = (correctPreditions)/len(inferedList)
	return classes, actualClass, cor, firsts, seconds, thirds, fails, perCorrect

def main():
	arr = loadParams()
	#Change this to take in another file
	xdata,ydata = getData("AllYearsCSC1016Classes.csv")
	resultFile = open("ResultFile.csv",'w')

	current = -1
	fileName = ""
	for i in range(2):
		corPred = save(arr[i],xdata,ydata)
		if corPred > current:
			current = corPred
			fileName = arr[i][0]
		resultFile.write(str(corPred) + ";" + arr[i][0] + "\n")
	print(fileName + " best file")
	print(str(current) + "'%' highest correct Predictions")
	resultFile.write("Best Entry \n")
	resultFile.write(str(current) + ";" +fileName)
	resultFile.close()

if __name__ == '__main__':
  main()