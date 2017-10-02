#Need to create directory "./l1_filter_visualization" for storing filters

import matplotlib.image as mpimg
import tensorflow as tf
import numpy as np
import os, os.path
from io import BytesIO
import pickle
import matplotlib.pyplot as plt
import cv2
from functools import partial
import PIL.Image
from IPython.display import clear_output, Image, display, HTML

train_size = 500#50000
test_size = 500#5000
n_classes = 1#10
N0 = 527 #image height or width
n_channel = 3
batch_size = 16#256#128
rate = 0.0002
decay = 0.65
schedule = 100
training_itr = 1000
threshold = 1#0.71
goodUntil = 3
display_step = 1
data_path = ""#enter the data path
beta = 0.01


def readImageFrom(path):
    global N0
    imgs = []
    valid_images = [".jpg"]
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        imgs.append(mpimg.imread(os.path.join(path,f)))
    X = np.array(imgs, dtype = 'float32')
    return X

def readLabelFrom(path):
    label = []
    with open(path) as labels:
        for line in labels:
            label.append(line)
    Y = np.array(label, dtype = 'int')
    return Y

def readDataFromPath(path):
    global train_size, test_size, n_classes
    #data_file = file(path,"rb")
    #train_x, train_y, test_x, test_y = pickle.load(data_file)
    
    train_x = readImageFrom("./train_img/")
    train_y = readLabelFrom("./label/train_label_class.txt")
    test_x = readImageFrom("./test_img/")
    test_y = readLabelFrom("./label/test_label_class.txt")
    
    #data_file.close()
    #type casting, and scaling
    train_x = np.float32(train_x)
    train_x = train_x/255.0
    test_x = np.float32(test_x)
    test_x = test_x/255.0
    #b = np.zeros((train_size, n_classes))
    #b[np.arange(train_size), train_y] = 1
    train_y = np.float32(train_y)
    #c = np.zeros((test_size, n_classes))
    #c[np.arange(test_size), test_y] = 1
    test_y = np.float32(test_y)
    #centering
    train_mean = np.mean(train_x, axis=0)
    test_mean = np.mean(test_x, axis=0)
    train_x = np.subtract(train_x, train_mean)
    test_x = np.subtract(test_x, test_mean)
    #normalization
    train_x = train_x/np.std(train_x)
    test_x = test_x/np.std(test_x)
    return train_x, train_y, test_x, test_y

def showarray(a, nth, fmt = 'jpeg'):
    a = np.uint8(np.clip(a, 0, 1)*255)
    f = BytesIO()
    im = PIL.Image.fromarray(a)
    im.save('./l1_filter_visualization/filter_'+str(nth)+'.jpeg')

def visstd(a, s=0.1):
    return (a-a.mean())/max(a.std(), 1e-4)*s + 0.5


def convLayer(x, W, b, strides = 1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID')
    x = tf.nn.bias_add(x, b)
    return tf.maximum(0.01*x,x)#leaky relu

def maxPoolLayer(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding = 'VALID')

def conv_net(x, weights, biases):
    global N0, n_channel
    conv1 = convLayer(x, weights['wc1'], biases['bc1'], 2)
    conv1 = maxPoolLayer(conv1, k=2)

    conv2 = convLayer(conv1, weights['wc2'], biases['bc2'], 2)
    conv2 = maxPoolLayer(conv2, k=2)

    
    #convolution layer 1, pooling layer 1
    conv3 = convLayer(conv2, weights['wc3'], biases['bc3'])
    conv3 = maxPoolLayer(conv3, k=2)

    #convolution layer 2, pooling layer 2
    conv4 = convLayer(conv3, weights['wc4'], biases['bc4'])
    conv4 = maxPoolLayer(conv4, k=2)

    #convolution layer 3
    conv5 = convLayer(conv4, weights['wc5'], biases['bc5'])

    #fully connected layer, output layer
    fc1 = tf.reshape(conv3, [-1, 3*3*64])
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

def trainCNN():
    global n_classes, N0, n_channel, train_size, batch_size, rate, decay, schedule, training_itr, display_step, data_path, threshold, goodUntil

    inputs = tf.placeholder(tf.float32, [None, N0, N0 ,n_channel])
    y = tf.placeholder(tf.float32, [n_classes])
    
    eta = tf.placeholder(tf.float32)
    weights = {
        'wc1': tf.get_variable("WC1",shape=[5,5,3,32], initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode="FAN_IN")),
        'wc2': tf.get_variable("WC2",shape=[5,5,32,32], initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode="FAN_IN")),
        'wc3': tf.get_variable("WC3",shape=[5,5,32,32], initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode="FAN_IN")),
        'wc4': tf.get_variable("WC4",shape=[5,5,32,32], initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode="FAN_IN")),
        'wc5': tf.get_variable("WC5",shape=[3,3,32,64], initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode="FAN_IN")),
        'out': tf.get_variable("WOUT",shape=[3*3*64, n_classes],initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode="FAN_IN"))
    }
    biases = {
        'bc1': tf.Variable(tf.constant(0.005, shape=[32]),name="BC1"),
        'bc2': tf.Variable(tf.constant(0.005, shape=[32]),name="BC2"),
        'bc3': tf.Variable(tf.constant(0.005, shape=[32]),name="BC3"),
        'bc4': tf.Variable(tf.constant(0.005, shape=[32]), name="BC4"),
        'bc5': tf.Variable(tf.constant(0.005, shape=[64]), name="BC5"),
        'out': tf.Variable(tf.constant(0.005, shape=[n_classes]), name="BOUT")
    }
    pred = conv_net(inputs, weights, biases)

    #saving: preparing model
    predict_op = tf.argmax(pred,1)
    tf.get_collection("validation_nodes")
    tf.add_to_collection("validation_nodes", inputs)
    tf.add_to_collection("validation_nodes", predict_op)
    
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
    cost = tf.reduce_mean(tf.square(pred-y))
    regularizer = tf.nn.l2_loss(weights['wc1'])
    regularizer += tf.nn.l2_loss(weights['wc2'])
    regularizer += tf.nn.l2_loss(weights['wc3'])
    regularizer += tf.nn.l2_loss(weights['wc4'])
    regularizer += tf.nn.l2_loss(weights['wc5'])
    regularizer += tf.nn.l2_loss(weights['out'])
    regularizer /= 6
    cost_reg = tf.reduce_mean(cost + beta * regularizer)

    optimizer = tf.train.AdamOptimizer(learning_rate=eta).minimize(cost_reg)
    #correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    correct_pred = tf.square(pred-y);
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))#mean square error
    init = tf.global_variables_initializer()
    
    train_x, train_y, test_x, test_y = readDataFromPath(data_path)
    train_acc_list = []
    test_acc_list = []
    loss_list = []
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        step = 1
        good = 0
        while step < training_itr:
            choices = np.random.choice(train_size, batch_size, replace = False)
            batch_x = train_x[choices]
            batch_y = train_y[choices]
            if step % schedule == 0:
                rate = rate*decay
                print 'learning rate decreased to: '+ str(rate)
            sess.run(optimizer, feed_dict={inputs: batch_x, y: batch_y, eta: rate})
            if step % display_step == 0:
                loss, train_acc = sess.run([cost, accuracy], feed_dict={inputs: batch_x, y: batch_y})
                test_acc = sess.run(accuracy, feed_dict={inputs: test_x, y: test_y})
                print "Iteration: "+str(step) + ", Loss= "+"{:.6f}".format(loss) + ", train accurary= "+"{:.5f}".format(train_acc) + ", test accuracy= "+"{:.5f}".format(test_acc)
                train_acc_list.append(train_acc)
                test_acc_list.append(test_acc)
                loss_list.append(loss)
                if test_acc > threshold:
                    good = good + 1
                    if good >= goodUntil:
                        break
            step += 1
        print "Done!"
        #print "Test accuracy:"
        #class1 = np.where(test_y[:,0]==1)
        #class2 = np.where(test_y[:,1]==1)
        # class3 = np.where(test_y[:,2]==1)
        # class4 = np.where(test_y[:,3]==1)
        # class5 = np.where(test_y[:,4]==1)
#class6 = np.where(test_y[:,5]==1)
#class7 = np.where(test_y[:,6]==1)
#class8 = np.where(test_y[:,7]==1)
#class9 = np.where(test_y[:,8]==1)
#class10 = np.where(test_y[:,9]==1)
#acc_1 = sess.run(accuracy, feed_dict={inputs: test_x, y: test_y})
        #acc_2 = sess.run(accuracy, feed_dict={inputs: test_x[class2], y: test_y[class2]})
        # acc_3 = sess.run(accuracy, feed_dict={inputs: test_x[class3], y: test_y[class3]})
        # acc_4 = sess.run(accuracy, feed_dict={inputs: test_x[class4], y: test_y[class4]})
        # acc_5 = sess.run(accuracy, feed_dict={inputs: test_x[class5], y: test_y[class5]})
#acc_6 = sess.run(accuracy, feed_dict={inputs: test_x[class6], y: test_y[class6]})
#acc_7 = sess.run(accuracy, feed_dict={inputs: test_x[class7], y: test_y[class7]})
#acc_8 = sess.run(accuracy, feed_dict={inputs: test_x[class8], y: test_y[class8]})
#acc_9 = sess.run(accuracy, feed_dict={inputs: test_x[class9], y: test_y[class9]})
#acc_10 = sess.run(accuracy, feed_dict={inputs: test_x[class10], y: test_y[class10]})
        acc_ave = sess.run(accuracy, feed_dict={inputs: test_x, y: test_y})
        #print "Class 1: "+"{:.6f}".format(acc_1)
        #print "Class 2: "+"{:.6f}".format(acc_2)
        # print "Class 3: "+"{:.6f}".format(acc_3)
        # print "Class 4: "+"{:.6f}".format(acc_4)
        # print "Class 5: "+"{:.6f}".format(acc_5)
#print "Class 6: "+"{:.6f}".format(acc_6)
#print "Class 7: "+"{:.6f}".format(acc_7)
#        print "Class 8: "+"{:.6f}".format(acc_8)
#        print "Class 9: "+"{:.6f}".format(acc_9)
#        print "Class 10: "+"{:.6f}".format(acc_10)
        print "Test accuracy: "+"{:.6f}".format(acc_ave)
        #print "Average: "+"{:.6f}".format(acc_ave)

        #save model
        save_path = saver.save(sess, "my_model")
        #save weights, for plotting filters
        filters = [weights['wc1'].eval(), weights['wc2'].eval(), weights['wc3'].eval(), weights['wc4'].eval(), weights['wc5'].eval(), weights['out'].eval()]
        with open('filters','wb') as fp:
            pickle.dump(filters, fp)
        BIAS = [biases['bc1'].eval(),biases['bc2'].eval(),biases['bc3'].eval(),
                biases['bc4'].eval(),biases['bc5'].eval(),biases['bout'].eval()]
        with open('biases','wb') as fp:
            pickle.dump(BIAS, fp)
        sess.close()
        #save accuracy lists and loss list, for plotting
        with open('train_acc','wb') as fp:
            pickle.dump(train_acc_list, fp)
        with open('test_acc','wb') as fp:
            pickle.dump(test_acc_list, fp)
        with open('cost','wb') as fp:
            pickle.dump(loss_list, fp)

def plotAcc():
    print "Plotting accuracy figures..."
    with open ('train_acc', 'rb') as fp:
        train_acc_list = pickle.load(fp)
    with open ('test_acc', 'rb') as fp:
        test_acc_list = pickle.load(fp)
    with open ('cost', 'rb') as fp:
        loss_list = pickle.load(fp)
    train_acc = train_acc_list[::1]
    test_acc = test_acc_list[::1]
    loss = loss_list[::1]

    t = list(range(len(loss)))
    fig, ax1 = plt.subplots()
    ax1.plot(t, loss,'b')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Cost', color='b')
    ax1.tick_params('y', colors='b')

    ax2 = ax1.twinx()
    ax2.plot(t, train_acc,'r')
    ax2.plot(t, test_acc,'g')
    ax2.set_ylabel('Accuracy', color='k')
    ax2.tick_params('y', colors='k')
    fig.tight_layout()
    plt.legend(['Train Accurary', 'Test Accuracy'], loc = 2)
    fig.savefig('accuracy.pdf')
    plt.close(fig)

def plotFilter1(nth):
    global N0, n_channel
    with open ('filters', 'rb') as fp:
        filters = pickle.load(fp)
    wc1 = filters[0]

    inputs = tf.placeholder(tf.float32, [1, N0, N0 ,n_channel])
    bias = np.zeros(32)
    conv1 = convLayer(inputs, wc1, bias)
    t_score = tf.reduce_mean(conv1[:,:,:,nth]) # defining the optimization objective
    t_grad = tf.gradients(t_score, inputs)[0] # behold the power of automatic differentiation!
    img_noise = np.random.uniform(size=(1,32,32,3)) + 0.5
    img = img_noise.copy()
    iter_n = 20
    step = 1.0
    sess = tf.Session()
    for i in range(iter_n):
        score, g = sess.run([t_score, t_grad], feed_dict={inputs: img})
        # normalizing the gradient, so the same step size should work
        g /= g.std()+1e-8# for different layers and networks
        img += g*step
    img = img.reshape(32,32,3)
    clear_output()
    showarray(visstd(img), nth)

def main():
    trainCNN()
    plotAcc()
    print "Exporting filters..."
    for i in range(32):
        plotFilter1(i)

if __name__ == '__main__':
    main()
