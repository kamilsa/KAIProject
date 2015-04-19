from pybrain.tools.shortcuts import buildNetwork
from FeatureRetrieval.ImageProcessing import get_train_set

__author__ = 'kamil'


import numpy as np
import pickle
import scipy
from pybrain.datasets import SupervisedDataSet, SequentialDataSet
from pybrain.structure import RecurrentNetwork, IdentityConnection, FeedForwardNetwork, LSTMLayer, BiasUnit
from pybrain.structure.connections.full import FullConnection
from pybrain.structure.modules.linearlayer import LinearLayer
from pybrain.structure.modules.sigmoidlayer import SigmoidLayer
from pybrain.supervised import BackpropTrainer, RPropMinusTrainer

import root

# def getRecNetFromFile(path):

def exportANN(net, fileName = root.path()+"/res/LSTMNN"):
    fileObject = open(fileName, 'w')
    pickle.dump(net, fileObject)
    fileObject.close()

def importANN(fileName = root.path()+"/res/LSTMNN"):
    fileObject = open(fileName, 'r')
    net = pickle.load(fileObject)
    fileObject.close()
    return net

def create1OrderDataSet():
    lab_images = get_train_set(instance=False, number_of_instances=10)
    ds = SupervisedDataSet(100, 1)
    for i in range(len(lab_images)):
        data = np.zeros((100))
        for j in range(100):
            data[j] = lab_images[i][0][j]
        ds.addSample(data, lab_images[i][1])
        print "creating dataset, iteration:",i,"of",len(lab_images)
    ds.saveToFile(root.path() + '/res/dataset1')
    return ds

def create2OrderDataSet():
    lab_images = get_train_set(instance=True)
    ds = SupervisedDataSet(5150, 1)
    for i in range(len(lab_images)):
        data = np.zeros((5150))
        for j in range(100):
            data[j] = lab_images[i][0][j]
        count = 100
        for x1 in range(100):
            for x2 in range(x1, 100):
                # print count
                data[count] = lab_images[i][0][x1]*lab_images[i][0][x2]
                count += 1
        ds.addSample(data, lab_images[i][1])
        print "creating dataset, iteration:",i,"of",len(lab_images)
    ds.saveToFile(root.path() + '/res/dataset2')
    return ds

def create3OrderDataSet():
    lab_images = get_train_set(instance=True)
    ds = SupervisedDataSet(176850, 1)
    for i in range(len(lab_images)):
        data = np.zeros((176850))
        for j in range(100):
            data[j] = lab_images[i][0][j]
        count = 100
        for x1 in range(100):
            for x2 in range(x1, 100):
                # print count
                data[count] = lab_images[i][0][x1]*lab_images[i][0][x2]
                count += 1
        for x1 in range(100):
            for x2 in range(x1, 100):
                for x3 in range(x2, 100):
                    data[count] = lab_images[i][0][x1]*lab_images[i][0][x3]
                    count += 1
        ds.addSample(data, lab_images[i][1])
        print "creating dataset, iteration:",i,"of",len(lab_images)
    ds.saveToFile(root.path() + '/res/dataset3')
    return ds

def load1OrderDataSet():
    ds = SupervisedDataSet.loadFromFile(root.path() + '/res/dataset1')
    return ds

def load2OrderDataSet():
    ds = SupervisedDataSet.loadFromFile(root.path() + '/res/dataset2')
    return ds

def load3OrderDataSet():
    ds = SupervisedDataSet.loadFromFile(root.path() + '/res/dataset3')
    return ds

def trainedLSTMNN():
    """
    n = RecurrentNetwork()

    inp = LinearLayer(100, name = 'input')
    hid = LSTMLayer(30, name='hidden')
    out = LinearLayer(1, name='output')

    #add modules
    n.addOutputModule(out)
    n.addInputModule(inp)
    n.addModule(hid)

    #add connections
    n.addConnection(FullConnection(inp, hid))
    n.addConnection(FullConnection(hid, out))

    n.addRecurrentConnection(FullConnection(hid, hid))
    n.sortModules()
    """
    n = buildNetwork(100, 50, 1, hiddenclass = LSTMLayer, outputbias=False, recurrent = True)

    print "Network created"
    d = load1OrderDataSet()
    print "Data loaded"
    t = BackpropTrainer(n, d, learningrate=0.001, momentum=0.75)
    # FIXME: I'm not sure the recurrent ANN is going to converge
    # so just training for fixed number of epochs
    print "Learning started"
    count = 0
    while True:
        globErr = t.train()
        print "iteration #", count," error = ", globErr
        if globErr < 0.1:
            break
        count = count + 1
        # if (count == 60):
        #     break

    # for i in range(100):
    #     print t.train()


    exportANN(n)

    return n


def buildSimpleLSTMNetwork(peepholes = False):
    N = RecurrentNetwork('simpleLstmNet')
    i = LinearLayer(100, name = 'i')
    h = LSTMLayer(10, peepholes = peepholes, name = 'lstm')
    o = LinearLayer(1, name = 'o')
    b = BiasUnit('bias')
    N.addModule(b)
    N.addOutputModule(o)
    N.addInputModule(i)
    N.addModule(h)
    N.addConnection(FullConnection(i, h, name = 'f1'))
    N.addConnection(FullConnection(b, h, name = 'f2'))
    N.addRecurrentConnection(FullConnection(h, h, name = 'r1'))
    N.addConnection(FullConnection(h, o, name = 'r1'))
    N.sortModules()
    return N

def trainedLSTMNN2():
    """
    n = RecurrentNetwork()

    inp = LinearLayer(100, name = 'input')
    hid = LSTMLayer(30, name='hidden')
    out = LinearLayer(1, name='output')

    #add modules
    n.addOutputModule(out)
    n.addInputModule(inp)
    n.addModule(hid)

    #add connections
    n.addConnection(FullConnection(inp, hid))
    n.addConnection(FullConnection(hid, out))

    n.addRecurrentConnection(FullConnection(hid, hid))
    n.sortModules()
    """
    n = buildSimpleLSTMNetwork()

    print "Network created"
    d = load1OrderDataSet()
    print "Data loaded"
    t = RPropMinusTrainer(n, dataset=d, verbose=True)
    t.trainUntilConvergence()

    exportANN(n)

    return n

def trained2ONN():
    n = FeedForwardNetwork()

    inp = LinearLayer(5150, name = 'input')
    hid = LinearLayer(3, name='hidden')
    out = LinearLayer(1, name='output')

    #add modules
    n.addOutputModule(out)
    n.addInputModule(inp)
    n.addModule(hid)

    #add connections
    n.addConnection(FullConnection(inp, hid, inSliceTo = 100, outSliceTo = 1))
    n.addConnection(FullConnection(inp, hid, inSliceFrom = 100, outSliceFrom = 1))
    n.addConnection(FullConnection(hid, out))

    n.sortModules()
    print "Network created"
    d = load2OrderDataSet()
    print "Data loaded"
    t = BackpropTrainer(n, d, learningrate=0.01, momentum=0.75)
    # FIXME: I'm not sure the recurrent ANN is going to converge
    # so just training for fixed number of epochs
    print "Learning started"
    count = 0
    while True:
        globErr = t.train()
        print "iteration #", count," error = ", globErr
        if globErr < 0.01:
            break
        count = count + 1
        # if (count == 100):
        #     break

    # for i in range(100):
    #     print t.train()


    exportANN(n)

    return n

def trained3ONN():
    n = FeedForwardNetwork()

    inp = LinearLayer(176850, name = 'input')
    hid = LinearLayer(3, name='hidden')
    out = LinearLayer(1, name='output')

    #add modules
    n.addOutputModule(out)
    n.addInputModule(inp)
    n.addModule(hid)

    #add connections
    n.addConnection(FullConnection(inp, hid, inSliceTo = 100, outSliceTo = 1))
    n.addConnection(FullConnection(inp, hid, inSliceFrom = 100, inSliceTo = 5150, outSliceFrom = 1, outSliceTo = 2))
    n.addConnection(FullConnection(inp, hid, inSliceFrom = 5150, outSliceFrom = 2))
    n.addConnection(FullConnection(hid, out))

    n.sortModules()
    print "Network created"
    d = load3OrderDataSet()
    print "Data loaded"
    t = BackpropTrainer(n, d, learningrate=0.001, momentum=0.75)
    # FIXME: I'm not sure the recurrent ANN is going to converge
    # so just training for fixed number of epochs
    print "Learning started"
    count = 0
    while True:
        globErr = t.train()
        print "iteration #", count," error = ", globErr
        if globErr < 0.01:
            break
        count = count + 1
        # if (count == 100):
        #     break

    # for i in range(100):
    #     print t.train()


    exportANN(n)

    return n

def draw_connections(net):

    for mod in net.modules:
        print "Module:", mod.name
        if mod.paramdim > 0:
            print "--parameters:", mod.params
        for conn in net.connections[mod]:
            print "-connection to", conn.outmod.name
            if conn.paramdim > 0:
                print "- parameters", conn.params
            if hasattr(net, "recurrentConns"):
                print "Recurrent connections"
                for conn in net.recurrentConns:
                   print "-", conn.inmod.name, " to", conn.outmod.name
                   if conn.paramdim > 0:
                      print "- parameters", conn.params

def initial_with_zeros(net):
    zeros = ([10.0]*len(net.params))
    net._setParameters(zeros)

def run():
    # trained2ONN()
    trainedLSTMNN2()
    n = importANN()
    testSet = get_train_set(instance=False)
    # print testSet[1][0]
    # print "len", len(testSet[0][1])
    for i in range(5000):
        number = i
        print "expected",testSet[number][1],
        print "calculated", n.activate(testSet[number][0])


# run()
count1 = 0
count2 = 0
count3 = 0
for i in range(100):
    count1 += 1
    for j in range(i, 100):
        count2 += 1
        for k in range(j, 100):
            count3 += 1

print count1, count2, count3, count1+count2+count3
run()
# create1OrderDataSet()