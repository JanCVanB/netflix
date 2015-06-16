# this is a modified version of the example script that comes with cudamat
from numpy import load, mean

def test_gnumpy(num_epochs):
    import gnumpy as gpu
    # load data. <dat> is 2 dimensional: 60000 X 784
    dat = gpu.garray(load('mnist_cudaTest').T/255.)
    # training parameters
    epsilon = 0.1
    momentum = 0.9
    batch_size = 128
    num_batches = dat.shape[0]/batch_size
    # model parameters
    num_vis = dat.shape[1]
    num_hid = 4096
    # initialize weights
    w_vh = 0.1 * gpu.randn(num_vis, num_hid)
    w_v = gpu.zeros(num_vis)
    w_h = -4. * gpu.ones(num_hid)
    # initialize weight updates
    wu_vh = gpu.zeros((num_vis, num_hid))
    wu_v = gpu.zeros(num_vis)
    wu_h = gpu.zeros(num_hid)
    for epoch in range(num_epochs):
        err = []
        for batch in range(num_batches):
            # positive phase
            v1 = dat[batch*batch_size : (batch + 1)*batch_size]
            h1 = (gpu.dot(v1, w_vh) + w_h).logistic()
            # sample hiddens
            hSampled = h1.rand() < h1
            # negative phase
            v2 = (gpu.dot(hSampled, w_vh.T) + w_v).logistic()
            h2 = (gpu.dot(v2, w_vh) + w_h).logistic()
            # update weights
            wu_vh = wu_vh * momentum + gpu.dot(v1.T, h1) - gpu.dot(v2.T, h2)
            wu_v = wu_v * momentum + v1.sum(0) - v2.sum(0)
            wu_h = wu_h * momentum + h1.sum(0) - h2.sum(0)

            w_vh += wu_vh * (epsilon/batch_size)
            w_v += wu_v * (epsilon/batch_size)
            w_h += wu_h * (epsilon/batch_size)
            # calculate reconstruction error
            err.append((v2-v1).euclid_norm()**2/(num_vis*batch_size))
    # print "Mean squared error: " + str(mean(err))
    return w_vh, w_v, w_h
