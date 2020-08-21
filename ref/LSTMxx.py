"""
Minimal character-level LSTM model. Written by Ngoc Quan Pham
Code structure borrowed from the Vanilla RNN model from Andreij Karparthy @karparthy.
BSD License
"""
import numpy as np
from random import uniform
import sys


# Since numpy doesn't have a function for sigmoid
# We implement it manually here
def sigmoid(x):
  return 1 / (1 + np.exp(-x))


# The derivative of the sigmoid function
def dsigmoid(y):
    return y * (1 - y)


# The derivative of the tanh function
def dtanh(x):
    return 1 - x*x


# The numerically stable softmax implementation
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


# data I/O
data = open('data/input.txt', 'r').read() # should be simple plain text file
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print('data has %d characters, %d unique.' % (data_size, vocab_size))
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }
std = 0.1

#option = sys.argv[1]
#option = 'train'
option = 'gradcheck'

# hyperparameters
emb_size = 8
hidden_size = 128  # size of hidden layer of neurons
seq_length = 32  # number of steps to unroll the RNN for
learning_rate = 10e-2
max_updates = 500000

concat_size = emb_size + hidden_size

# model parameters
# char embedding parameters
Wex = np.random.randn(emb_size, vocab_size)*std # embedding layer

# LSTM parameters
Wf = np.random.randn(hidden_size, concat_size) * std # forget gate
Wi = np.random.randn(hidden_size, concat_size) * std # input gate
Wo = np.random.randn(hidden_size, concat_size) * std # output gate
Wc = np.random.randn(hidden_size, concat_size) * std # c term

bf = np.zeros((hidden_size, 1)) # forget bias
bi = np.zeros((hidden_size, 1)) # input bias
bo = np.zeros((hidden_size, 1)) # output bias
bc = np.zeros((hidden_size, 1)) # memory bias

# Output layer parameters
Why = np.random.randn(vocab_size, hidden_size)*0.01 # hidden to output
by = np.zeros((vocab_size, 1)) # output bias


def forward(inputs, targets, memory):
    """
    inputs,targets are both list of integers.
    hprev is Hx1 array of initial hidden state
    returns the loss, gradients on model parameters, and last hidden state
    """

    # The LSTM is different than the simple RNN that it has two memory cells
    # so here you need two different hidden layers
    hprev, cprev = memory

    # Here you should allocate some variables to store the activations during forward
    # One of them here is to store the hiddens and the cells
    xs, cs, hs, os, ps, ys = {}, {}, {}, {}, {}, {}
    wes, zs, f_gate, i_gate, c_hat, o_gate= {}, {}, {}, {}, {}, {}

    hs[-1] = np.copy(hprev)
    cs[-1] = np.copy(cprev)

    loss = 0
    # forward pass
    for t in range(len(inputs)):
        xs[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
        xs[t][inputs[t]] = 1

        # convert word indices to word embeddings
        wes[t] = np.dot(Wex, xs[t])

        # LSTM cell operation
        # first concatenate the input and h
        # This step is irregular (to save the amount of matrix multiplication we have to do)
        # I will refer to this vector as [h X]
        zs[t] = np.row_stack((hs[t-1], wes[t]))

        # YOUR IMPLEMENTATION should begin from here

        # compute the forget gate
        f_gate[t] = sigmoid(np.dot(Wf, zs[t]) + bf)

        # compute the input gate
        i_gate[t] = sigmoid(np.dot(Wi, zs[t]) + bi)

        # compute the candidate memory
        c_hat[t] = np.tanh(np.dot(Wc, zs[t]) + bc)

        # new memory: applying forget gate on the previous memory
        # and then adding the input gate on the candidate memory
        cs[t] = f_gate[t] * cs[t-1] + i_gate[t] * c_hat[t]

        # output gate
        o_gate[t] = sigmoid(np.dot(Wo, zs[t])+bo)

        # new hidden state for the LSTM
        hs[t] = o_gate[t] * np.tanh(cs[t])

        # DONE LSTM
        # output layer - softmax and cross-entropy loss
        # unnormalized log probabilities for next chars

        os[t] = np.dot(Why, hs[t]) + by

        # softmax for probabilities for next chars
        ps[t] = softmax(os[t])

        # cross-entropy loss
        # cross entropy loss at time t:
        # create an one hot vector for the label y
        ys[t] = np.zeros((vocab_size, 1))
        ys[t][targets[t]] = 1
        # and then cross-entropy (see the elman-rnn file for the hint)
        loss += -np.log(ps[t][targets[t],0]) #same function as the hint but using numpy array indexing
    # define your activations
    activations = (xs, cs, hs, ps, ys, zs, f_gate, i_gate, c_hat, o_gate)
    memory = (hs[len(inputs)-1], cs[len(inputs)-1])
    return loss, activations, memory


def backward(activations, clipping=True):

    # backward pass: compute gradients going backwards
    # Here we allocate memory for the gradients
    dWex, dWhy = np.zeros_like(Wex), np.zeros_like(Why)
    dby = np.zeros_like(by)
    dWf, dWi, dWc, dWo = np.zeros_like(Wf), np.zeros_like(Wi),np.zeros_like(Wc), np.zeros_like(Wo)
    dbf, dbi, dbc, dbo = np.zeros_like(bf), np.zeros_like(bi),np.zeros_like(bc), np.zeros_like(bo)

    xs, cs, hs, ps, ys, zs, f_gate, i_gate, c_hat, o_gate = activations
    # similar to the hidden states in the vanilla RNN
    # We need to initialize the gradients for these variables
    dhnext = np.zeros_like(hs[0])
    dcnext = np.zeros_like(cs[0])

    # back propagation through time starts here
    for t in reversed(range(len(inputs))):

        # IMPLEMENT YOUR BACKPROP HERE
        # refer to the file elman_rnn.py for more details
        dos = ps[t] - ys[t]

        dWhy += np.dot(dos, hs[t].T)
        dby += dos

        dh = np.dot(Why.T, dos) + dhnext
        dc = dh * o_gate[t] * dtanh(np.tanh(cs[t])) + dcnext
        
        delta_c = o_gate[t] * dtanh(np.tanh(cs[t]))
        dhpre = np.dot(Wo[:,:hidden_size].T, dsigmoid(o_gate[t])) * np.tanh(cs[t])
        dhpre += np.dot(Wf[:,:hidden_size].T, (delta_c * dsigmoid(f_gate[t])) * cs[t-1])
        dhpre += np.dot(Wc[:,:hidden_size].T, (delta_c * i_gate[t] * dtanh(c_hat[t])))
        dhpre += np.dot(Wi[:,:hidden_size].T, (delta_c * c_hat[t] * dsigmoid(i_gate[t])))
        dhnext = dhpre * dh
        dcnext = dc * f_gate[t]

        df_gate = dc * cs[t-1] * dsigmoid(f_gate[t])
        di_gate = dc * c_hat[t] * dsigmoid(i_gate[t])
        dc_hat = dc * i_gate[t] * dtanh(c_hat[t])
        do_gate = dh * np.tanh(cs[t]) * dsigmoid(o_gate[t])

        dWf += np.dot(df_gate, zs[t].T)
        dWi += np.dot(di_gate, zs[t].T)
        dWc += np.dot(dc_hat, zs[t].T)
        dWo += np.dot(do_gate, zs[t].T)

        dbf += df_gate
        dbi += di_gate
        dbc += dc_hat
        dbo += do_gate

        dzs = np.dot(Wf.T, df_gate)
        dzs += np.dot(Wi.T, di_gate)
        dzs += np.dot(Wc.T, dc_hat)
        dzs += np.dot(Wo.T, do_gate)
        
        dwes = dzs[hidden_size:, 0].reshape((emb_size,1))
        dWex += np.dot(dwes, xs[t].T)

    if clipping:
        # clip to mitigate exploding gradients
        for dparam in [dWex, dWf, dWi, dWo, dWc, dbf, dbi, dbo, dbc, dWhy, dby]:
            np.clip(dparam, -5, 5, out=dparam)

    gradients = (dWex, dWf, dWi, dWo, dWc, dbf, dbi, dbo, dbc, dWhy, dby)

    return gradients


def sample(memory, seed_ix, n):
    """
    sample a sequence of integers from the model
    h is memory state, seed_ix is seed letter for first time step
    """
    h, c = memory
    x = np.zeros((vocab_size, 1))
    x[seed_ix] = 1
    generated_chars = []
    for t in range(n):
        # IMPLEMENT THE FORWARD FUNCTION ONE MORE TIME HERE
        # BUT YOU DON"T NEED TO STORE THE ACTIVATIONS
        we = np.dot(Wex, x)
        z = np.row_stack((h, we))
        f = sigmoid(np.dot(Wf, z) + bf)
        i = sigmoid(np.dot(Wi, z) + bi)
        ch = np.tanh(np.dot(Wc, z) + bc)
        c = f * c + i * ch
        o = sigmoid(np.dot(Wo, z) + bo)
        h = o * np.tanh(c)
        out = np.dot(Why, h) + by
        p = softmax(out)
        ix = np.random.choice(range(vocab_size), p=p.ravel()) #using numpy random choice
        x = np.zeros((vocab_size, 1))
        x[ix] = 1
        generated_chars.append(ix)
    return generated_chars

if option == 'train':

    n, p = 0, 0
    n_updates = 0

    # momentum variables for Adagrad
    mWex, mWhy = np.zeros_like(Wex), np.zeros_like(Why)
    mby = np.zeros_like(by) 

    mWf, mWi, mWo, mWc = np.zeros_like(Wf), np.zeros_like(Wi), np.zeros_like(Wo), np.zeros_like(Wc)
    mbf, mbi, mbo, mbc = np.zeros_like(bf), np.zeros_like(bi), np.zeros_like(bo), np.zeros_like(bc)

    smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0
    
    while True:
        # prepare inputs (we're sweeping from left to right in steps seq_length long)
        if p+seq_length+1 >= len(data) or n == 0:
            hprev = np.zeros((hidden_size,1)) # reset RNN memory
            cprev = np.zeros((hidden_size,1))
            p = 0 # go from start of data
        inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
        targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

        # sample from the model now and then
        if n % 100 == 0:
            sample_ix = sample((hprev, cprev), inputs[0], 200)
            txt = ''.join(ix_to_char[ix] for ix in sample_ix)
            print ('----\n %s \n----' % (txt, ))

        # forward seq_length characters through the net and fetch gradient
        loss, activations, memory = forward(inputs, targets, (hprev, cprev))
        gradients = backward(activations)

        hprev, cprev = memory
        dWex, dWf, dWi, dWo, dWc, dbf, dbi, dbo, dbc, dWhy, dby = gradients
        smooth_loss = smooth_loss * 0.999 + loss * 0.001
        if n % 100 == 0: print ('iter %d, loss: %f' % (n, smooth_loss)) # print progress

        # perform parameter update with Adagrad
        for param, dparam, mem in zip([Wf, Wi, Wo, Wc, bf, bi, bo, bc, Wex, Why, by],
                                    [dWf, dWi, dWo, dWc, dbf, dbi, dbo, dbc, dWex, dWhy, dby],
                                    [mWf, mWi, mWo, mWc, mbf, mbi, mbo, mbc, mWex, mWhy, mby]):
            mem += dparam * dparam
            param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

        p += seq_length # move data pointer
        n += 1 # iteration counter
        n_updates += 1
        if n_updates >= max_updates:
            break

elif option == 'gradcheck':

    p = 0
    inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
    targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

    delta = 0.001

    hprev = np.zeros((hidden_size, 1))
    cprev = np.zeros((hidden_size, 1))

    memory = (hprev, cprev)

    loss, activations, _ = forward(inputs, targets, memory)
    gradients = backward(activations, clipping=False)
    dWex, dWf, dWi, dWo, dWc, dbf, dbi, dbo, dbc, dWhy, dby = gradients

    for weight, grad, name in zip([Wf, Wi, Wo, Wc, bf, bi, bo, bc, Wex, Why, by], 
                                   [dWf, dWi, dWo, dWc, dbf, dbi, dbo, dbc, dWex    , dWhy, dby],
                                   ['Wf', 'Wi', 'Wo', 'Wc', 'bf', 'bi', 'bo', 'bc', 'Wex', 'Why', 'by']):

        str_ = ("Dimensions dont match between weight and gradient %s and %s." % (weight.shape, grad.shape))
        assert(weight.shape == grad.shape), str_

        print(name)
        for i in range(weight.size):
      
            # evaluate cost at [x + delta] and [x - delta]
            w = weight.flat[i]
            weight.flat[i] = w + delta
            loss_positive, _, _ = forward(inputs, targets, memory)
            weight.flat[i] = w - delta
            loss_negative, _, _ = forward(inputs, targets, memory)
            weight.flat[i] = w  # reset old value for this parameter

            grad_analytic = grad.flat[i]
            grad_numerical = (loss_positive - loss_negative) / ( 2 * delta )

            # compare the relative error between analytical and numerical gradients
            rel_error = abs(grad_analytic - grad_numerical) / abs(grad_numerical + grad_analytic)

            if rel_error > 0.01:
                print ('WARNING %f, %f => %e ' % (grad_numerical, grad_analytic, rel_error))