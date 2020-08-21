#!/usr/bin/env python3

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
option = "train"
#option = "gradcheck"
#option = sys.argv[1]

# hyperparameters
emb_size = 8 # size of x
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
bo = np.zeros((hidden_size, 1)) # output gate bias
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
    hs, cs, xs, wes, zs, ps, ys, o_gate, f_gate, c_hat, i_gate = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}

    hs[-1] = np.copy(hprev) # the first memory (before training)
    cs[-1] = np.copy(cprev)

    loss = 0
    # forward pass
    for t in range(len(inputs)):
        xs[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
        xs[t][inputs[t]] = 1

        # convert word indices to word embeddings
        wes[t] = np.dot(Wex, xs[t])

        # LSTM cell operation
        # first concatenate the input and h as one single input vector
        # This step is irregular (to save the amount of matrix multiplication we have to do)
        # I will refer to this vector as [h X]
        zs[t] = np.row_stack((hs[t-1], wes[t]))

        # YOUR IMPLEMENTATION should begin from here

        # compute the forget gate
        f_gate[t] = sigmoid (np.dot(Wf,  zs[t]) + bf)

        # compute the input gate
        i_gate[t] = sigmoid (np.dot(Wi,  zs[t]) + bi)

        # compute the candidate memory
        c_hat[t] = np.tanh (np.dot(Wc,  zs[t]) + bc)

        # new memory: applying forget gate on the previous memory
        # and then adding the input gate on the candidate memory
        cs[t] = f_gate[t] * cs[t-1] + i_gate[t] * c_hat[t]

        # output gate
        o_gate[t] = sigmoid (np.dot(Wo,  zs[t]) + bo)

        # new hidden state for the LSTM
        hs[t] = o_gate[t] * np.tanh(cs[t])

        # DONE LSTM
        # output layer - softmax and cross-entropy loss
        # unnormalized log probabilities for next chars

        os = np.dot(Why, hs[t]) + by

        # softmax for probabilities for next chars
        ps[t] = softmax(os)

        # cross-entropy loss
        # cross entropy loss at time t:
        ys[t] = np.zeros((vocab_size, 1))
        # create an one hot vector for the label y
        ys[t][targets[t]] = 1
        # and then cross-entropy (see the elman-rnn file for the hint)
        loss_t = np.sum(-np.log(ps[t]) * ys[t])

        loss += loss_t
    # define your activations
    activations = (hs, cs, xs, wes, zs, ps, ys, o_gate, f_gate, c_hat, i_gate)
    next_memory = (hs[len(inputs)-1], cs[len(inputs)-1])
    return loss, activations, next_memory

def backward(activations, clipping=True):

    # backward pass: compute gradients going backwards
    # Here we allocate memory for the gradients
    Delta_Wex, Delta_Why = np.zeros_like(Wex), np.zeros_like(Why)
    Delta_by = np.zeros_like(by)
    Delta_Wf, Delta_Wi, Delta_Wc, Delta_Wo = np.zeros_like(Wf), np.zeros_like(Wi),np.zeros_like(Wc), np.zeros_like(Wo)
    Delta_bf, Delta_bi, Delta_bc, Delta_bo = np.zeros_like(bf), np.zeros_like(bi),np.zeros_like(bc), np.zeros_like(bo)

    (hs, cs, xs, wes, zs, ps, ys, o_gate, f_gate, c_hat, i_gate) = activations
    # similar to the hidden states in the vanilla RNN
    # We need to initialize the gradients for these variables
    dE_dhs = np.zeros_like(hs[0])
    dE_dcs = np.zeros_like(cs[0])
    # back propagation through time starts here
    for t in reversed(range(len(inputs))):
        # IMPLEMENT YOUR BACKPROP HERE
        # refer to the file elman_rnn.py for more details
        dE_do = ps[t] - ys[t]
        Delta_Why += np.dot(dE_do, hs[t].T)
        Delta_by += dE_do
        # because h is connected to both o and the next h, we sum the gradients up

        # a memory is added to itself
        dE_dhs = np.dot(Why.T, dE_do)+ dE_dhs
        ##
        dE_do_gate = dE_dhs * np.tanh(cs[t])
        dE_do_gate_pre = dE_do_gate * dsigmoid(o_gate[t])
        #dE_do_gate_pre = dE_dhs * np.tanh(cs[t]) * dsigmoid(o_gate[t])
        ################
        #dE_dz = np.dot(Wo.T, dE_do_gate_pre)
        Delta_Wo += np.dot(dE_do_gate_pre, zs[t].T)
        Delta_bo += dE_do_gate_pre
        ############
        
        ##############
        dhs_dcs = o_gate[t] * dtanh(np.tanh(cs[t]))


        # a memory is added to itself
        dE_dcs = dE_dhs * dhs_dcs + dE_dcs# + dE_dcs_next * f_gate_next
        #dE_dcs = dE_dhs * o_gate[t] * dtanh(np.tanh(cs[t])) + dE_dcs
    #    dE_dcs_next = dE_dcs
        ##
        dE_dc_hat = dE_dcs * i_gate[t]
        dE_dc_hat_pre = dE_dc_hat * dtanh(c_hat[t])
        #dE_dc_hat_pre = dE_dcs * i_gate[t] * dtanh(c_hat[t])
        ##
        dE_di_gate = dE_dcs * c_hat[t]
        dE_di_gate_pre = dsigmoid(i_gate[t]) * dE_di_gate
        #dE_di_gate_pre = dE_dcs * c_hat[t] * dsigmoid(i_gate[t])
        ##

        dE_df_gate = dE_dcs * cs[t-1]
        dE_df_gate_pre = dE_df_gate * dsigmoid(f_gate[t])
        #dE_df_gate_pre = dE_dcs * cs[t - 1] * dsigmoid(f_gate[t])
        Delta_Wc += np.dot(dE_dc_hat_pre, zs[t].T)
        Delta_bc += dE_dc_hat_pre
        Delta_Wi += np.dot(dE_di_gate_pre, zs[t].T)
        Delta_bi += dE_di_gate_pre
        Delta_Wf += np.dot(dE_df_gate_pre, zs[t].T)
        Delta_bf += dE_df_gate_pre
        # if t == len(inputs) - 1:
        #     f_gate_next = 0
        # else:
        #     f_gate_next = f_gate[t + 1]
        dE_dcs = dE_dcs * f_gate[t]

        dE_dz = np.dot(Wc.T, dE_dc_hat_pre) + np.dot(Wi.T, dE_di_gate_pre) + np.dot(Wf.T, dE_df_gate_pre) + np.dot(Wo.T, dE_do_gate_pre)
        ##
        dE_dhs = dE_dz[:hidden_size]
        dE_dwes = dE_dz[hidden_size:]
        Delta_Wex += np.dot(dE_dwes, xs[t].T)
    if clipping:
        # clip to mitigate exploding gradients
        for dparam in [Delta_Wex, Delta_Wf, Delta_Wi, Delta_Wo, Delta_Wc, Delta_bf, Delta_bi, Delta_bo, Delta_bc, Delta_Why, Delta_by]:
            np.clip(dparam, -5, 5, out=dparam)

    gradients = (Delta_Wex, Delta_Wf, Delta_Wi, Delta_Wo, Delta_Wc, Delta_bf, Delta_bi, Delta_bo, Delta_bc, Delta_Why, Delta_by)

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
        wes = np.dot(Wex, x)

        # LSTM cell operation
        # first concatenate the input and h as one single input vector
        # This step is irregular (to save the amount of matrix multiplication we have to do)
        # I will refer to this vector as [h X]
        zs = np.row_stack((h, wes))

        # YOUR IMPLEMENTATION should begin from here
        # compute the input gate
        i_gate = sigmoid (np.dot(Wi,  zs) + bi)
        # compute the forget gate
        f_gate = sigmoid (np.dot(Wf,  zs) + bf)
        # compute the candidate memory
        c_hat = np.tanh (np.dot(Wc,  zs) + bc)

        # new memory: applying forget gate on the previous memory
        # and then adding the input gate on the candidate memory
        c = c * f_gate + c_hat * i_gate

        # output gate
        o_gate = sigmoid (np.dot(Wo,  zs) + bo)

        # new hidden state for the LSTM
        h = o_gate * np.tanh(c)

        # DONE LSTM
        # output layer - softmax and cross-entropy loss
        # unnormalized log probabilities for next chars

        os = np.dot(Why, h) + by

        # softmax for probabilities for next chars
        ps = softmax(os)

        ix = np.random.multinomial(1, ps.ravel())
        x = np.zeros((vocab_size, 1))

        for j in range(len(ix)):
            if ix[j] == 1:
                index = j
        x[index] = 1
        generated_chars.append(index)

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
            p = 0 # ??go from start of data
        inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
        targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

        # sample from the model now and then
        if n % 100 == 0:
            sample_ix = sample((hprev, cprev), inputs[0], 2000)
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