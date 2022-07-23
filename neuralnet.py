import numpy as np

#useful functions 0: f, 1: f_prime, 2: name in string

#activation
relu = (lambda x: x * (x>0), lambda x: 1 * (x>0))
relu[0].__name__ = 'relu'


linear = (lambda x: x, lambda x: 1)
linear[0].__name__ = 'linear'

__dsig_helper = lambda x: x*(1-x)
sigmoid = (lambda x: 1/(1+np.exp(-x)) , lambda x: __dsig_helper(sigmoid[0](x)))
sigmoid[0].__name__ = 'sigmoid'

#loss
mse = (lambda x, y: np.square(x-y).mean(), lambda x, y: 2*(x-y)) #mean square error
mse[0].__name__ = 'mean_square_error'

bce = (lambda x, y: -(y*np.log(x+1e-8)+(1-y)*np.log(1-x+1e-8)).mean(), lambda x, y: -y/(x+1e-8) + (1-y)/(1-x+1e-8)) #binary cross entropy
bce[0].__name__ = 'binary_cross_entropy'

#metric
bmap = lambda x: (x+0.5)//1 #binaary categorical data mapper with threshold 0.5
bacc = lambda x, y: np.mean(bmap(x) == y) #binary accuracy
bacc.__name__ = 'binary_accuracy'


class NeuralNet:
    
    # input_size: number of input
    # hidden_size: number of hidden layer units; last number should be output size
    # activation: tuples 0f activation functions for hidden layers;
    # loss: loss function
    # metric: list of metric functions
    # optimizer: 'sgd': stochastic gradient descent, 'adam': adam optimizer
    def __init__(self, input_size:int, hidden_size:tuple, activation:tuple, loss:tuple, metric:list=[], optimizer:str='sgd', random_state:int=None):
        assert len(hidden_size) > 0, "at least one hidden layer should exist"
        assert len(hidden_size) == len(activation)
        
        # layer configurations
        self.__input_size = input_size
        self.__hidden_size = hidden_size
        
        # initialize activation and loss function
        self.__activation = activation
        self.__loss = loss
        self.__metric = metric
        
        # set optimizer
        if optimizer == 'sgd':
            self.__optimizer = self.__sgd
        elif optimizer == 'adam':
            self.__optimizer = self.__adam
        else:
            assert False, "invalid optimizer"
        
        # initialize weight
        self.__rand_seed = random_state
        self.reset_weights()


    # metric: list of metric functions
    # optimizer: 'sgd': stochastic gradient descent, 'adam': adam optimize
    def recompile(self, metric:list=None, optimizer:str=None):
        if metric:
            self.__metric = metric

        if optimizer:
            if optimizer == 'sgd':
                self.__optimizer = self.__sgd
            elif optimizer == 'adam':
                self.__optimizer = self.__adam
            else:
                assert False, "invalid optimizer"
        
            self.reset_weights() #reset weights

 
    # reset training attributes
    def reset_weights(self, random_state:int=None):
        if random_state:
              self.__rand_gen = np.random.RandomState(seed=random_state)
        else:
            self.__rand_gen = np.random.RandomState(seed=self.__rand_seed)

        self.__W, self.__B = [], [] #weight and bias
        
        # initialize weights and bias
        prev_size=self.__input_size
        for layer_size in self.__hidden_size:
            self.__W.append(self.__rand_gen.randn(prev_size, layer_size).astype(np.float128) * np.sqrt(2/prev_size)) #He initialization
            self.__B.append(np.zeros(shape=(1, layer_size)).astype(np.float128)) #initialize bias to zero
            prev_size=layer_size
            
        # reset 1st and 2nd moment for adam
        if self.__optimizer == self.__adam:
            self.__mW = [[0,0] for i in range(len(self.__hidden_size))]
            self.__mB = [[0,0] for i in range(len(self.__hidden_size))]
            self.__mWhat = [[0,0] for i in range(len(self.__hidden_size))]
            self.__mBhat = [[0,0] for i in range(len(self.__hidden_size))]

            # exponential decay rates for moment estimates
            self.__p = [0.9, 0.999]
            #small constant
            self.__eps = 1e-8


    # X: input data
    # return: yhat; predicted output
    def __forward_pass(self, X:np.ndarray):
        self.__A, self.__dA = [], [] #reset activations and da/dz
        
        a = X
        for i in range(len(self.__W)):
            self.__A.append(a)
            a = self.__activation[i][0](np.dot(a, self.__W[i])+self.__B[i]) #a*w+b
            self.__dA.append(self.__activation[i][1](a))

        return a


    # yhat: output of forward propagation
    # y: target output
    def __backward_pass(self, yhat:np.ndarray, y:np.ndarray):
        self.__dW, self.__dB = [], [] #reset dz/dw, dz/db

        E = self.__loss[1](yhat, y) * self.__dA[-1] #dC/da * da/dz
        for i, w in reversed(list(enumerate(self.__W))):
            dw = np.dot(self.__A[i].T, E) / E.shape[0] #dc/dw
            db = np.dot(np.ones(shape=(1, E.shape[0])).astype(np.float128), E) / E.shape[0] #dc/dw

            self.__dW.insert(0, dw)
            self.__dB.insert(0, db)
            
            if i > 0:
                E = np.dot(E, w.T) * self.__dA[i-1] #dC/da * da/dz *...* da/dz


    # learning_rate: learning rate
    def __update_weight(self, learning_rate:float):
        for i in range(len(self.__W)):
            self.__W[i] -= learning_rate * self.__dW[i]
            self.__B[i] -= learning_rate * self.__dB[i]
            #print("w + lr * dw: %g + %g * %g" % (self.__W[i], learning_rate, self.__dW[i]))
            #print("b + lr * db: %g + %g * %g" % (self.__B[i], learning_rate, self.__dB[i]))


    # X: input data
    # y: target output
    # batch_size: batch size
    # epochs: number of epochs
    # learning_rate: learning rate
    # valid_data: set of input and target validation data; (X_valid, y_valid)
    # history: dictionary object for loss and metric history
    # verbose: 0: silent, 1: print progress_bar and loss, 2: print all metrics one line per epoch
    def fit(self, X:np.ndarray, y:np.ndarray, batch_size=32, epochs:int=50, learning_rate:float=None, valid_data:tuple=None, history:dict=None, verbose:int=1):
        #setup data
        self.__X,  self.__y = self.__check_shape(X, y)
        self.__valid_data = self.__check_shape(valid_data[0], valid_data[1]) if valid_data else None
        
        for i_epoch in range(epochs):
            self.__optimizer(batch_size, learning_rate)

            #callbacks
            loss, metric = self.__validation()
            self.__history(loss, metric, history)
            self.__print_log(i_epoch+1, epochs, loss, metric, verbose)


    # X: input data
    # return: yhat; predicted output
    def predict(self, X:np.ndarray):
        return self.__forward_pass(self.__check_shape(X))
    

    # batch_size: batch size
    # learning_rate: learning rate
    def __sgd(self, batch_size:int, learning_rate:float):
        X_shuffle, y_shuffle = self.__shuffle() #shuffle sample order
        for num_batch in range(X_shuffle.shape[0]//batch_size):
            start_batch = num_batch*batch_size
            end_batch = start_batch+batch_size if start_batch+batch_size < X_shuffle.shape[0] else X_shuffle.shape[0]
            X_mini, y_mini = X_shuffle[start_batch: end_batch], y_shuffle[start_batch: end_batch] #mini batch

            #feed data to the network
            yhat_mini = self.__forward_pass(X_mini)
            self.__backward_pass(yhat_mini, y_mini)
            
            self.__update_weight(learning_rate if learning_rate else 0.01) #default learning_rate for sgd is 0.01
       

    # batch_size: batch size
    # learning_rate: learning rate
    def __adam(self, batch_size:int, learning_rate:float):
        X_shuffle, y_shuffle = self.__shuffle() #shuffle sample order

        for num_batch in range(X_shuffle.shape[0]//batch_size):
            start_batch, end_batch = num_batch*batch_size, (num_batch+1)*batch_size
            X_mini, y_mini = X_shuffle[start_batch: end_batch], y_shuffle[start_batch: end_batch] #mini batch

            #feed data to the network
            yhat_mini = self.__forward_pass(X_mini)
            self.__backward_pass(yhat_mini, y_mini)
            
            for i in range(len(self.__dW)):
                #update 1st and 2nd moments
                self.__mW[i][0] = self.__p[0] * self.__mW[i][0] + (1-self.__p[0]) * self.__dW[i] 
                self.__mW[i][1] = self.__p[1] * self.__mW[i][1] + (1-self.__p[1]) * (self.__dW[i]**2)

                self.__mB[i][0] = self.__p[0] * self.__mB[i][0] + (1-self.__p[0]) * self.__dB[i] 
                self.__mB[i][1] = self.__p[1] * self.__mB[i][1] + (1-self.__p[1]) * (self.__dB[i]**2)
                
                #bias correction
                self.__mWhat[i][0] = self.__mW[i][0] / (1 - self.__p[0]**(num_batch+1))
                self.__mWhat[i][1] = self.__mW[i][1] / (1 - self.__p[1]**(num_batch+1))

                self.__mBhat[i][0] = self.__mB[i][0] / (1 - self.__p[0]**(num_batch+1))
                self.__mBhat[i][1] = self.__mB[i][1] / (1 - self.__p[1]**(num_batch+1))

                #update parameter
                self.__dW[i] = self.__mWhat[i][0] / (np.sqrt(self.__mWhat[i][1]) + self.__eps)
                self.__dB[i] = self.__mBhat[i][0] / (np.sqrt(self.__mBhat[i][1]) + self.__eps)
            
            self.__update_weight(learning_rate if learning_rate else 0.001) #default learning_rate for adam is 0.001
    

    # return: loss, metric
    #         loss[0 or 1]; 0: train, 1: validation
    #         metric[i][0 or 1]; i is index of metric function in self.__metric
    def __validation(self):
        #calculate loss
        yhat = self.__forward_pass(self.__X) 
        loss = [self.__loss[0](yhat, self.__y), None]
        metric = []

        if self.__valid_data:
            valid_yhat = self.__forward_pass(self.__valid_data[0])
            loss[1]= self.__loss[0](valid_yhat, self.__valid_data[1])

        #calculate metrics
        for i, func in enumerate(self.__metric):
            metric.append([func(yhat, self.__y), None])

            if self.__valid_data:
                metric[-1][1] = func(valid_yhat, self.__valid_data[1])

        return loss, metric


    # loss: loss calculation for single epoch
    # metric: metrics calculation for single epoch
    # history: dictionary object for loss and metrics history
    def __history(self, loss:list, metric: list, history:dict):
        if not history:
            return
        
        #loss
        if 'train_loss' not in history:
            history['train_loss'] = []
        history['train_loss'].append(loss[0])

        if self.__valid_data:
            if 'valid_loss' not in history:
                history['valid_loss'] = []
            history['valid_loss'].append(loss[1])

        #metrics
        for i, func in enumerate(self.__metric):
            train_label = 'train_'+func.__name__

            if train_label not in history:
                history[train_label] = []
            history[train_label].append(metric[i][0])

            if self.__valid_data:
                valid_label = 'valid_'+func.__name__
                
                if valid_label not in history:
                    history[valid_label] = []
                history[valid_label].append(metric[i][1])

    
    # i_epoch: number of current epoch
    # max_epoch:  total number of epochs
    # loss: loss calculation for single epoch
    # metric: metrics calculation for single epoch
    # verbose: 0: silent, 1: print progress_bar and loss, 2: print all metrics one line per epoch
    def __print_log(self, i_epoch:int, max_epoch:int, loss:list, metric, verbose:int):
        if verbose == 0: #do nothing
            return

        PROGRESS_BAR_SIZE = 30

        log_str = ("epoch %"+str(len(str(max_epoch)))+"d/%d ")%(i_epoch, max_epoch) #i_epoch/max_epoch in str with spacing

        if verbose == 1:
            num_bar = int(i_epoch/max_epoch * PROGRESS_BAR_SIZE) #number of bar for progress
            log_str += ("[%-"+str(PROGRESS_BAR_SIZE)+"s]  ")%("="*num_bar)

            if not self.__valid_data:
                log_str += "loss: %g" % (loss[0])

            else:
                log_str += "train_loss: %g  valid_loss: %g" % (loss[0], loss[1])

            print(log_str, end='\r')

        else: #verbose == 2
            if not self.__valid_data:
                log_str += "  train_loss: %g  " % (loss[0])

            else:
                log_str += "  train_loss: %g  valid_loss: %g  " % (loss[0], loss[1])

            for i, func in enumerate(self.__metric):
                if not self.__valid_data:
                    log_str += "%s: %g  " % (func.__name__, metric[i][0])

                else:

                    log_str += "%s: %g  " % ('train_'+func.__name__, metric[i][0])
                    log_str += "%s: %g  " % ('valid_'+func.__name__, metric[i][1])

            print(log_str)

        return log_str
    
    
    # X: input data
    # y: target output
    # return X or (X, y)
    def __check_shape(self, X:np.ndarray, y:np.ndarray=None):
        if len(X.shape)==1:
            assert len(X.shape)==1 and self.__input_size == 1, "input size does not match"
            X = X.reshape(X.shape[0], 1) #reshape 1d array to matrix   
        else:
            assert X.shape[1] == self.__input_size, "input size does not match"
        
        if y is None:
            return X #return X only
            
        if len(y.shape)==1:
            assert len(y.shape)==1 and self.__hidden_size[-1] == 1, "output size does not match"
            y = y.reshape(y.shape[0], 1)
        else:
            assert y.shape[1] == self.__hidden_size[-1], "output size does not match"
            
        assert X.shape[0] == y.shape[0], "sample size does not match"
        
        return X.astype(np.float128), y.astype(np.float128)


    # return: X, y in shuffled sample order
    def __shuffle(self):
        ind = list(range(self.__X.shape[0]))
        self.__rand_gen.shuffle(ind)
        return self.__X[ind], self.__y[ind]

    
    # prints summary of the constructed network
    def summary(self):
        sum_unit, sum_param = self.__input_size, 0
        
        summary_str = "loss='%s', optimizer='%s'\n" % (self.__loss[0].__name__, self.__optimizer.__name__[2:])
        summary_str += '-'*50+'\n'
        
        # input layer
        summary_str += "%10s: %5s unit(s)\n" % ('input', self.__input_size)
        
        # hidden layer
        for i, size in enumerate(self.__hidden_size[:-1]):
            summary_str += "%10s: %5s unit(s), activation='%s'\n" % ('hidden_'+str(i), size, self.__activation[i][0].__name__)

            sum_unit += i
            sum_param += self.__W[i].shape[0]*self.__W[i].shape[1] + self.__B[i].shape[1]

        # output layer
        summary_str += "%10s: %5s unit(s), activation='%s'\n" % ('output', self.__hidden_size[-1], self.__activation[-1][0].__name__)
        summary_str += '-'*50+'\n'

        sum_unit += self.__hidden_size[-1]
        sum_param += self.__W[-1].shape[0]*self.__W[-1].shape[1] + self.__B[-1].shape[1]

        str_temp =  "total_unit: %s  total_param: %s " % (sum_unit, sum_param)
        summary_str += "%50s\n" % (str_temp)

        print(summary_str)

        return summary_str

