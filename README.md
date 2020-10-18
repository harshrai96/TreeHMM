# TreeHMM 

[Sonething on HMM](https://docs.google.com/document/d/15gZgCOISeQDQXO1C0CoqywoMY2ENmYYNke_fu-jUVJ0/edit) 


## How to install

### Create a virtual environment 

* At its core, the main purpose of Python virtual environments is to create an isolated environment for Python projects. This means that each project can have its own dependencies, regardless of what dependencies every other project has.

* On macOS and Linux:

        python3 -m venv envname
        
* On Windows:

        py -m venv envname        
### Activate the virtual environment

* On macOS and Linux:

        source env/bin/activate
* On Windows:

        .\env\Scripts\activate

### Run the following in the virtual environment to install the package

        pip install treehmm
    

## Package Documentation

There are six python files of interest.

* initHMM.py
* bwd_seq_gen.py
* fwd_seq_gen.py
* backward.py
* forward.py
* baumWelch.py


## initHMM.py : Initializing treeHMM with given parameters

### Description 

* Initializing treeHMM with given parameters

### Usage 

* def initHMM(States, Symbols, treemat, startProbs = None, transProbs = None, emissionProbs = None):

### Arguments

* states : It is a numpy array with first element being discrete state value for the cases(or positive) and second element being discrete state value for the controls(or negative) for given treeHMM.
* symbols: It is a list of numpy array consisting discrete values of
            emissions(where "M" is the possible number of emissions) for each
            feature variable
* adjacent_symmetry_matrix_values: It is the Adjacent Symmetry Matrix that
            describes the topology of the tree
* initial_probabilities: It is a numpy array of shape (N * 1) containing
            initial probabilities for the states, where "N" is the possible
            number of states(Optional). Default is equally probable states
* state_transition_probabilities: It is a numpy array of shape (N * N)
            containing transition probabilities for the states, where "N" is the
            possible number of states(Optional).
* emission_probabilities: It is a list of numpy arrays of shape (N * M)
            containing emission probabilities for the states, for each feature
            variable(optional). Default is equally probable emissions
	    
### Returns
        
* hmm: A dictionary describing the parameters of treeHMM

### Examples 

    sample_tree = np.array([0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape(5, 5)
    states = ['P', 'N']
    symbols = [['L', 'R']]
    hmm = initHMM.initHMM(states, symbols, sample_tree)


## bwd_seq_gen.py : Calculate the order in which nodes in the tree should be traversed during the backward pass(leaves to roots)

### Description

* Tree is a complex graphical model where we can have multiple parents and multiple children for a node. Hence the order in which the tree should be tranversed becomes significant. Backward algorithm is a dynamic programming problem where to calculate the values at a node, we need the values of the children nodes beforehand, which need to be traversed before this node. This algorithm outputs a possible(not unique) order of the traversal of nodes ensuring that the childrens are traversed first before the parents

### Usage 

* def bwd_seq_gen(hmm, number_of_levels=100):

### Arguments

* hmm: It is a dictionary given as output by initHMM.py file
* number_of_levels: No. of levels in the tree, if known. Default is 100

###  Returns

* backward_tree_sequence: A list of size "D", where "D" is the number of nodes in the tree

### Examples

    sample_tree = np.array([0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape(5, 5)
    states = ['P', 'N']
    symbols = [['L', 'R']]
    hmm = initHMM.initHMM(states, symbols, sample_tree)
    backward_tree_sequence = bwd_seq_gen.bwd_seq_gen(hmm)

## fwd_seq_gen.py : Calculate the order in which nodes in the tree should be traversed during the forward pass(roots to leaves)

### Description 

* Tree is a complex graphical model where we can have multiple parents and multiple children for a node. Hence the order in which the tree should be tranversed becomes significant. Forward algorithm is a dynamic programming problem where to calculate the values at a node, we need the values of the parent nodes beforehand, which need to be traversed before this node. This algorithm outputs a possible(not unique) order of the traversal of nodes ensuring that the parents are traversed first before the children.

### Usage 

* def fwd_seq_gen(hmm, number_of_levels=100):

### Arguments

* hmm: It is a dictionary given as output by initHMM.py file
* number_of_levels: No. of levels in the tree, if known. Default is 100

###  Returns

* forward_tree_sequence: A list of size "D", where "D" is the number of nodes in the tree

### Examples

    sample_tree = np.array([0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape(5, 5)
    states = ['P', 'N']
    symbols = [['L', 'R']]
    hmm = initHMM.initHMM(states, symbols, sample_tree)
    forward_tree_sequence = fwd_seq_gen.fwd_seq_gen(hmm)

## backward.py : Infer the backward probabilities for all the nodes of the treeHMM

### Description 

* backward calculates the backward probabilities for all the nodes

### Usage

* def backward(hmm, observation, backward_tree_sequence, kn_states = None):

### Arguments

* hmm: It is a dictionary given as output by initHMM.py file
* observation: observation is a list of list consisting "k" lists for "k" features, each vector being a character series of discrete emission values at different nodes serially sorted by node number
* backward_tree_sequence: It is a list denoting the order of nodes in which the tree should be traversed in backward direction(from leaves to roots).It's the output of bwd_seq_gen function.
* kn_states: (Optional) It is a (L * 2) dataframe where L is the number of training nodes where state values are known. First column should be the node number and the second column being the corresponding known state values of the nodes.

### Returns

* backward_probs: A dataframe of size (N * D) denoting the backward probabilities at each node of the tree, where "N" is possible no. of states and "D" is the total number of nodes in the tree

### Details

* The backward probability for state X and observation at node k is defined as the probability of observing the sequence of observations e_k+1, ... ,e_n under the condition that the state at node k is X. That is: b[X,k] := Prob(E_k+1 = e_k+1,... ,E_n = e_n | X_k = X) where E_1...E_n = e_1...e_n is the sequence of observed emissions and X_k is a random variable that represents the state at node k

### Examples 

    sample_tree = np.array([0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape(5, 5)  # for "X" (5 nodes) shaped tree
    states = ['P', 'N']  # "P" represent cases(or positive) and "N" represent controls(or negative)
    symbols = [['L', 'R']]  # one feature with two discrete levels "L" and "R"
    hmm = initHMM.initHMM(states, symbols, sample_tree)
    observation = [["L", "L", "R", "R", "L"]]
    backward_tree_sequence = bwd_seq_gen.bwd_seq_gen(hmm)
    data = {'node': [1], 'state': ['P']}
    kn_states = pd.DataFrame(data=data, columns=["node", "state"])
    backward_probs = backward.backward(hmm, observation, backward_tree_sequence, kn_states)

## forward.py : Infer the forward probabilities for all the nodes of the treeHMM

### Description 

* forward calculates the forward probabilities for all the nodes

### Usage

* def forward(hmm, observation, forward_tree_sequence, kn_states = None):

### Arguments

* hmm: It is a dictionary given as output by initHMM.py file
* observation: observation is a list of list consisting "k" lists for "k" features, each vector being a character series of discrete emission values at different nodes serially sorted by node number
* forward_tree_sequence: It is a list denoting the order of nodes in which the tree should be traversed in backward direction(from leaves to roots).It's the output of bwd_seq_gen function.
* kn_states: (Optional) It is a (L * 2) dataframe where L is the number of training nodes where state values are known. First column should be the node number and the second column being the corresponding known state values of the nodes.

### Returns

* forward_probs: A dataframe of size (N * D) denoting the forward probabilities at each node of the tree, where "N" is possible no. of states and "D" is the total number of nodes in the tree

### Details

* The forward probability for state X up to observation at node k is defined as the probability of observing the sequence of observations e_1,..,e_k given that the state at node k is X. That is: f[X,k] := Prob( X_k = X | E_1 = e_1,..,E_k = e_k) where E_1...E_n = e_1...e_n is the sequence of observed emissions and X_k is a random variable that represents the state at node k

### Examples 

    sample_tree = np.array([0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape(5, 5)  # for "X" (5 nodes) shaped tree
    states = ['P', 'N']  # "P" represent cases(or positive) and "N" represent controls(or negative)
    symbols = [['L', 'R']]  # one feature with two discrete levels "L" and "R"
    hmm = initHMM.initHMM(states, symbols, sample_tree)
    observation = [["L", "L", "R", "R", "L"]]
    forward_tree_sequence = fwd_seq_gen.fwd_seq_gen(hmm)
    data = {'node': [1], 'state': ['P']}
    kn_states = pd.DataFrame(data=data, columns=["node", "state"])
    forward_probs = forward.forward(hmm, observation, forward_tree_sequence, kn_states)


## baumWelch.py : It consists of two functions

* baumWelchRecursion
* baumWelch

## baumWelchRecursion : Implementation of the Baum Welch Algorithm as a special case of EM algorithm

### Description

* baumWelch recursively calls this function to give a final estimate of parameters for tree HMM Uses Parallel Processing to speed up calculations for large data. Should not be used directly.

### Usage

* def baumWelchRecursion(hmm, observation, kn_states=None, kn_verify=None):

### Arguments

* hmm: It is a dictionary given as output by initHMM.py file
* observation: observation is a list of list consisting "k" lists for "k" features, each vector being a character series of discrete emission values at different nodes serially sorted by node number
* kn_states: (Optional) It is a (L * 2) dataframe where L is the number of training nodes where state values are known. First column should be the node number and the second column being the corresponding known state values of the nodes
* kn_verify: (Optional) It is a (L * 2) dataframe where L is the number of validation nodes where state values are known. First column should be the node number and the second column being the corresponding known state values of the nodes

### Returns
* newparam: A dictionary containing estimated Transition and Emission probability matrices

### Examples

    sample_tree = np.array([0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape(5, 5)  # for "X" (5 nodes) shaped tree
    states = ['P', 'N']  # "P" represent cases(or positive) and "N" represent controls(or negative)
    symbols = [['L', 'R']]  # one feature with two discrete levels "L" and "R"
    hmm = initHMM.initHMM(states, symbols, sample_tree)
    data = {'node': [1], 'state': ['P']}
    kn_states = pd.DataFrame(data=data, columns=["node", "state"])
    data1 = {'node' : [2,3,4], 'state' : ['P','N','P']}
    kn_verify = pd.DataFrame(data = data1,columns=["node","state"])
    newparam = baumWelch.baumWelchRecursion(copy.deepcopy(hmm),observation,kn_states, kn_verify)

## baumWelch : Inferring the parameters of a tree Hidden Markov Model via the Baum-Welch algorithm

### Description

* For an initial Hidden Markov Model (HMM) with some assumed initial parameters and a given set of observations at all the nodes of the tree, the Baum-Welch algorithm infers optimal parameters to the HMM. Since the Baum-Welch algorithm is a variant of the Expectation-Maximisation algorithm, the algorithm converges to a local solution which might not be the global optimum. Note that if you give the training and validation data, the function will message out AUC and AUPR values after every iteration. Also, validation data must contain more than one instance of either of the possible states

### Usage

* def baumWelch(hmm, observation, kn_states=None, kn_verify=None, maxIterations=50, delta=1e-5, pseudoCount=0):


### Arguments

* hmm: It is a dictionary given as output by initHMM.py file
* observation: observation is a list of list consisting "k" lists for "k" features, each vector being a character series of discrete emission values at different nodes serially sorted by node number
* kn_states: (Optional) It is a (L * 2) dataframe where L is the number of training nodes where state values are known. First column should be the node number and the second column being the corresponding known state values of the nodes
* kn_verify: (Optional) It is a (L * 2) dataframe where L is the number of validation nodes where state values are known. First column should be the node number and the second column being the corresponding known state values of the nodes
* maxIterations: It is the maximum number of iterations in the Baum-Welch algorithm. Default is 50
* delta: Additional termination condition, if the transition and emission matrices converge, before reaching the maximum number of iterations(code{maxIterations}). The difference of transition and emission parameters in consecutive iterations must be smaller than code{delta} to terminate the algorithm. Default is 1e-5.
* pseudoCount: Adding this amount of pseudo counts in the estimation-step of the Baum-Welch algorithm. Default is zero

### Returns

* learntHMM:  A dictionary of three elements, first being the infered HMM whose representation is equivalent to the representation in initHMM, second being a list of statistics of algorithm and third being the final state probability distribution at all nodes.


### Examples

    sample_tree = np.array([0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape(5, 5)  # for "X" (5 nodes) shaped tree
    states = ['P', 'N']  # "P" represent cases(or positive) and "N" represent controls(or negative)
    symbols = [['L', 'R']]  # one feature with two discrete levels "L" and "R"
    hmm = initHMM.initHMM(states, symbols, sample_tree)
    data = {'node': [1], 'state': ['P']}
    kn_states = pd.DataFrame(data=data, columns=["node", "state"])
    data1 = {'node' : [2,3,4], 'state' : ['P','N','P']}
    kn_verify = pd.DataFrame(data = data1,columns=["node","state"])
    learntHMM = baumWelch.baumWelch(copy.deepcopy(hmm),observation,kn_states, kn_verify)

## Contributors

## Citations


