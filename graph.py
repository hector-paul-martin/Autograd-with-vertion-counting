import numpy as np


#throughout this project I represent derivitves as such, dy_dx = derivitve of y with respect to x
#dl_ds means derivitve loss W.R.T the value of the nodes initial value

#note that everything is built on the assumption that np.arrays/scalars will be used.  Some functions might function with python ints as a byproduct, but in general it is better to use numpy datatypes as that is what it is designed for

#global variable to track wether or not a child is unassigned, we need this becasue pythons __getitem__ does not return a variable
#  so instead when __getitem__ is ran we flip this flag, witch is unfiped by the get_child_getitem() function. 
# An error will be raised if a new node is created or backwards pass is initiated while this flag is flipped
unasigned_child = False


def get_child_for_setitem(parents):
    '''finds the child given a tuple of parents, THIS FUNCTION MUST BE RAN AFTER A GETITEM TO PUT THE RESULT OF GETITEM INTO THE CORRECT VAIRIABLE'''

    global unasigned_child

    node_levels = parents[0].levels
    max_parent_level = 0

    for parent in parents:
        max_parent_level = max(max_parent_level,parent.level)

    child = node_levels[max_parent_level+1][-1] 

    # if the setitem was done in tandem with in in place opp, e.g a[index] *= b what happens is we run a[index] = a[index].imul(b), witch first creates a slice node, and then an imul node, and then finaly the resulting replace node.
    #this means that the child we are seraching for is either 1 or three levels above the parents
    if type(child) == Slice:
        child = node_levels[max_parent_level + 3][-1]

    #set the unasigned_child flag to false now that the child has assumedly been put into a variable
    unasigned_child = False

    return child
    


def SGD(node_levels,learning_rate):
    'to be ran after the bacwards pass, for every root node with .should_update == True node.value -= self.dl_ds * learning rate '
    for node in node_levels[0]:
        node.update_parameters_SGD(learning_rate)

def backwards_pass(node_levels,loss_nodes:tuple):#loss nodes are the nodes whose values represent the loss, their values will be minimised
    '''gets graidietns with respect to loss nodes. 
    make sure to zero graidients from last backwards pass first.
    node_levels is a list of lists where node_levels[a][b] = the bth node at level a where level tracks topilogical order, the node class automaticly builds this list, so pass graph.node.levels into this function
     
    this function: sets loss nodes to have dl_ds = 1,
    walks graph in reverse topilogical order, for all graphs with nonzero dl_ds call.backwards() witch accumalates graidients into parents '''

    if unasigned_child:
        raise RuntimeError('a gettitem was ran without assigning the result to a variable, after running a[index] = b you MUST run:  a = get_child_for_setitem((a,b)) to put the result of setitem in a')

    if type(loss_nodes) != tuple:# I keep forgeting to make loss nodes a tuple for some reson so I put this to remind myself
        input('loss nodes should be a tuple!!!')
    
    #make los nodes dl_ds 1
    for node in loss_nodes:
        node.dl_ds += 1 # += so that it works for any size dl_ds
    
    for level in reversed(node_levels[1:]):  #all levels apart from level zero in reverse order, since node_levels stores in topilogical order reverse it so that we go in reverse topilogical order
        for node in level:
            if not np.all(node.dl_ds == 0):#doing backwards pass on nodes where dl_ds is zero woudl be a waste as thier dl_dp will also be 0.
                node.backwards()

def drop_non_updated_params(node_levels):
    '''removes internal referances to all root nodes that have should_update = False '''
    i = len(node_levels[0]) -1

    while i >= 0:
        if not node_levels[0][i].should_update:   
            del node_levels[0][i]
        i -= 1


def cleanup(node_levels):#deleats none root nodes and zeros grads and drops the dictionrays of remaining storages, becasue the graph is directed acyclic and storage node_dict is emptied referace counter will fall to zero and the gc will collect all the nodes and storages
    '''this function:
    removes internal referacnes to none root nodes   
    removes internal referaces for root nodes whose .should_update is false
    zeros grads for reamaining nodes 
    empties storage objects in all nodes, to revome circular refearces from in place nodes and allow for the node.key to reset
    
    
    becasue eatch node only referances its parents, and the storage dicts we are just lefted with a directed aciclic graph that the gc will clear automaticly'''

    empty_storages_node_dict(node_levels)

    del node_levels[1:] # deleats all nodes exept for the root nodes, 

    drop_non_updated_params(node_levels)

    for node in node_levels[0]:
        node.zero_grads()#zero grads

def empty_storages_node_dict(node_levels):
    'empties the storage objects node_dicts and resets the node.key to zero'
    for level in node_levels:
        for node in level:
            node.storage.node_dict = {}#reset to empty dict, this removes

    node.node_key = 0

def free_roots(node_levels):
    'remove graphs internal referances to all root nodes so they can be collected by the gc'
    del node_levels[0]

def zero_grads(node_levels):#only zero graidients for the root nodes, I dont see a situation where you would need allready made nodes to zero their grads. if user needs to they can do so manualy
    for node in node_levels[0]:
        node.zero_grads()

def free_non_roots(node_levels):
    'get rid of all internal referances to none root nodes so that they can be collected by the gc'
    del node_levels[1:]


#storage class to store data and its vertion counter together
class storage():#storage objects store data and a vertion counter
    """Holds data array plus version tracking for gradient safety.
    
    Detects when values needed for the backwards pass have been invalidated by in place operations.
    the node_dict maps node keys to the version used at forward-pass time.
    """
    def __init__(self,data):
        self.data = data
        self.vertion = [0] #store in a list so that veiws can share one vertion. vertion will incriment wenever an in place operation is done, if a value is needed to calculate a grad in the backwards pass that doesnt match the vertion of the forwards pass an error will be raised.  
        self.node_dict = {}  #store the vertion at time of forwards pass in a dictionary, this can be referanced in the backwards pass

    def unlink(self): # return a new storage object, with a copy of the data, vertion and node dict of self
        fresh_storage = storage(self.data.copy())#return a fresh storage with a copy of the data
        fresh_storage.vertion = self.vertion.copy()#set the vertion of the new storage to be the same as the old storage but not linked
        fresh_storage.node_dict = self.node_dict.copy()
        return fresh_storage


    def get_data_forwards(self,node):
        self.node_dict[node.key] = self.vertion[0]  #store the vertion used to create the node so that we can verify it hasnt changed in the backwards pass
        return self.data

    
    def get_data_backwards(self,node):
        if self.node_dict[node.key] != self.vertion[0]:  #this will also cause an error if the data wasnt called in the forwards pass
            raise RuntimeError(f'the value needed for the backwards pass at node{node} with key {node.key} has been changed by an in place operation')
        else:
            return self.data



#base node class       
class node:

    """Base class for all computation graph nodes.
    
    Represents a node in the computational graph. Subclasses implement specific
    operations by overriding operate() and derivitive()
    
    Node types:
        - outofplace: creates new storage-default
        - inplace: modifies parent's storage directly - bumps vertion counter
        - view: shares memory with parent but has own storage object - storage objects share one vertion counter
    
    Attributes:
        parents: tuple of parent nodes
        key: unique ID for this node
        level: topological level (roots=0)
        dl_ds: gradient of loss w.r.t. this node's value
        storage: storage object holding data and version info
    """
    __array_priority__ = 100 # higher than numpy arrays

    #these are shared across all node instances
    node_key = 0
    levels = []#this list will store lists of all nodes at eatch level in order:  [[nodes at level 0],[nodes at level 1]...] levels[i] only contains nodes whose parents are all in levels[<i].


    def value(self):
        '''returns a referance to the data associated with this node, do not edit this data, copy it first otherwise stuff will break'''
        return self.storage.data  

    
    def __init__(self,parents,extras = None):#parents should be a tuple of nodes, extras allows any extra data needed to be passed into the operation, e.g: slice indexing

        if unasigned_child:
            raise RuntimeError('a gettitem was ran without assigning the result to a variable, after running a[index] = b you MUST run:  a = get_child_for_setitem((a,b)) to put the result of setitem in a')

        self.parents = parents
        self.key = self.node_key#node_key is the global counter. key is this nodes unique ID, used by storages node_dict during the backwards pass to look up the version recorded in the forwards pass.
        self.node_key+=1 # incriment nodekey so that the next node wil have a unique key

        max_parent_level = 0
        parents_storages = []
        
        #find the level that you should be and put the values in a tuple
        for parent in parents:
            parents_storages.append(parent.storage)#get the parents values
            max_parent_level = max(max_parent_level,parent.level)
        
        parents_storages = tuple(parents_storages)
        self.level = max_parent_level + 1

        if extras is not None:
            result = self.operate(parents_storages,extras)
        else:
            result = self.operate(parents_storages)#operate should return result for out of place operations, for in place it should operate in place on the value retured from the storage.get_data_forwards

        #start your own dl_ds at 0, it will accumalate from children in backwards pass
        self.dl_ds = np.zeros_like(result,dtype= float) 
        
        #inset self into the class list based on level
        if len(node.levels) == self.level:
            node.levels.append([self])#if this is the first node of a its level create a new list with yourself in it
        else:
            node.levels[self.level].append(self)



        #find wether self is inplace outofplace or zero
        self.node_type = 'outofplace'

        #check inplace first and then veiw becasue may_share_memory of the same array will always be true
        #first check if it is an inplace operation
        for parent in self.parents:

            if parent.value() is result:#this checks if same object, i.e if the result is an in place of parent.value()
                self.node_type = 'inplace'
                self.storage = parent.storage#share storage for in place operations
                self.storage.vertion[0] += 1 #increment vertion as an inplace operation has changed the value 
                break        

        if self.node_type == 'outofplace':#only check for view if we know its not inplace.
            for parent in self.parents:
                if np.may_share_memory(parent.value(), result):#use may_share_memory as its faster and only gives false positives, so overly strict.
                    self.node_type = 'view'
                    self.storage = storage(result)#make storage item with this veiw

                    #note that vertion is wrapped in a list so that it is mutable and can be shared across storage objects
                    self.storage.vertion = parent.storage.vertion #share a vertion so that when veiw changes all other affected arrays also update vertion. I know this is overy strict but its impossible to know what is and isnt effected for any abitrary veiw returning operation
                    break

        if self.node_type == 'outofplace':#only set up for out of place node if we know it isnt veiw or in place
            self.storage = storage(result)
            
            

    def backwards(self):
        
        #get the parent storages
        parent_storages = []
        for parent in self.parents:
            parent_storages.append(parent.storage)
        

        dl_dp = self.derivitive(tuple(parent_storages))

        
        #accumalate graidients in the parents
        for i in range(len(dl_dp)):
            self.parents[i].dl_ds += dl_dp[i]



    def zero_grads(self):
        #set grads to zero
        if type(self.dl_ds) == int:
            self.dl_ds = 0
        else:
            self.dl_ds = np.zeros_like(self.dl_ds)
    
    def __getitem__(self, key):#

        mask = np.ones_like(self.value())
        mask[key] = 0
         
        newnode = Slice((self,),key)
        return newnode

    def __setitem__(self,key,other):
        global unasigned_child
        '''note that if this function is called due to a line such as node[key] a= <value> (where a is an arbitray in place operation) other will be the result of node[key] a <value>,
        becasue of this we can just do replacevalues(other) becasue even though there will be no graidient flow between node and newnode, the graidients will flow from newnode into other and then into node
        if other isnt the result of a calulation invloveling node then the replaced value is just an arbitrary value and the graidient flow should be zero
        
        becasue setitem doesnt return anything we need to use get_child_for_setitem() imidiately after the setitem line to put the result in the varaible'''
        if isinstance(other,node):
            newnode = ReplaceValues((self,other),key)
        else:
            newnode = ReplaceValuesOneNode((self,),(key,other))

        #set the global unasigned child flag to True, if a new node is created or backwards pass is initiated befor find_child() resets it to False an error will be raised
        unasigned_child = True
        
    def __add__(self, other):
        if isinstance(other,node):
            newnode = AddOutOfPlace((self,other))
        elif np.isscalar(other) or isinstance(other,np.ndarray):
            newnode = AddOutOfPlaceOneNode((self,),other)
        else:
            raise TypeError(f'cannot do add operation with node and {type(other)}')
        return newnode

    def __radd__(self, other):

        newnode = self + other#adding isnt effected by order so just do the normal add
        return newnode

    def __iadd__(self,other):
        if isinstance(other,node):
            newnode = AddInPlace((self,other))
        elif np.isscalar(other) or isinstance(other,np.ndarray):
            newnode = AddInPlaceOneNode((self,),other)
        else:
            raise TypeError(f'add operation not supported with node and {type(other)}')
        return newnode

    def __neg__(self):
        newnode = Negative((self,))
        return newnode

    def __mul__(self,other):
        if isinstance(other,node):
            newnode = ElementMulOutOfPlace((self,other))
        elif np.isscalar(other) or isinstance(other,np.ndarray):
            newnode = ElementMulOneNodeOutOfPlace((self,),other)
        else:
            raise TypeError(f'cannot do multiplication with node and type {type(other)}')
        return newnode

    def __rmul__(self,other):
        return self * other

    def __imul__(self, other):
        """
        Two-node in-place mul is intentionally unsupported. The gradient
        w.r.t. the second operand needs the value of the first
        operand, but p1 has already been overwritten in the forward pass. 
        Caching p1 would cost the same as an out-of-place mul so just do out of place instead.
        """

        if np.isscalar(other) or isinstance(other,np.ndarray):
            newnode = ElementMulInPlace((self,),other)
        else:
            raise TypeError(f'cannot do in place mul with node and type{type(other)} imul must be with a scaler / np array. NOT A NODE')
        return newnode

    def __matmul__(self, other):
        if isinstance(other,node):
            newnode = MatMulOutOfPlace((self,other))
        else:
            raise TypeError(f'cannot matmul between node and type {type(other)}')
        return newnode

    def fresh_storage(self):#gives the node a fresh storage, so that it is not going to be affected by in place operations on other nodes
        self.storage = self.storage.unlink()

class root(node):#root nodes are the parameters and inputs, they have no parents, forwards, backwards, operate or derivitive methods
    def __init__(self,value,should_update):
        self.should_update = should_update
        self.key = node.node_key#although the only reson to have the node key is so that the storage objects can track where their data was used it makes sense to give root nodes a unique key aswell, never know when u need it
        node.node_key += 1

        self.level = 0
        self.storage = storage(value.astype(float)) #cast to float


        self.dl_ds = np.zeros_like(self.value(),dtype= float)

        if node.levels == []:
            node.levels.append([self])
        else:
            node.levels[0].append(self)

    def update_parameters_SGD(self,learning_rate):
        if self.should_update:
            if self.storage.vertion[0] != 0:
                raise RuntimeError(f'parameters that are set to update should not be editied by in place operations. parameter {self} was edited by in place operation')
            self.storage.data -= self.dl_ds * learning_rate

#operator classes:

def get_brodcast_dims(p1_shape,p2_shape):
        '''takes two tuples who are the dimentions of two arrays, i.e the value returned by np.shape(array)
         and returns the dimentions across witch each array would be brodcast in an element wise operation between the two
         this is usefull for defineing the derivitves of element wise operations that allow brodcasting, as it tells you witch dimentions to sum the graidients across.'''
        p1_brodcast_dims = []
        p2_brodcast_dims = []

        p1_dims = len(p1_shape)
        p2_dims = len(p2_shape)

        #pad the shape to align the arrays from the right like numpy does
        if p1_dims < p2_dims:
            padded_p1_shape = (1,)* (p2_dims - p1_dims) + p1_shape
            result_dims = p2_dims
            padded_p2_shape = p2_shape
        else:#if they have equal number of dims they must be the same shape otherwise the operation would have failed
            padded_p2_shape = (1,)* (p1_dims - p2_dims) + p2_shape
            result_dims = p1_dims
            padded_p1_shape = p1_shape

        
        #find the dimentions the array was brodcast over
        for i in range(result_dims):
            if padded_p2_shape[i] == 1 and padded_p1_shape[i] != 1:
                p2_brodcast_dims.append(i)
            elif padded_p1_shape[i] == 1 and padded_p2_shape[i] != 1:
                p1_brodcast_dims.append(i)

        return tuple(p1_brodcast_dims),tuple(p2_brodcast_dims)

class Slice(node):
    def operate(self,parents_storages,index):
        self.index = index

        data = parents_storages[0].get_data_forwards(self)#use get data forwards to collect data from nodes

        self.base_shape = np.shape(data)#this can be stored safley as numpy arrays shapes cannot be changed in place
        return data[index]

    def derivitive(self,parent_storages):
        dl_dp = np.zeros(shape=self.base_shape)#create an array with identical dims to parent, and put derivitive from this node in
        dl_dp[self.index] += self.dl_ds
        return (dl_dp,)#a tuple is expected


class ElementMulOutOfPlace(node):#hardcoded two parent vertion to handle the most commen case more efficiently
    def operate(self,parents_storages):
        p1 = parents_storages[0].get_data_forwards(self)
        p2 = parents_storages[1].get_data_forwards(self)
        result = p1*p2
        return result
    
    def derivitive(self,parent_storages):
        p1 = parent_storages[0].get_data_backwards(self)
        p2 = parent_storages[1].get_data_backwards(self)

        if np.shape(p1) == np.shape(p2):
            return (p2*self.dl_ds,p1*self.dl_ds)#this means no brodcasting has happended, and we can just return the graidients directly

        #find the dims that p1 and p2 were brordcast across, so that we can sum across said dims for acurate grads.


        p1_shape = np.shape(p1)
        p2_shape = np.shape(p2)

        p1_brodcast_dims,p2_brodcast_dims = get_brodcast_dims(p1_shape,p2_shape)


        #sum over brodcasted dimentions to attain the true , 
        #note that we don't need to pad at all as numpy brodcasts in the self.dl_ds*p1/2 operation
        dl_dp1 = np.sum(self.dl_ds*p2 , axis = p1_brodcast_dims)

        dl_dp2 = np.sum(self.dl_ds*p1 , axis = p2_brodcast_dims)

        #reshape dl_dp1 and dl_dp2 to give back their 1 dimentions that were lost in the sum.
        dl_dp1 = np.reshape(dl_dp1,shape = p1_shape)
        dl_dp2 = np.reshape(dl_dp2,shape = p2_shape)

        return(dl_dp1,dl_dp2)

class ElementMulOneNodeOutOfPlace(node):
    '''note that self.multiplier is stored as a referacne not as a copy for efficientcy, 
    if the multiplier passed into this function is edited there will be incorrect graidients!!!'''

    def operate(self,parents_storages,mulitplier):
        tensor = parents_storages[0].get_data_forwards(self)

        result = tensor * mulitplier

        #store for backwards pass, store the parents shape to account for in place operations.
        self.multiplier = mulitplier
        self.parent_shape = np.shape(tensor)

        return result

    def derivitive(self,parent_storages):

        if self.parent_shape == np.shape(self.multiplier):
            return (self.multiplier*self.dl_ds,)#this means no brodcasting has happended, and we can just return the graidients directly

        #find the dims that p1 and p2 were brordcast across, so that we can sum across said dims for acurate grads.
        brodcast_dims = []


        mul_shape = np.shape(self.multiplier)
        mul_dims = len(mul_shape)
        parent_dims = len(self.parent_shape)

        #pad the shape to align the arrays from the right
        if mul_dims < parent_dims:
            padded_mul_shape = (1,)* (parent_dims - mul_dims) + mul_dims
            result_dims = parent_dims
            padded_parent_shape = self.parent_shape
        else:#if they have equal number of dims they must be the same shape otherwise the operation would have failed
            padded_parent_shape = (1,)* (mul_dims - parent_dims) + self.parent_shape
            result_dims = mul_dims
            padded_mul_shape = mul_shape

        
        #find the dimentions the array was brodcast over
        for i in range(result_dims):
            if padded_parent_shape[i] == 1 and padded_mul_shape[i] != 1:
                brodcast_dims.append(i)

        


        dl_dp = np.sum(self.multiplier*self.dl_ds,(tuple(brodcast_dims)))
        dl_dp = np.reshape(dl_dp,(self.parent_shape))
        return (dl_dp,)  
    
class ElementMulInPlace(node):
    '''note that self.multiplier is stored as a referacne not as a copy for efficientcy, 
    if the multiplier passed into this function is edited there will be incorrect graidients!!!'''
    def operate(self,parents_storages,multiplier):
        value = parents_storages[0].get_data_forwards(self)

        value *= multiplier

        #store for backwards pass.
        self.multiplier = multiplier
        self.parent_shape = np.shape(value)
        return value


    def derivitive(self,parent_storages):

        if self.parent_shape == np.shape(self.multiplier):
            return (self.multiplier*self.dl_ds,)#this means no brodcasting has happended, and we can just return the graidients directly

        #find the dims that p1 and p2 were brordcast across, so that we can sum across said dims for acurate grads.
        brodcast_dims = []


        mul_shape = np.shape(self.multiplier)
        mul_dims = len(mul_shape)
        parent_dims = len(self.parent_shape)

        #pad the shape to align the arrays from the right
        if mul_dims < parent_dims:
            padded_mul_shape = (1,)* (parent_dims - mul_dims) + mul_dims
            result_dims = parent_dims
            padded_parent_shape = self.parent_shape
        else:#if they have equal number of dims they must be the same shape otherwise the operation would have failed
            padded_parent_shape = (1,)* (mul_dims - parent_dims) + self.parent_shape
            result_dims = mul_dims
            padded_mul_shape = mul_shape

        
        #find the dimentions the array was brodcast over
        for i in range(result_dims):
            if padded_parent_shape[i] == 1 and padded_mul_shape[i] != 1:
                brodcast_dims.append(i)

        


        dl_dp = np.sum(self.multiplier*self.dl_ds,tuple(brodcast_dims))

        #reshape to preserve 1 dims that were destroyed in the sum
        dl_dp = np.reshape(dl_dp,(self.parent_shape))
        return (dl_dp,)        

class AddInPlace(node):
    def operate(self,parents_storages):
        p1 = parents_storages[0].get_data_forwards(self)
        p2 = parents_storages[1].get_data_forwards(self)

        p1 += p2
        self.p1_shape = np.shape(p1)
        self.p2_shape = np.shape(p2)

        return p1

    def derivitive(self,parents_storages):

        if self.p1_shape == self.p2_shape:#if same shape then no bordcast so just return self.dl_ds
            return (self.dl_ds,self.dl_ds)

        
        p1_brodcast_dims,p2_brodcast_dims = get_brodcast_dims(self.p1_shape,self.p2_shape)


        #sum across brodcast dims to get grads
        dl_dp1 = np.sum(self.dl_ds,tuple(p1_brodcast_dims))
        dl_dp2 = np.sum(self.dl_ds,tuple(p2_brodcast_dims))

        #rehsape to presrve dims lost in the sum
        dl_dp1 = np.reshape(dl_dp1,self.p1_shape)
        dl_dp2 = np.reshape(dl_dp2,self.p2_shape)

        return (dl_dp1,dl_dp2)

class AddOutOfPlace(node):
    def operate(self,parents_storages):

        p1 = parents_storages[0].get_data_forwards(self)
        p2 = parents_storages[1].get_data_forwards(self)

        result = p1 + p2
        self.p1_shape = np.shape(p1)
        self.p2_shape = np.shape(p2)

        return result

    def derivitive(self,parents_storages):

        if self.p1_shape == self.p2_shape:
            return (self.dl_ds,self.dl_ds)
         
        p1_brodcast_dims,p2_brodcast_dims = get_brodcast_dims(self.p1_shape,self.p2_shape)


        #sum across brodcast dims to get grads
        dl_dp1 = np.sum(self.dl_ds,tuple(p1_brodcast_dims))
        dl_dp2 = np.sum(self.dl_ds,tuple(p2_brodcast_dims))

        #rehsape to presrve 1 dims lost in the sum
        dl_dp1 = np.reshape(dl_dp1,self.p1_shape)
        dl_dp2 = np.reshape(dl_dp2,self.p2_shape)

        return (dl_dp1,dl_dp2)

class AddOutOfPlaceOneNode(node):
    def operate(self,parents_storages,adder):
        p1 = parents_storages[0].get_data_forwards(self)

        result = adder + p1

        self.p1_shape = np.shape(p1)
        self.adder_shape = np.shape(adder)

        return result

    def derivitive(self,parents_storages):

        if self.p1_shape == self.adder_shape:
            return (self.dl_ds,)
         
        p1_brodcast_dims,adder_brodcast_dims = get_brodcast_dims(self.p1_shape,self.adder_shape)


        #sum across brodcast dims to get grads
        dl_dp1 = np.sum(self.dl_ds,tuple(p1_brodcast_dims))

        #rehsape to presrve 1 dims lost in the sum
        dl_dp1 = np.reshape(dl_dp1,self.p1_shape)

        return (dl_dp1,)

class AddInPlaceOneNode(node):
    def operate(self,parents_storages,adder):
        p1 = parents_storages[0].get_data_forwards(self)

        self.p1_shape = np.shape(p1)
        self.adder_shape = np.shape(adder)

        p1 += adder

        return p1

    def derivitive(self,parents_storages):

        if self.p1_shape == self.adder_shape:#no brodcasting return self.dl_ds
            return (self.dl_ds,)
         
        p1_brodcast_dims,adder_brodcast_dims = get_brodcast_dims(self.p1_shape,self.adder_shape)


        #sum across brodcast dims to get grads
        dl_dp1 = np.sum(self.dl_ds,tuple(p1_brodcast_dims))

        #rehsape to presrve 1 dims lost in the sum
        dl_dp1 = np.reshape(dl_dp1,self.p1_shape)

        return (dl_dp1,)

class MatMulOutOfPlace(node):#this will do the matmul in the order of the first two parents.  Assumes that neither is greater than 3 dims, It might still work but numpy brodcasting idk?
    def operate(self,parents_storages):
        left = parents_storages[0].get_data_forwards(self)
        right = parents_storages[1].get_data_forwards(self)

        result = left @ right
        return result

    def derivitive(self,parents_storages):
        '''what this function does is:
        
            reshape the inputs and dl_ds to the mathmatical shape
                -by adding the dummy dimentions that numpy would have used in the forwards pass
                -and flattening extra dims into one batch dimention, in the same way numpy woudl have in the forwards pass (we have do mess about a bit with rights batch dimention to put it in the collums becasue numpy reshapes in row majour order)
            then it finds the derivitive wrt right and left useing the normal method
            
            note that in the case where they both have the same number of batch dims we dont fully flattern, we cheat a little and use the numpy batched multiplication, meaning we only flatten to 3 dims for the sake of makeing the transpose eaisier.
            this flattening can be done becasue the rows in left, and the collums in right are inderpendant of eatch other, so the result of a multiplication involveing batch dims is the same no matter how mutch I flatten it as long as i preseve the collums / rows
            this property arises form the definition of a matmul as concecutive tranformations, as where one baisis vector moves is inderpendant of where the other goes'''


        
        dl_ds = self.dl_ds #dl_ds will be reshaped in this function, so use this local variable to store it.  I know this is just a referance but I need somewhere to store the reshaped veiws and it makes sense to use one variable througout

        left = parents_storages[0].get_data_backwards(self)
        right = parents_storages[1].get_data_backwards(self)



        L_shape = np.shape(left)
        R_shape = np.shape(right)
        dl_ds_shape = np.shape(dl_ds)
        L_dims = len(L_shape)
        R_dims = len(R_shape)

        #deal with 1D arrays
        left_is_reshaped = False
        right_is_reshaped = False

        if L_dims == 1:
            left = np.reshape(left,(1,-1))#rehsape l to its mathmatical shape
            left_is_reshaped = True
            left_og_shape = L_shape
            L_shape = np.shape(left)
            L_dims = 2

            if dl_ds_shape == ():
                dl_ds_shape = (1,1)
            else:
                dl_ds_shape = dl_ds_shape[:-1]+(1,)+(dl_ds_shape[-1],)
            dl_ds = np.reshape(dl_ds, dl_ds_shape )#reshape dl_ds to its mathmatical shape, insert a one in
            

        if R_dims == 1:
            right = np.reshape(right,(-1,1))
            right_is_reshaped = True
            right_og_shape = R_shape
            R_shape = np.shape(right)
            R_dims = 2

            if dl_ds_shape == ():
                dl_ds_shape = (1,1)
            else:
                dl_ds_shape = (dl_ds_shape) + (1,)
            dl_ds = np.reshape(dl_ds,dl_ds_shape)
            




        if L_dims == 2 and R_dims == 2:#normal matrix multiplication, no batching
            dl_dright = left.T @ dl_ds
            dl_dleft = dl_ds @ right.T

        #batch matrix multiplication
        elif L_dims == R_dims:#in this case numpy does batched multiplication, so we can flatten the batch dimentions into one so that transpose still works,

            #flatten and transpose
            #note that -1 will be the product of the batch dims, the reshape at the end will put us back into the same shape
            left_flat = np.reshape(left,(-1,L_shape[-2],L_shape[-1]))
            left_flat_T = np.swapaxes(left_flat,-1,-2)

            right_flat = np.reshape(right,(-1,R_shape[-2],R_shape[-1]))
            right_flat_T = np.swapaxes(right_flat,-1,-2)

            dl_ds_flat = np.reshape(dl_ds,(-1,dl_ds_shape[-2],dl_ds_shape[-1]))

            #get derivitives and reshape to put the batch dims back how they were
            dl_dleft = np.reshape(dl_ds_flat @ right_flat_T,(L_shape))
            dl_dright = np.reshape(left_flat_T @ dl_ds_flat,(R_shape))

        elif R_dims == 2:#only batching on the left dimention so right is shared across batches
            left_superflat = np.reshape(left,(-1,L_shape[-1]))

            #reshape to only have one batch dimention
            dl_ds_superflat = np.reshape(dl_ds,(-1,dl_ds_shape[-1]))

            #this automaticaly sums across the batch dimention
            dl_dright = left_superflat.T @ dl_ds_superflat

            dl_dleft = dl_ds @ right.T  #numpy should do batched multiplication across the batch dims in dl_ds

        elif L_dims == 2:#only batching on the right dimenton so left is shared across batches


            right_swapped = np.swapaxes(right,-1,-2)#traspose last dims so that the rows become the colums
            right_superflat_T = np.reshape(right_swapped,(-1,R_shape[-2]))#now that the collums have become the rows they will be preserved in the reshape
  

            dl_ds_swapped = np.swapaxes(dl_ds,-1,-2)
            dl_ds_superflat_T = np.reshape(dl_ds_swapped , shape = (-1,dl_ds_shape[-2]))
            dl_ds_superflat = dl_ds_superflat_T.T#undo the transpose

            dl_dleft = dl_ds_superflat @ right_superflat_T

            dl_dright =  left.T @ dl_ds



        #reshape back to one-d to match left/right.dl_ds if a dummy dim was added to match mathmatical shape
        if right_is_reshaped:
            dl_dright = np.reshape(dl_dright, right_og_shape)
        if left_is_reshaped:
            dl_dleft = np.reshape(dl_dleft,left_og_shape)

        return(dl_dleft,dl_dright)
        
class Summation(node):
    def operate(self,parent_storages,axis_keepdims):#expects axis keepdims to be (tuple of axis to sum over,wether or not keep dims) to sum over all axis input an empty tuple
        data = parent_storages[0].get_data_forwards(self)

        if axis_keepdims[0] == ():
            result = np.sum(data,keepdims=axis_keepdims[1])
        else:
            result = np.sum(data,axis=axis_keepdims[0],keepdims=axis_keepdims[1])

        return result

    def derivitive(self,parent_storages):
        dl_dp = self.dl_ds

        return (dl_dp,)#this will brodcast to the parents array so its fine

class LeakyRelu(node):
    def operate(self,parent_storages,alpha = 0.01):
        parent = parent_storages[0].get_data_forwards(self)

        result = np.where(parent < 0, parent*alpha,parent)

        self.ds_dp = np.where(parent < 0 ,alpha, 1)  #this mask represents ds_dp, we can times it by dl_ds element wise to get dl_dp, make it now so that inplace ops on parent can be allowed
        self.alpha = alpha

        return result

    def derivitive(self,parent_storages):

        dl_dp = self.dl_ds * self.ds_dp
        return (dl_dp,)
    
class Softmax(node):#applies the softmax function across the given axis
    def operate(self,parent_storages,axis = -1):  #automaticaly applies across rows if last axis not given

        x = parent_storages[0].get_data_forwards(self)

        #shift x to keep the numbers smaller.  The shift doesnt effect the probabilites becasue shift in the x direction on exponential is the same as scaleing the y axis. since x_exp and x_exp_sum are both shifted the shifts cacel out in the divition
        x_shifted = x - np.max(x,axis,keepdims=True) # shift by max x so that the largest value is zero to prevent overflow

        x_exp = np.exp(x_shifted)
        
        x_exp_sum = np.sum(x_exp,axis,keepdims=True)

        result = x_exp/x_exp_sum 

        self.axis = axis
        return result

    def derivitive(self):
        raise NotImplementedError('i havent implemented this yet, usualy softmax and corss entropy are used together anyway so just use that instead, only use this function for output')

class SoftmaxCrossEntropy(node):
    def operate(self,parent_storages,axis_True_distrabution):#axis_true_distrabution should be a tuple with, the axis to do softmax + cross entopy over, and the 
        'input should be format ((parent),(axis, true distabution))'
        operation_axis = axis_True_distrabution[0]
        self.true_distrabution = axis_True_distrabution[1]

        #softmax----------------------------------------

        logits = parent_storages[0].get_data_forwards(self)

        #shift x to keep the numbers smaller.  The shift doesnt effect the probabilites becasue shift in the x direction on exponential is the same as scaleing the y axis. since x_exp and x_exp_sum are both shifted the shifts cacel out in the divition
        logits_shifted = logits - np.max(logits,operation_axis,keepdims=True) # shift by max x so that the largest value is zero to prevent overflow

        logits_exp = np.exp(logits_shifted)
        logits_exp_sum = np.sum(logits_exp,operation_axis,keepdims=True)

        #store probabilites as an attribute of self so that they can be accessed
        self.probabilitys = logits_exp/logits_exp_sum 



        #cross entropy-----------------------------------
        result = np.sum(  -np.log(self.probabilitys) *self.true_distrabution  ,axis = operation_axis)

        return result



    def derivitive(self,parent_storages):
        dl_dp = self.probabilitys - self.true_distrabution

        return(dl_dp,)

class Negative(node):

    def operate(self,parents_storages):
        data = parents_storages[0].get_data_forwards(self)

        result = - data
        return result

    def derivitive(self,parents_storages):
        dl_dp = -self.dl_ds

        return (dl_dp,)

class recipricol(node):

    def operate(self,parent_storages):
        operand = parent_storages[0].get_data_forwards(self)
        result = 1 / operand

        
        return result

    def derivitive(self,parent_storages):

        operand = parent_storages[0].get_data_backwards(self)

        dp_ds = - 1/np.square(operand)

        dl_dp = self.dl_ds * dp_ds

        return(dl_dp,)

class ReplaceValues(node):
    def operate(self,parent_storages,index):
        base_array = parent_storages[0].get_data_forwards(self)
        ovewriting_array = parent_storages[1].get_data_forwards(self)

        base_array[index] = ovewriting_array
        self.index = index
        self.base_shape = np.shape(base_array)

        return base_array

    def derivitive(self,parent_storages):

        dl_dbase = np.ones(shape=self.base_shape)

        dl_dbase[self.index] = 0

        dl_dbase *= self.dl_ds



        dl_doverwrite = self.dl_ds[self.index]

        return (dl_dbase,dl_doverwrite)

class ReplaceValuesOneNode(node):
    def operate(self,parent_storages,index_overwrite):

        base = parent_storages[0].get_data_forwards(self)

        overwrite = index_overwrite[1]
        self.index = index_overwrite[0]

        self.base_shape = np.shape(base)

        base[self.index] = overwrite

        return base

    def derivitive(self,parent_storages):

        #make a mask with zero at the overwritten arrays
        dl_dbase = np.ones(shape=self.base_shape)
        dl_dbase[self.index] = 0

        #use the mask on dl_ds to get dl_dbase
        dl_dbase *= self.dl_ds

        return (dl_dbase,)
