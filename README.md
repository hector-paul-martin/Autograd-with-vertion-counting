# Autograd-with-vertion-counting
An autograd system with version counting implemented, allowing for in-place and view operations to happen safely.  Created to learn how autograd works.  this repo also contains an FFN made with my system to demosntrate how it works


## Overview
This library tracks operations on NumPy arrays by wrapping them in node (and storage) objects. Each operation creates a new node that knows how to compute its local derivative with respect to its parents. Calling backwards_pass() walks the graph in reverse topological order, accumulating gradients (dl_ds) at every node.

node objects all inherit from the node class,  operation nodes are defined by overriding the operate() and derivitive() methods, operate must return the result of the calculation, even if it was an inplace operation.  This is so that the nodes type (inplace, veiw, outofplace) can be decided.  derivitive must return a tuple, where the ith element in said tuple is a numpy array of the derivitive wrt the loss of the ith parent.

The design prioritizes:

Explicit in-place vs. out-of-place semantics — in-place operations bump a version counter so the engine can detect when a value needed for the backward pass has been invalidated.

Broadcasting correctness — gradients are summed across broadcast dimensions automatically.

Memory efficiency — a cleanup() routine drops intermediate nodes and clears storage dictionaries, letting Python's garbage collector reclaim memory once user references to nodes are overwritten in the next training iteration.


### Gradients
dl_ds — derivative of the loss with respect to a node's value.

Loss nodes are seeded with dl_ds = 1 before the backward pass.

Each node's derivitive() method returns a tuple of gradients, one per parent.

### Node Types

outofplace - Creates a new storage object. Default.

inplace - Shares the parent's storage and increments its version counter.

view - Shares memory with the parent but has its own storage object; version counter is shared with the parent.

### Version Tracking

Each storage object holds a vertion list and a node_dict mapping node keys to the version seen during the forward pass. If an in-place operation modifies a value that the backward pass needs, get_data_backwards() raises a RuntimeError rather than silently producing wrong gradients.

### API Reference

backwards_pass(node_levels, loss_nodes) - Seeds loss nodes with dl_ds = 1 and walks the graph in reverse topological order. loss_nodes must be a tuple.

SGD(node_levels, learning_rate) - Updates all root nodes with should_update=True via value -= dl_ds * learning_rate.

zero_grads(node_levels) - Zeros gradients on all root nodes.

cleanup(node_levels) - Removes intermediate nodes, drops non-updating roots, zeros remaining gradients, and empties storage dictionaries to break reference cycles. means that once the user removes references to nodes they will be collected by gc, if user keeps references to high level nodes memory will fill up as each node will keeps its parents alive, witch keeps its parents alive ect...

free_roots(node_levels) - Removes the graph's references to root nodes.

free_non_roots(node_levels) - Removes the graph's references to all non-root nodes.

drop_non_updated_params(node_levels) - Removes root nodes with should_update=False from the graph.

## Node classes / available operations

### Leaf:

root(value, should_update) — parameters and inputs.

### Element-wise arithmetic:

AddOutOfPlace, AddInPlace

AddOutOfPlaceOneNode, AddInPlaceOneNode

ElementMulOutOfPlace, ElementMulInPlace

ElementMulOneNodeOutOfPlace

Negative

recipricol

### Matrix:

MatMulOutOfPlace — supports 1D, 2D, batched, and broadcast matmul with correct gradient reduction.

### Shape / indexing:

Slice — supports arbitrary NumPy indexing.

ReplaceValues — replaces values at an index with values from another node.

ReplaceValuesOneNode — replaces values at an index with a constant.

### Reduction:

Summation — sums over given axes with optional keepdims.

### Activations:

LeakyRelu — with configurable alpha.

Softmax — forward only; derivative not implemented.

SoftmaxCrossEntropy — fused softmax + cross-entropy with a numerically stable forward pass and the standard (probabilities - true_distribution) gradient.

## custom functions
by defining a class that inherets from the node class you can define your own custom functions.  node will asses whether the operation was inplace, outofplace or view-returning automaticly.  It will not detect if the result is a view/in place operation of a non parent value. So only define operations that are out of place, or view/inplace of a parent.  

do NOT perform in place operations on multiple paretns

### you will need to define:

#### operate(self,parent_storages, extras) note that extras is optional.
you MUST collect data from parents using: data = parent_storages[index].get_data_forwards(self).  this function is what tells the system to track the vertion for the backwards pass

operate must return the result of the operation,  for in place operations simply operate in place on the data, and then return a reference to the data that has been operated in place on, this must be done so that we can asses whether the node is in place, out of place or veiw.

if you want to cache data for backwards pass store it as an attribute of self
  
#### derivitive(self, parent_storages)
you MUST collect data from parents using: data = parent_storages[index].get_data_backwards(self).  this function is where the system checks the vertion counter.

return a tuple of form (dl_dparent1, dl_dparent2, dl_dparent3, ...)


## Operator Overloading
node supports the following Python operators:

Operator	Method	Notes
__add__, __radd__, +	supports Node or scalar/array

__iadd__, supports Node or scalar/array

__neg__	

*	__mul__, __rmul__	Node or scalar/array
*	
*=	__imul__	Scalar/array only; two-node in-place mul is unsupported (see docstring)

@	__matmul__	Node only

a[i]	__getitem__	Returns a Slice node

a[i] = b	__setitem__	See note below

### note on setitem
Important: Setitem and unasigned_child
Python's __setitem__ cannot return a value, so the resulting node cannot be assigned back to the variable in the same statement. After running:

a[index] = b
you must immediately run:

a = get_child_for_setitem((a, b))

This retrieves the newly created ReplaceValues node and assigns it to a. A module-level flag (unasigned_child) is set to True during __setitem__ and reset by get_child_for_setitem. If any new node is created or a backward pass is initiated while the flag is set, a RuntimeError is raised.

## Constraints and Gotchas

Use NumPy types. The library assumes np.ndarray or numpy scalars.

Don't mutate .value() in place. node.value() returns a reference to the underlying data. Copy it before editing, or use the provided in-place operations so version tracking stays consistent.

Parameters that update must not be edited in place. update_parameters_SGD checks that a parameter's version counter is still 0 and raises otherwise.

Two-node in-place multiplication is intentionally unsupported. The gradient with respect to the second operand requires the pre-multiplication value of the first operand, which has already been overwritten. Use out-of-place multiplication instead.

Softmax.derivitive() is not implemented. Use SoftmaxCrossEntropy when softmax is paired with cross-entropy loss, or use Softmax only for inference.

loss_nodes must be a tuple. The backward pass checks this and prompts if you forget.

items inputted as extras into operation nodes are stored by reference, not copied, for better memory efficiency. If you mutate them after the forward pass, gradients will be wrong.

Version sharing for views is conservative. Any view-returning operation shares a version counter with its parent, so in-place edits to any related array invalidate the view. This is intentionally strict because determining the exact set of affected arrays for an arbitrary view is not generally possible.


