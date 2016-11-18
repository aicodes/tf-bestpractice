TensorFlow Best Practices
------------------------------

This is a collection of gotchas collected by AI.codes engineers while working with TensorFlow. It is, without doubt, opinionated.

#### Working with computational graph

A TensorFlow model is essentially a computational graph, where nodes are [operations](https://www.tensorflow.org/versions/r0.11/api_docs/python/framework.html#Operation) and links in-between are [tensors](https://www.tensorflow.org/versions/r0.11/api_docs/python/framework.html#Tensor). The computational graph is defined as a  [GraphDef](https://www.tensorflow.org/versions/r0.9/how_tos/tool_developers/index.html#graphdef) protobuf message. `Graph` and `operation` are first class citizens in TF, where tensor is an intermediate object that is only used at runtime for passing data. From this perspective _OpGraph_ is probably a more accurate name than _TensorFlow_.

#### Object or not object

How you write the code to weave operations into the computational graph is a purely personal choice, as long as the code is clean and readable. What we find, in practice, is that using _light model objects_ helps organizing the code better. By light model object we mean grouping model into an object, and group the constructions of computational graph into components such as `inference`, `loss` and `optimize` (defined as member functions of the model object). However, do not go crazy about other OO features such as inherence and design patterns--they rarely make sense in this context. Organizing the computational graph into groups promotes code reuse and helps human readers to understand the overall model structure.

The code looks like this.
```python
class AwesomeModel(object):
  def __init__(self):
    """ init the model with hyper-parameters etc """

  def inference(self, x):
    """ This is the forward calculation from x to y """
    return some_op(x, name="inference")

  def loss(self, batch_x, batch_y=None):
    y_predict = self.inference(batch_x)
    self.loss = tf.loss_function(y, y_predict, name="loss") # supervised
    # loss = tf.loss_function(x, y_predicted) # unsupervised

  def optimize(self, batch_x, batch_y):
    return tf.train.optimizer.minimize(self.loss, name="optimizer")
```
You may notice that we name these operations. See the section below for a detailed explanation.

#### Use `Saver` liberally

Unless dealing with toy examples, we rarely just train a real-world model in one-shot. It may take days to train one. Model trained on a set of data may later gets trained further on a different (larger) set of data, or with a different set of hyper-parameters. In all these cases, treating training as an ongoing process is more rewarding than treating it as a one-shot process. This means you should include `saver.restore` and `saver.save` from the get go. It will help you immensely down the road. The `Saver` API doc is [here](https://www.tensorflow.org/versions/r0.11/api_docs/python/state_ops.html#Saver).

_Gotcha_: When saving variables, make sure that `Saver` class is constructed **after** all variables are defined. Saver only captures all variables at the time it is constructed, so any new variables defined after `Saver()` will not be saved automatically.

#### Use `tf.app.run()` and FLAGS from day one.

Along the same logic mentioned in the previous best practice, you are likely to tweak the model as you progress.  Because most of our models are end-to-end, there are bunch of different parameters to tweak. Using global variables to hold hyper-parameters will quickly go out of control, as there are too many of them.

At AI.codes, we enforce the convention that all hyper-parameters are defined as FLAGS. It is actually not hard to define flags, once you get used to it. Another advantage is that flags come with default values and documentation. Documentation is more important than you think--it becomes a nightmare three month later when you forget about why you set a global variable to certain magic number. With FLAGS you can document it easily. All it takes is a few lines of code.

```python
tf.app.flags.DEFINE_boolean("some_flag", False, "Documentation")

FLAGS = tf.app.flags.FLAGS

def main(_):
  # your code goes here...
  # use FLAGS.some_flag in the code.

if __name__ == '__main__':
    tf.app.run()
```
Note that we use `tf.app.run()` so it takes care of flag parsing. We put the main logic inside `main(_)`.


#### Name thy operations

As mentioned earlier, operation is first class citizen in TF. It is obvious that `tf.matmul` is an operation. It is less obvious that `tf.placeholder` and `tf.train.optimizer.minimize` are also operations. When we persist the model, all these operations are persisted.

When we first define the model Python, key operations such as `minimize` and `inference` are referenced using Python variables, and they can be used in code easily. When the computational graph is reloaded from GraphDef and checkpoint files, however, we no longer have the Python variables referring to them.

We recommend naming key operations that you may need later, so when the graph is reloaded, you can retrieve these operations by name, using `graph.get_operation_by_name(name)`. A good practice is to standardize the names from an agreed vocabulary. For instance, if we agree upon that "loss" is always used as the name for loss function in models, in MNIST example, we'd use
```python
tf.nn.softmax_cross_entropy_with_logits(logits, labels, dim=-1, name="loss")
```
to label the loss function. Shared vocabulary enables efficient communication between engineers. Here is a set of words in our vocabulary, independent of the model.

* x (x is the name of the placeholder. As mentioned, placeholder is an operation)
* y (in supervised case)
* loss
* inference
* learning_rate
* optimizer

Engineers can of course name additional operations in specific cases. We define the common vocabulary in a file that can be imported by other models, and use Python's `namedtuple` to add a bit type-safety, instead of using raw strings as keys. The following lines of code in `vocab.py` seems cryptic, but really it allows us to use references `vocab.x` or `vocab.inference` as names, instead of a more error-prone one like `vocab['inference']`.

```python
from collections import namedtuple

Vocab = namedtuple('Vocab', ['x', 'y', 'loss', 'inference', 'learning_rate', 'optimizer'])
vocab = Vocab('x', 'y', 'cost', 'inference', 'learning_rate', 'optimizer')
```

_Gotcha_: **Passing values to placeholders in restored models**

We know it well that we can use `feed_dict` to pass values to placeholders, when we have a references to the placeholders. For instance, if we have `x = tf.placeholder(...)`, we can later use `session.run(some_tensor, feed_dict={x: value})` to pass value to the computational graph. What is less obvious is when you restore the model, where you do not have a direct reference to `x` anymore.

If you correctly name your placeholder operation, as in `x = tf.placeholder(..., name=vocab.x)`, you can retrieve the operation later by using `graph.get_operation_by_name(vocab.x)`. A caveat here is that you are getting the **operation**, not the **tensor**. To get the actual tensor that can be used as the key in feed_dict, you can either take the ugly route, using `graph.get_tensor_by_name(vocab.x + ':0')`, or, a better way, `graph.get_operation_by_name(vocab.x).outputs[0]`. The ugly way works because tensors, as outputs of operations, are by default named as `operation_name:N` where `N` is the nth output. Placeholder tensor only has one output, thus the `:0` part.

_Gotcha_: **Evaluate operations in restored models**

`Session.run()` can take both operations and tensors. If you pass in operations, however, the returned value would be `None`. This is [documented](https://www.tensorflow.org/versions/r0.11/api_docs/python/client.html#Session) but you may easily skim through the document and get puzzled by the fact that you get nothing back. This actually reveal some peculiarities of TensorFlow's Python API that we will explain here.

Take a look at the statement `a = tf.placeholder(...)`. It looks like a constructor call, isn't it? You'd think that `a` must be of `placeholder` type, whatever that type is. Now let's look at `a = tf.matmul(B, C)`. Well, this time you may say that `a` should be a tensor, as the result of `B * C`. The truth is, `a` is tensor in both cases, except that the first case is a bit misleading in light of the Python coding convention.

The way to understand statements like `a = tf.op(..., name='operation_name')` is to break it down to two components. First, calling `tf.op(..., name='operation_name')` would indeed lead to the construction of a new operation. The operation is also added to the computational graph. Second, though this looks like a constructor, it is just a function call with side-effect, and the return value of this function is a tensor. Again, this is documented well, but you may easily skim through.

Thus, to get the output of an operation, you will have to pass the output tensor to `session.run`, not the operator itself. For instance, use `session.run(op.outputs, feed_dict=...)`.
