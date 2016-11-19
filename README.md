TensorFlow Best Practices
------------------------------

This is a collection of gotchas collected by AI.codes engineers while working with TensorFlow. It is, without doubt, opinionated.

#### TL,DR:
* Represent the model as a Python class, with `loss`, `inference` and `optimize` member functions.
* Use checkpoints in training.
* Use summaries.
* Use flags to represent hyper-parameters.
* Name important operations, with a standard vocabulary

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

Unless dealing with toy examples, we rarely just train a real-world model in one-shot. It may take days to train one. Model trained on a set of data may later gets trained further on a different (larger) set of data, or with a different set of hyper-parameters. In all these cases, treating training as an ongoing process is more rewarding than treating it as a one-shot process. This means you should include `saver.restore` and `saver.save` from the get go. It will help you immensely down the road. The `Saver` API doc is [here](https://www.tensorflow.org/versions/r0.11/api_docs/python/state_ops.html#Saver). To store a computation graph, you use:
```python
saver = tf.train.Saver()
saver.save(sess, checkpoints_file_name)
```

To restore a graph, use
```python
saver = tf.train.import_meta_graph(checkpoints_file_name + '.meta')
saver.restore(sess, checkpoints_file_name)
```

_Gotcha_: When saving/restoring variables, make sure that `Saver` class is constructed **after** all variables are defined. Saver only captures all variables at the time it is constructed, so any new variables defined after `Saver()` will not be saved/restored automatically.

Because of that, when you restore from a previously checkpoint, you will have to construct the saver from meta graph, not via `saver = tf.train.Saver()` (unless you define again all the variables). What we found is that sometimes it is hard to separate variable definition from model definition (especially when you use predefined modules like LSTM where variable definitions are embedded in these units). As a result, restoring variables from meta graph is much easier than restoring them by defining them again.

#### Use summaries and Tensorboard from day one

Like checkpoints, summaries are fantastic ways to let you get a snapshot of the model during the training process. Summary is extremely easy to use. Think of summary as a glorified `printf(tensor)`, where you can visualize the output later. Tensorflow provides scalar, image, audio, and histogram summaries. To use summary, just pass the tensor to a summary operation:
```python
# 1. Declare summaries that you'd like to collect.
tf.scalar_summary("summary_name", tensor, name = "summary_op_name")

# 2. Construct a summary writer object for the computation graph, once all summaries are defined.
summary_writer = tf.train.SummaryWriter(summary_dir_name, sess.graph)

# 3. Group all previously declared summaries for serialization. Usually we want all summaries defined
# in the computation graph. To pick a subset, use tf.merge_summary([summaries]).
summaries_tensor = tf.merge_all_summaries()

# 4. At runtime, in appropriate places, evaluate the summaries_tensor, to assign value.
summary_value, loss, ... = sess.run([summaries_tensor, loss, ...], feed_dict={...})

# 5. Write the summary value to disk, using summary writer.
summary_writer.add_summary(summary, global_step=step)
```

_Gotcha_: `summary_name` is the metric name in visualization. When restoring a computation graph, all summary operations are restored. However, summary writer and the merged tensor are not. Thus, you will need to do step 2 and 3 again, after restoring the graph from disk.

TensorBoard not only helps visualize the computation graph, it also gives you a good idea of learning rate if you put the loss value in summary. You can certainly just print loss values to stdout during training, but we human are not good at visualizing the shape of the loss curve just by looking at a series of numbers. When the curve is plotted in TensorBoard, you will immediately see the trend and know if learning rate is too low or too high.

At AI.codes, we make it **mandatory** to export training `loss` and validation `error` to summaries. It also helps us to justify if a model is appropriately trained, over-trained, or can be further improved.

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

When we first define the model in Python, operations such as `optimize` and `inference` are referenced by variables, i.e. `opt = model.optimize(...)` where we can later use `opt` in the code. When the computational graph is restored from serialized GraphDef and checkpoint files, however, we no longer have a Python reference to the operation. We need a way to get the correct operation from the graph.

We recommend naming operations that you may need later, so when the graph is restored, you can retrieve these operations by their names, using `graph.get_operation_by_name(name)`. A good practice is to standardize the names from an agreed vocabulary. For instance, if we agree upon that _loss_ is always used as the name for loss function in models, in MNIST example, we'd use

```python
loss_tensor = tf.nn.softmax_cross_entropy_with_logits(logits, labels, dim=-1, name="loss")
```
to label the loss operation. We will no longer have `loss_tensor` when we restore the model, but we can always call `graph.get_operation_by_name("loss")` to get the operation.

_Gotcha_: Shared vocabulary enables efficient communication between engineers. Here is a set of words in our vocabulary, independent of the models defined.

* x (x is the name of the placeholder. As mentioned, placeholder is an operation)
* y (in supervised case)
* loss
* inference
* learning_rate
* optimizer

Engineers can of course name additional operations. The ones we defined here are fairly generic. We put the definition in a common file called `vocab.py` that can be imported by others. We use Python's `namedtuple` to get a bit compile-time check, instead of using raw strings directly. The code seems cryptic, but really it allows us to use references `vocab.x` or `vocab.inference` as names, as oppose to the more error-prone ones like `vocab['inference']` or `vocab['loss']`.

```python
from collections import namedtuple

Vocab = namedtuple('Vocab', ['x', 'y', 'loss', 'inference', 'learning_rate', 'optimizer'])
vocab = Vocab('x', 'y', 'loss', 'inference', 'learning_rate', 'optimizer')
```

_Gotcha_: **Passing values to placeholders in restored models**

We know it well that we can use `feed_dict` to pass values to placeholders, when we have a references to the placeholders. For instance, if we have `x = tf.placeholder(...)`, we can later use `session.run(some_tensor, feed_dict={x: value})` to pass value to the computational graph. What is less obvious is when you restore the model, where you do not have a direct reference to `x` anymore.

If you correctly name your placeholder operation, as in `x = tf.placeholder(..., name=vocab.x)`, you can retrieve the operation later by using `graph.get_operation_by_name(vocab.x)`. A caveat here is that you are getting the **operation**, not the **tensor**. To get the actual tensor that can be used as the key in feed_dict, you can either take the ugly route, using `graph.get_tensor_by_name(vocab.x + ':0')`, or, a better way, `graph.get_operation_by_name(vocab.x).outputs[0]`. The ugly way works because tensors, as outputs of operations, are by default named as `operation_name:N` where `N` is the nth output. Placeholder tensor only has one output, thus the `:0` part.

_Gotcha_: **Evaluate operations in restored models**

`Session.run()` can take both operations and tensors. If you pass in operations, however, the returned value would be `None`. This is [documented](https://www.tensorflow.org/versions/r0.11/api_docs/python/client.html#Session) but you may easily skim through the document and get puzzled by the fact that you get nothing back. This actually reveal some peculiarities of TensorFlow's Python API that we will explain here.

Take a look at the statement `a = tf.placeholder(...)`. It looks like a constructor call, isn't it? You'd think that `a` must be of `placeholder` type, whatever that type is. Now let's look at `a = tf.matmul(B, C)`. Well, this time you may say that `a` should be a tensor, as the result of `B * C`. The truth is, `a` is tensor in both cases, except that the first case is a bit misleading in light of the Python coding convention.

The way to understand statements like `a = tf.op(..., name='operation_name')` is to break it down to two components. First, calling `tf.op(..., name='operation_name')` would indeed lead to the construction of a new operation. The operation is also added to the computational graph. Second, though this looks like a constructor, it is just a function call with side-effect, and the return value of this function is a tensor. Again, this is documented well, but you may easily skim through.

Thus, to get the output of an operation, you will have to pass the output tensor to `session.run`, not the operator itself. For instance, use `session.run(op.outputs, feed_dict=...)`.
