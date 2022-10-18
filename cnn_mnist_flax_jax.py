import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import numpy as np
import optax
import tensorflow_datasets as tfds
import wandb
from hyper_params import *
from rich.pretty import pprint
from tqdm import tqdm
import streamlit as st


class CNN(nn.Module):
    """
    CNN class with
    3 Conv2D layers
    4 relu activation layers
    2 max pool and 1 avg pool layer
    512 Dense units (hidden)
    10 output layer
    """

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(32, kernel_size=(3, 3), )(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(128, kernel_size=(3, 3), )(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(512)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        return x


# returns the cross entropy loss across all classes onehot encoded uses optax
def cross_entropy_loss(*, logits, labels):
    labels_onehot = jax.nn.one_hot(labels, num_classes=10)
    return optax.softmax_cross_entropy(logits=logits, labels=labels_onehot).mean()


# computes loss and accuracy metrics
def compute_metrics(logits, labels):
    loss = cross_entropy_loss(logits=logits, labels=labels)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    return loss, accuracy


# downloads and prepares the dataset using tfds
def datasets(name):
    ds_builder = tfds.builder(name)
    ds_builder.download_and_prepare()
    train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
    train_ds['image'] = jnp.float32(train_ds['image']) / 255.
    test_ds['image'] = jnp.float32(test_ds['image']) / 255.
    return train_ds, test_ds


def create_train_state(random_number_gen, lr, m):
    cnn = CNN()
    # initializes the cnn weights based on the dimension of jp.ones and random_number_generator
    weight_params = cnn.init(random_number_gen, jnp.ones([1, 28, 28, 1]))
    # optimizer takes the next step
    optim = optax.sgd(lr, m)
    # applies the optimizer to the cnn's weight params
    return train_state.TrainState.create(
        apply_fn=cnn.apply, params=weight_params['params'], tx=optim
    )


@jax.jit
def train_step(current_state, img_batch):
    def loss_fn(params):
        # calculates the loss after one forward pass with params of the model and the batch image data
        logits = CNN().apply({'params': params}, img_batch['image'])
        loss = cross_entropy_loss(logits=logits, labels=img_batch['label'])
        return loss, logits

    # calculates the gradient from the loss obtained and returns the new loss
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_gits, logits), grads = grad_fn(current_state.params)
    current_state = current_state.apply_gradients(grads=grads)
    loss, accuracy = compute_metrics(logits=logits, labels=img_batch['label'])
    metrics = {
        'loss': loss,
        'accuracy': accuracy
    }
    return current_state, metrics


# evaluates the loss across each step in the epoch
@jax.jit
def eval_step(params, batch):
    logits = CNN().apply({'params': params}, batch['image'])
    loss, acc = compute_metrics(logits=logits, labels=batch['label'])
    return {'loss': loss, 'acc': acc}


# brings all the individual pieces together
def train_epoch(state, train_ds, batch_size, epoch, rng):
    train_ds_size = train_ds['image'].shape[0]
    sps = train_ds_size // batch_size
    # picks random data sample using permutation
    perms = jax.random.permutation(rng, train_ds_size)
    perms = perms[:sps * batch_size]  # skip incomplete batch
    perms = perms.reshape((sps, batch_size))
    batch_metrics = []
    for perm in perms:
        batch = {k: v[perm, ...] for k, v in train_ds.items()}
        state, metrics = train_step(state, batch)
        # logs the metrics to weights&bias log board
        wandb.log({
            'training_loss': metrics['loss'],
            "training_acc": metrics['accuracy']
        })
        batch_metrics.append(metrics)
    batch_metrics_np = jax.device_get(batch_metrics)
    epoch_metrics_np = {
        k: np.mean([metrics[k] for metrics in batch_metrics_np])
        for k in batch_metrics_np[0]}
    st.text('train epoch: %d, loss: %.4f, accuracy: %.2f' % (
        epoch, epoch_metrics_np['loss'], epoch_metrics_np['accuracy'] * 100))
    print('train epoch: %d, loss: %.4f, accuracy: %.2f' % (
        epoch, epoch_metrics_np['loss'], epoch_metrics_np['accuracy'] * 100))

    return state


# evaluates the model's performance
def eval_model(params, test_ds):
    metrics = eval_step(params, test_ds)
    metrics = jax.device_get(metrics)
    summary = jax.tree_util.tree_map(lambda x: x.item(), metrics)
    return summary['loss'], summary['acc']


def main():
    st.info('Loading MNIST dataset')
    train_ds, test_ds = datasets('mnist')
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)
    st.info('Creating training state')
    state = create_train_state(init_rng, learning_rate, momentum)
    st.write(state)
    wandb.config = {
        "learning_rate": learning_rate,
        "epochs": num_epochs,
        "batch_size": batch_size,
    }

    pprint(wandb.config)
    st.text(wandb.config)

    progress_bar = st.progress(0)
    for index, epoch in tqdm(enumerate(range(1, num_epochs + 1))):
        st.info('Starting Training Now! If on CPU, wait for a long time')
        st.info('Progress bar will show the training status along with verbose')
        rng, input_rng = jax.random.split(rng)
        state = train_epoch(state, train_ds, batch_size, epoch, input_rng)
        test_loss, test_accuracy = eval_model(state.params, test_ds)
        st.text(' test epoch: %d, loss: %.2f, accuracy: %.2f' % (
            epoch, test_loss, test_accuracy * 100))

        pprint(' test epoch: %d, loss: %.2f, accuracy: %.2f' % (
            epoch, test_loss, test_accuracy * 100))
        wandb.log({
            'test_loss': test_loss,
            'test_accuracy': test_accuracy
        })
        progress_bar.progress(index)


if __name__ == '__main__':
    main()
