# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Licensed under the MIT license.

from chainer import training


class GradientAccumulationUpdater(training.StandardUpdater):
    """ The optimizer is run once every `n_accs` minibatches.
    The gradients over `n_accs` minibatches are accumulated.
    It virtually enlarges minibatch size.
    """
    def __init__(self, iterator, optimizer, converter, device, n_accs=1):
        super(GradientAccumulationUpdater, self).__init__(
              iterator, optimizer, converter=converter, device=device)
        self.step = 0
        self.n_accs = n_accs

    def update_core(self):
        self.step += 1
        iterator = self.get_iterator('main')
        optimizer = self.get_optimizer('main')
        batch = iterator.next()
        # converter outputs 'dict', 'tuple', or array
        x = self.converter(batch, self.device)
        if self.step == 1:
            optimizer.target.cleargrads()
        # Compute the loss at this time step and accumulate it
        if isinstance(x, tuple):
            loss = optimizer.target(*x) / self.n_accs
        elif isinstance(x, dict):
            loss = optimizer.target(**x) / self.n_accs
        else:
            loss = optimizer.target(x) / self.n_accs
        loss.backward()
        loss.unchain_backward()
        # Update parameters once every n_accs
        if self.step % self.n_accs == 0:
            optimizer.update()
            optimizer.target.cleargrads()
