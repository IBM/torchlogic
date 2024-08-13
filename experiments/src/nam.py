r"""Training script for Neural Additive Models.

"""
import os
import os.path as osp
import numpy as np
from absl import flags
import tensorflow.compat.v1 as tf

from uuid import uuid4


from .neural_additive_models.nam_train import (_create_computation_graph, _get_train_and_lr_decay_ops,
                                               _create_graph_saver, _update_metrics_and_checkpoints)
from .neural_additive_models import graph_builder
from .neural_additive_models import data_utils
from .neural_additive_models.graph_builder import create_nam_model

gfile = tf.io.gfile
DatasetType = data_utils.DatasetType


class NAM(object):

    def __init__(
            self,
            training_epochs,
            learning_rate,
            output_regularization,
            l2_regularization,
            batch_size,
            decay_rate,
            dropout,
            tf_seed,
            feature_dropout,
            num_basis_functions,
            units_multiplier,
            shallow,
            early_stopping_epochs
    ):
        self.FLAGS = dict()

        self.FLAGS['training_epochs'] = training_epochs
        self.FLAGS['learning_rate'] = learning_rate
        self.FLAGS['output_regularization'] = output_regularization
        self.FLAGS['l2_regularization'] = l2_regularization
        self.FLAGS['batch_size'] = batch_size
        self.FLAGS['logdir'] = f"./logs/{uuid4()}/"
        self.FLAGS['dataset_name'] = "data"
        self.FLAGS['decay_rate'] = decay_rate
        self.FLAGS['dropout'] = dropout
        self.FLAGS['data_split'] = 1
        self.FLAGS['tf_seed'] = tf_seed
        self.FLAGS['feature_dropout'] = feature_dropout
        self.FLAGS['num_basis_functions'] = num_basis_functions
        self.FLAGS['units_multiplier'] = units_multiplier
        self.FLAGS['cross_val'] = False
        self.FLAGS['max_checkpoints_to_keep'] = 1
        self.FLAGS['save_checkpoint_every_n_epochs'] = 10
        self.FLAGS['n_models'] = 1
        self.FLAGS['num_splits'] = 1
        self.FLAGS['fold_num'] = 1
        self.FLAGS['activation'] = 'exu'
        self.FLAGS['regression'] = False
        self.FLAGS['debug'] = False
        self.FLAGS['shallow'] = shallow
        self.FLAGS['use_dnn'] = False
        self.FLAGS['early_stopping_epochs'] = early_stopping_epochs

    def training(self, x_train, y_train, x_validation,
                 y_validation,
                 logdir = None):
        """Trains the Neural Additive Model (NAM).

        Args:
          x_train: Training inputs.
          y_train: Training labels.
          x_validation: Validation inputs.
          y_validation: Validation labels.
          logdir: dir to save the checkpoints.

        Returns:
          Best train and validation evaluation metric obtained during NAM training.
        """
        if logdir is None:
            logdir = self.FLAGS['logdir']
        tf.logging.info('Started training with logdir %s', logdir)
        batch_size = min(self.FLAGS["batch_size"], x_train.shape[0])
        num_steps_per_epoch = x_train.shape[0] // batch_size
        # Keep track of the best validation RMSE/AUROC and train AUROC score which
        # corresponds to the best validation metric score.
        if self.FLAGS["regression"]:
            best_train_metric = np.inf * np.ones(self.FLAGS["n_models"])
            best_validation_metric = np.inf * np.ones(self.FLAGS["n_models"])
        else:
            best_train_metric = np.zeros(self.FLAGS["n_models"])
            best_validation_metric = np.zeros(self.FLAGS["n_models"])
            best_test_metric = np.zeros(self.FLAGS["n_models"])
        # Set to a large value to avoid early stopping initially during training
        curr_best_epoch = np.full(self.FLAGS["n_models"], np.inf)
        # Boolean variables to indicate whether the training of a specific model has
        # been early stopped.
        early_stopping = [False] * self.FLAGS["n_models"]
        # Classification: AUROC, Regression : RMSE Score
        metric_name = 'RMSE' if self.FLAGS["regression"] else 'AUROC'
        tf.reset_default_graph()
        with tf.Graph().as_default():
            tf.compat.v1.set_random_seed(self.FLAGS["tf_seed"])
            # Setup your training.
            graph_tensors_and_ops, metric_scores = _create_computation_graph(
                x_train, y_train, x_validation, y_validation, batch_size, self.FLAGS)

            train_ops, lr_decay_ops = _get_train_and_lr_decay_ops(
                graph_tensors_and_ops, early_stopping)
            global_step = tf.train.get_or_create_global_step()
            increment_global_step = tf.assign(global_step, global_step + 1)
            saver_hooks, model_dirs, best_checkpoint_dirs = _create_graph_saver(
                graph_tensors_and_ops, logdir, num_steps_per_epoch, self.FLAGS)
            if self.FLAGS["debug"]:
                summary_writer = tf.summary.FileWriter(os.path.join(logdir, 'tb_log'))

            with tf.train.MonitoredSession(hooks=saver_hooks) as sess:
                for n in range(self.FLAGS["n_models"]):
                    sess.run([
                        graph_tensors_and_ops[n]['iterator_initializer'],
                        graph_tensors_and_ops[n]['running_vars_initializer']
                    ])
                for epoch in range(1, self.FLAGS["training_epochs"] + 1):
                    if not all(early_stopping):
                        for _ in range(num_steps_per_epoch):
                            sess.run(train_ops)  # Train the network
                        # Decay the learning rate by a fixed ratio every epoch
                        sess.run(lr_decay_ops)
                    else:
                        tf.logging.info('All models early stopped at epoch %d', epoch)
                        break

                    for n in range(self.FLAGS["n_models"]):
                        if early_stopping[n]:
                            sess.run(increment_global_step)
                            continue
                        # Log summaries
                        if self.FLAGS["debug"]:
                            global_summary, global_step = sess.run([
                                graph_tensors_and_ops[n]['summary_op'],
                                graph_tensors_and_ops[n]['global_step']
                            ])
                            summary_writer.add_summary(global_summary, global_step)

                        if epoch % self.FLAGS["save_checkpoint_every_n_epochs"] == 0:
                            (curr_best_epoch[n], best_validation_metric[n],
                             best_train_metric[n], best_test_metric[n]) = _update_metrics_and_checkpoints(
                                sess, epoch, metric_scores[n], curr_best_epoch[n],
                                best_validation_metric[n], best_train_metric[n], best_test_metric[n], model_dirs[n],
                                best_checkpoint_dirs[n], metric_name, self.FLAGS)
                            if curr_best_epoch[n] + self.FLAGS["early_stopping_epochs"] < epoch:
                                tf.logging.info('Early stopping at epoch {}'.format(epoch))
                                early_stopping[n] = True  # Set early stopping for model `n`.
                                train_ops, lr_decay_ops = _get_train_and_lr_decay_ops(
                                    graph_tensors_and_ops, early_stopping)
                        # Reset running variable counters
                        sess.run(graph_tensors_and_ops[n]['running_vars_initializer'])

        tf.logging.info('Finished training.')
        for n in range(self.FLAGS["n_models"]):
            tf.logging.info(
                'Model %d: Best Epoch %d, Individual %s: Train %.4f, Validation %.4f',
                n, curr_best_epoch[n], metric_name, best_train_metric[n],
                best_validation_metric[n])

        return np.mean(best_validation_metric)

    def testing(self, x_train, y_train, x_validation,
                 y_validation, x_test, y_test,
                 logdir = None):
        """Trains the Neural Additive Model (NAM).

        Args:
          x_train: Training inputs.
          y_train: Training labels.
          x_validation: Validation inputs.
          y_validation: Validation labels.
          logdir: dir to save the checkpoints.

        Returns:
          Best train and validation evaluation metric obtained during NAM training.
        """
        if logdir is None:
            logdir = self.FLAGS['logdir']
        tf.logging.info('Started training with logdir %s', logdir)
        batch_size = min(self.FLAGS["batch_size"], x_train.shape[0])
        num_steps_per_epoch = x_train.shape[0] // batch_size
        # Keep track of the best validation RMSE/AUROC and train AUROC score which
        # corresponds to the best validation metric score.
        if self.FLAGS["regression"]:
            best_train_metric = np.inf * np.ones(self.FLAGS["n_models"])
            best_validation_metric = np.inf * np.ones(self.FLAGS["n_models"])
        else:
            best_train_metric = np.zeros(self.FLAGS["n_models"])
            best_validation_metric = np.zeros(self.FLAGS["n_models"])
            best_test_metric = np.zeros(self.FLAGS["n_models"])
        # Set to a large value to avoid early stopping initially during training
        curr_best_epoch = np.full(self.FLAGS["n_models"], np.inf)
        # Boolean variables to indicate whether the training of a specific model has
        # been early stopped.
        early_stopping = [False] * self.FLAGS["n_models"]
        # Classification: AUROC, Regression : RMSE Score
        metric_name = 'RMSE' if self.FLAGS["regression"] else 'AUROC'
        tf.reset_default_graph()
        with tf.Graph().as_default():
            tf.compat.v1.set_random_seed(self.FLAGS["tf_seed"])
            # Setup your training.
            graph_tensors_and_ops, metric_scores = _create_computation_graph(
                x_train, y_train, x_validation, y_validation, batch_size, self.FLAGS, x_test, y_test)

            train_ops, lr_decay_ops = _get_train_and_lr_decay_ops(
                graph_tensors_and_ops, early_stopping)
            global_step = tf.train.get_or_create_global_step()
            increment_global_step = tf.assign(global_step, global_step + 1)
            saver_hooks, model_dirs, best_checkpoint_dirs = _create_graph_saver(
                graph_tensors_and_ops, logdir, num_steps_per_epoch, self.FLAGS)
            if self.FLAGS["debug"]:
                summary_writer = tf.summary.FileWriter(os.path.join(logdir, 'tb_log'))

            with tf.train.MonitoredSession(hooks=saver_hooks) as sess:
                for n in range(self.FLAGS["n_models"]):
                    sess.run([
                        graph_tensors_and_ops[n]['iterator_initializer'],
                        graph_tensors_and_ops[n]['running_vars_initializer']
                    ])
                for epoch in range(1, self.FLAGS["training_epochs"] + 1):
                    if not all(early_stopping):
                        for _ in range(num_steps_per_epoch):
                            sess.run(train_ops)  # Train the network
                        # Decay the learning rate by a fixed ratio every epoch
                        sess.run(lr_decay_ops)
                    else:
                        tf.logging.info('All models early stopped at epoch %d', epoch)
                        break

                    for n in range(self.FLAGS["n_models"]):
                        if early_stopping[n]:
                            sess.run(increment_global_step)
                            continue
                        # Log summaries
                        if self.FLAGS["debug"]:
                            global_summary, global_step = sess.run([
                                graph_tensors_and_ops[n]['summary_op'],
                                graph_tensors_and_ops[n]['global_step']
                            ])
                            summary_writer.add_summary(global_summary, global_step)

                        if epoch % self.FLAGS["save_checkpoint_every_n_epochs"] == 0:
                            (curr_best_epoch[n], best_validation_metric[n],
                             best_train_metric[n], best_test_metric[n]) = _update_metrics_and_checkpoints(
                                sess, epoch, metric_scores[n], curr_best_epoch[n],
                                best_validation_metric[n], best_train_metric[n], best_test_metric[n], model_dirs[n],
                                best_checkpoint_dirs[n], metric_name, self.FLAGS)
                            if curr_best_epoch[n] + self.FLAGS["early_stopping_epochs"] < epoch:
                                tf.logging.info('Early stopping at epoch {}'.format(epoch))
                                early_stopping[n] = True  # Set early stopping for model `n`.
                                train_ops, lr_decay_ops = _get_train_and_lr_decay_ops(
                                    graph_tensors_and_ops, early_stopping)
                        # Reset running variable counters
                        sess.run(graph_tensors_and_ops[n]['running_vars_initializer'])

        tf.logging.info('Finished training.')
        for n in range(self.FLAGS["n_models"]):
            tf.logging.info(
                'Model %d: Best Epoch %d, Individual %s: Train %.4f, Validation %.4f',
                n, curr_best_epoch[n], metric_name, best_train_metric[n],
                best_validation_metric[n])

        return np.mean(best_test_metric)
