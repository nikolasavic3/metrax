# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A collection of different metrics for NLP models."""

from clu import metrics as clu_metrics
import flax
import jax
import jax.numpy as jnp


@flax.struct.dataclass
class Perplexity(clu_metrics.Metric):
  r"""Computes perplexity for sequence generation.

  Perplexity is a measurement of how well a probability distribution predicts a
  sample. It is defined as the exponentiation of the cross-entropy. A low
  perplexity indicates the probability distribution is good at predicting the
  sample.

  For language models, it can be interpreted as the weighted average branching
  factor of the model - how many equally likely words can be selected at each
  step.

  Given a sequence of :math:`N` tokens, perplexity is calculated as:

  .. math::
      Perplexity = \exp\left(-\frac{1}{N}\sum_{i=1}^{N} \log P(x_i|x_{<i})\right)

  When sample weights :math:`w_i` are provided:

  .. math::
      Perplexity = \exp\left(-\frac{\sum_{i=1}^{N} w_i\log P(x_i|x_{<i})}{\sum_{i=1}^{N} w_i}\right)

  where:
      - :math:`P(x_i|x_{<i})` is the predicted probability of token :math:`x_i`
        given previous tokens
      - :math:`w_i` are sample weights
      - :math:`N` is the sequence length

  Lower perplexity indicates better prediction - the model is less "perplexed" by the data.
  """

  aggregate_crossentropy: jax.Array
  num_samples: jax.Array

  @classmethod
  def empty(cls) -> 'Perplexity':
    return cls(
      aggregate_crossentropy=jnp.array(0, jnp.float32),
      num_samples=jnp.array(0, jnp.float32))

  @classmethod
  def from_model_output(
      cls,
      predictions: jax.Array,
      labels: jax.Array,
      sample_weights: jax.Array | None = None,
  ) -> 'Perplexity':
    """Updates the metric.

    Args:
      predictions: A floating point tensor representing the prediction
      generated from the model. The shape should be (batch_size, seq_len,
      vocab_size).
      labels: True value. The shape should be (batch_size, seq_len).
      sample_weights: An optional tensor representing the
        weight of each token. The shape should be (batch_size, seq_len).

    Returns:
      Updated Perplexity metric.

    Raises:
      ValueError: If type of `labels` is wrong or the shapes of `predictions`
      and `labels` are incompatible.
    """
    predictions = predictions / jnp.sum(predictions, axis=-1, keepdims=True)
    labels_one_hot = jax.nn.one_hot(labels, predictions.shape[-1], axis=-1)
    log_prob = jnp.log(predictions)
    crossentropy = -jnp.sum(labels_one_hot * log_prob, axis=-1)

    # Sum across sequence length dimension first.
    if sample_weights is not None:
      crossentropy = crossentropy * sample_weights
      # Normalize by the sum of weights for each sequence.
      crossentropy = jnp.sum(crossentropy) / jnp.sum(sample_weights)
    else:
      crossentropy = jnp.mean(crossentropy)

    batch_size = jnp.array(labels.shape[0])
    return cls(
        aggregate_crossentropy=(batch_size * crossentropy),
        num_samples=batch_size,
    )

  def merge(self, other: 'Perplexity') -> 'Perplexity':
    return type(self)(
        aggregate_crossentropy=(
            self.aggregate_crossentropy + other.aggregate_crossentropy
        ),
        num_samples=self.num_samples + other.num_samples,
    )

  def compute(self) -> jax.Array:
    return jnp.exp(self.aggregate_crossentropy / self.num_samples)
