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

import metrax
from metrax.nnx import nnx_wrapper

NnxWrapper = nnx_wrapper.NnxWrapper


class AUCPR(NnxWrapper):
  """An NNX class for the Metrax metric AUCPR."""

  def __init__(self):
    super().__init__(metrax.AUCPR)


class AUCROC(NnxWrapper):
  """An NNX class for the Metrax metric AUCROC."""

  def __init__(self):
    super().__init__(metrax.AUCROC)


class Average(NnxWrapper):
  """An NNX class for the Metrax metric Average."""

  def __init__(self):
    super().__init__(metrax.Average)


class AveragePrecisionAtK(NnxWrapper):
  """An NNX class for the Metrax metric AveragePrecisionAtK."""

  def __init__(self):
    super().__init__(metrax.AveragePrecisionAtK)


class MSE(NnxWrapper):
  """An NNX class for the Metrax metric MSE."""

  def __init__(self):
    super().__init__(metrax.MSE)


class Perplexity(NnxWrapper):
  """An NNX class for the Metrax metric Perplexity."""

  def __init__(self):
    super().__init__(metrax.Perplexity)


class Precision(NnxWrapper):
  """An NNX class for the Metrax metric Precision."""

  def __init__(self):
    super().__init__(metrax.Precision)


class Recall(NnxWrapper):
  """An NNX class for the Metrax metric Recall."""

  def __init__(self):
    super().__init__(metrax.Recall)


class RMSE(NnxWrapper):
  """An NNX class for the Metrax metric RMSE."""

  def __init__(self):
    super().__init__(metrax.RMSE)


class RSQUARED(NnxWrapper):
  """An NNX class for the Metrax metric RSQUARED."""

  def __init__(self):
    super().__init__(metrax.RSQUARED)


class WER(NnxWrapper):
  """An NNX class for the Metrax metric WER."""

  def __init__(self):
    super().__init__(metrax.WER)
