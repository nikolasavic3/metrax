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

from metrax import base
from metrax import classification_metrics
from metrax import nlp_metrics
from metrax import ranking_metrics
from metrax import regression_metrics

Average = base.Average
AUCPR = classification_metrics.AUCPR
AUCROC = classification_metrics.AUCROC
Precision = classification_metrics.Precision
Recall = classification_metrics.Recall
Perplexity = nlp_metrics.Perplexity
WER = nlp_metrics.WER
AveragePrecisionAtK = ranking_metrics.AveragePrecisionAtK
MSE = regression_metrics.MSE
RMSE = regression_metrics.RMSE
RSQUARED = regression_metrics.RSQUARED
