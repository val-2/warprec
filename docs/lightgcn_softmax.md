# LightGCN with full-catalog softmax

`LightGCNSoftmax` inherits the graph, symmetric degree normalization, layer
average, initial embeddings, optimizer and prediction of `LightGCN`. It replaces
the BPR data term with mean cross entropy over all real catalog items. Training
positives remain in the denominator, padding is excluded, temperature is one,
and there is no score normalization, logQ correction or negative subsampling.
The sampled negative is used only by the unchanged base-embedding `EmbLoss`.
Configuration parameters are identical to `LightGCN`.

The public `data_loss(users, items, positives)` accepts propagated batch user
embeddings and all item embeddings. It computes the data term without another
graph propagation, regularization or optimization step. A caller can therefore
accumulate a full epoch's gradient in chunks, then propagate that gradient to
the base embeddings and perform one optimizer update. This preserves a
full-batch experimental schedule; ordinary `training_step` remains minibatch.
The full-catalog logits require memory proportional to chunk size times catalog
size; the function does not silently substitute sampled softmax.

Tests check the independent log-sum-exp formula, padding exclusion, encoder and
L2 parity, equivalence of chunked and unchunked epoch gradients, and registry
and checkpoint round trip. This is a feature, not a correction to LightGCN.
