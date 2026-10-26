# Full-softmax matrix factorization

`MFSoftmax` has the same dot-product user/item embeddings, Xavier initialization,
prediction, triple loader and optimizer as `BPR`. It changes the data loss to
cross entropy against every real catalog item, with temperature one. Other
training positives remain in the denominator; the padding row does not.

The sampled negative is used only in the existing batch-mean `EmbLoss` penalty:
`reg_weight / (2 * batch_size)` times the summed squared user, positive and
negative embedding norms. This deliberately preserves BPR's regularizer; it is
not global L2 over the entire catalog. Configure it with BPR's same parameters
under the model name `MFSoftmax`. There is no score normalization, in-batch
candidate set, log-frequency correction or new optimizer weight decay.

The logits use memory proportional to batch size times catalog size. This
variant is intended for catalogs where full softmax fits; sampled softmax is a
different objective and is not silently substituted. Tests independently check
the log-sum-exp value, padding gradient, negative-only regularization, registry
configuration and checkpoint round trip. This is a feature, not a BPR bug fix.
