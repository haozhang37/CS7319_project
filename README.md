# CS7319_project
This is the project code for CS 7319



## Experiment Result


### 3 reconstruction in few-shot learning scenarios
we sample a small train epoch from MNIST/F-MNIST, each epoch contain `num_batch` batches,
 `n_per_batch` samples per batch, `way` class per batch. We create the following 2 classic few-shot learning scenarios.

- **5-way, 1-shot**: 
 `num_batch=100`, `n_per_batch=1`, `way=5`

- **5-way, 5-shot**: 
 `num_batch=100`, `n_per_batch=1`, `way=5`
 
 other training parameters
 `epoch=10`
 #### training loss
 
 |scenarios     | AE   | Lmser(DPN) | Lmser(DCW) |
 | ---          | ---  | -----      | ---- |
 | 5-way 1-shot | 0.60 |    0.44    |      |
 | 5-way 5-shot | 0.57 |    0.41    |      |



besides training loss, we found that Lmser(with DPN) converges much faster than traditional AutoDecoder