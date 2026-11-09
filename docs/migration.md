# Changes from the 2017 scripts

Original revision `806bcb7` and all three scripts are preserved byte-for-byte under `legacy/`. The maintained implementation uses current native APIs and a simpler explicit baseline. It does not emulate removed scikit-learn arguments or claim identical historical weights.

| Original choice | Maintained choice and rationale |
|---|---|
| Feature lists created through sets | Ordered schema retained and checked at every prediction boundary |
| Label vocabulary fits train plus test | Fold-trained one-hot encoding with defined unknown-category handling |
| Per-fold decompositions/embedding/clustering | Preserve the training-only fitting boundary; start with simple models and one optional appended-SVD comparison |
| Training median encodings include each row's own target | Omit target encodings from this baseline; a cross-fitted version needs a separate assessed design |
| 10,000 blend weights scored on the same OOF labels | Fixed equal mean declared before evaluation; no optimized-score claim from reused labels |
| Removed `mse`, `auto`, `normalize` API arguments and DataFrame append | Current supported estimators/Pandas interfaces, recorded by a dependency lock |
| CSV outputs without a complete source/model contract | Saved fold pipelines, input/config/schema/ID hashes, reconstruction and recoverable delivery |

The original decomposition, embedding and clustering fits already occur inside outer folds. The vocabulary and target-encoding/selection limitations are distinct; calling all legacy preprocessing “validation leakage” would misrepresent the code.

Random record-level folds assess this competition population. Repeated anonymized feature vectors are not proof of the same physical vehicle, so they are retained rather than automatically grouped or removed. Future-manufacturing or unseen-configuration claims require a separately defined population.
