# TSFilter

## Installation

## Examples
### Feature extraction with TSFuse
TSFuse requires the data to be in a specific format, see [TSFuse](https://github.com/arnedb/tsfuse#data-format) for more information.
```python
from tsfuse.construction.mlj20 import TSFuseExtractor
from tsfilter.filters.tsfilter import TSFilter

[...] # load data, split in train and test set, etc.

extractor = TSFuseExtractor(transformers=tsfuse_transformers, compatible=compatible, random_state=SEED,
                                    series_filter=TSFilter())
extractor.fit(x_train, y_train)
features_train = extractor.transform(x_train)
features_test = extractor.transform(x_test)

[...] # Training a model, etc.
```

### Feature extraction with MiniRocket

```python
tsfilter.minirocket import MiniRocketExtractor
from tsfilter.filters.tsfilter import TSFilter

[...] # load data, split in train and test set, etc.

# If fusion is used, the `views`, `add_tags`, and `compatible` arguments must also be specified for correct data transformation. More information on these arguments can be found on [TSFuse](https://github.com/arnedb/tsfuse#data-format).
extractor = MiniRocketExtractor(series_fusion=True, irrelevant_filter=True, redundant_filter=True)
extractor.fit(x_train, y_train)
features_train = extractor.transform(x_train)
features_test = extractor.transform(x_test)

[...] # Training a model, etc.
```
