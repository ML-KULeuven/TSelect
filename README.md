# TSelect

## Installation
### Pip install
```
pip install tselect
```

### Clone repository
(Temporary until published)
Clone the repository.
```
git clone https://github.com/LorenNuyts/TSelect.git
```
Go to the newly created repository and install the requirements with pip.
```
pip install -r requirements.txt
```

### Known issues
On Windows, the installation of the pycatch22 package can fail. Installing the package with the following command
usually fixes this.
```
pip install pycatch22==0.4.2 --use-deprecated=legacy-resolver
```

## Examples
### Feature extraction with TSFuse
TSFuse requires the data to be in a specific format, see [TSFuse](https://github.com/arnedb/tsfuse#data-format) for more information.
```python
from tsfuse.construction.mlj20 import TSFuseExtractor
from tselect.channel_selectors.tselect import TSelect

[...] # load data, split in train and test set, etc.

extractor = TSFuseExtractor(transformers=tsfuse_transformers, compatible=compatible, random_state=SEED,
                                    series_filter=TSelect())
extractor.fit(x_train, y_train)
features_train = extractor.transform(x_train)
features_test = extractor.transform(x_test)

[...] # Training a model, etc.
```

### Feature extraction with MiniRocket
```python
from tselect.minirocket import MiniRocketExtractor
from tselect.channel_selectors.tselect import TSelect

[...] # load data, split in train and test set, etc.

# If fusion is used, the `views`, `add_tags`, and `compatible` arguments must also be specified for correct data transformation.
# More information on these arguments can be found on [TSFuse](https://github.com/arnedb/tsfuse#data-format).
extractor = MiniRocketExtractor(series_fusion=True, irrelevant_filter=True, redundant_filter=True)
extractor.fit(x_train, y_train)
features_train = extractor.transform(x_train)
features_test = extractor.transform(x_test)

[...] # Training a model, etc.
```

### Feature extraction with Catch22
```python
from tselect.catch22 import Catch22Extractor
from tselect.channel_selectors.TSelect import TSelect

[...] # load data, split in train and test set, etc.

# If fusion is used, the `views`, `add_tags`, and `compatible` arguments must also be specified for correct data transformation.
# More information on these arguments can be found on [TSFuse](https://github.com/arnedb/tsfuse#data-format).
extractor = Catch22Extractor(series_fusion=True, irrelevant_filter=True, redundant_filter=True)
extractor.fit(x_train, y_train)
features_train = extractor.transform(x_train)
features_test = extractor.transform(x_test)

[...] # Training a model, etc.
```

### Fusion and filtering only
If you only want to use the fusion and/or filtering components of this package, you can use the `FusionFilter` class directly.
```python
from tselect import FusionFilter

[...] # load data, split in train and test set, etc.

# Default arguments, please specify this for the used data
views = None
add_tags = id
compatible = lambda x: True

fusionfilter = FusionFilter(views=views, add_tags=add_tags, compatible=compatible)
x_pd_train = fusionfilter.fit(x_train, y, return_format='dataframe')
x_pd_test = fusionfilter.transform(x_test, return_format='dataframe')

[...] # Further processing of the fused and selected signals.

```
