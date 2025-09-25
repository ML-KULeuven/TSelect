# TSelect

## Installation
### Option 1: Pip install
TSelect can be installed with pip.
```
pip install tselect
```

### Option 2: Clone repository
Alternatively, the repository can be cloned with:
```
git clone https://github.com/ML-KULeuven/TSelect.git
```
Afterward, the requirements should be installed:
```
pip install -r requirements.txt
```

### Known issues
On Windows, the installation of the pycatch22 package can fail. Installing the package with the following command
usually fixes this.
```
pip install pycatch22==0.4.2 --use-deprecated=legacy-resolver
```

## Quick start
TSelect is a package for selecting relevant and non-redundant channels from multivariate time series data (_n_ instances,
_t_ timepoints, _d_ channels). It accepts 
the following data formats as input:
- MultiIndex Pandas DataFrame (with index levels: (_n_, _t_) and _d_ columns)
- 3D NumPy array (with shape: (_n_, _d_, _t_))
- a Dictionary with TSFuse Collection objects (see https://github.com/arnedb/tsfuse for more information)

The general set-up is as follows:
```python
from tselect.channel_selectors.tselect import TSelect

# Load your data, split in train and test set, etc.
x_train, x_test = ... 
y_train, y_test = ...

channel_selector = TSelect(irrelevant_percentage_to_keep=0.6,
                           redundant_correlation_threshold=0.7)
channel_selector.fit(x_train, y_train)
x_train_selected = channel_selector.transform(x_train)
x_test_selected = channel_selector.transform(x_test)

clf = <some MTSC classifier> # Can be any classifier for multivariate time series classification
clf.fit(x_train_selected, y_train)
y_pred = clf.predict(x_test_selected)
```

### Hyperparameters
TSelect has several hyperparameters that can be adapted to the specific dataset and use case.

The hyperparameters to configure the irrelevant channel selector:
- `irrelevant_selector`: bool, default=True
  - Whether to use the irrelevant channel selector.
- `irrelevant_percentage_to_keep`: float, default=0.6
  - The percentage of channels that are expected to be relevant. TSelect will keep this 
  percentage of channels after the irrelevant channel selector step.
  - A value between 0 and 1, where 1 means all channels are kept.
- `irrelevant_hard_threshold`: float, default=0.5
  - All channels with an evaluation metric (e.g. ROCAUC) below this threshold are considered worse than random and are removed, unless
  this would remove all channels.
  
The hyperparameters to configure the redundant channel selector:
- `redundant_selector`: bool, default=True
  - Whether to use the redundant channel selector.
- `redundant_correlation_threshold`: float, default=0.7
  - The correlation threshold to use for the redundant channel selector step. Channels that make predictions with a correlation
  higher than this threshold are considered redundant.
  - A value between 0 and 1, where 1 means that the predictions have to be identical.

Other hyperparameters:
- `validation_size`: float, default=None
  - The size of the validation set used to compute the evaluation metric. If None, the validation size is derived from
  max(100, 0.25*nb_instances). The train set then includes the remaining instances. 
- `random_state`: int, default=0
  - The random state to use for reproducibility.

