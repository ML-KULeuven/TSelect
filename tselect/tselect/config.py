from tselect.utils.constants import SEED


class Config:
    """ Class to track the configuration of the TSelect channel selector.

        Args:
            irrelevant_filter: bool, default=True
            Whether to filter out irrelevant series based on their AUC score
        redundant_filter: bool, default=True
            Whether to filter out redundant series based on their rank correlation
        random_state: int, default=SEED
            The random state used throughout the class.
        filtering_threshold_auc: float, default=0.5
            The threshold to use for filtering out irrelevant series based on their AUC score. All signals below this
            threshold are removed.
        auc_percentage: float, default=0.6
            The percentage of series to keep based on their AUC score. This parameter is only used if
            irrelevant_filter=True. If auc_percentage=0.6, the 60% series with the highest AUC score are kept.
        filtering_threshold_corr: float, default=0.7
             The threshold used for clustering rank correlations. All predictions with a rank correlation above this
             threshold are considered correlated.
        irrelevant_better_than_random: bool, default=False
            Whether to filter out irrelevant series by comparing them with a model trained on a randomly shuffled
            target. If the channel performs Better Than Random (BTR), it is kept.
        filtering_test_size: float, default=None
            The test size to use for filtering out irrelevant series based on their AUC score. The test size is the
            percentage of the data that is used for computing the AUC score. The remaining data is used for training.
            If None, the train size is derived from max(100, 0.25*nb_instances). The test size are then the remaining
            instances.

        """

    def __init__(self,
                 irrelevant_filter=True,
                 redundant_filter=True,
                 random_state: int = SEED,
                 filtering_threshold_auc: float = 0.5,
                 auc_percentage: float = 0.6,
                 filtering_threshold_corr: float = 0.7,
                 feature_extractor = None,
                 hierarchical_clustering = False,
                 spectral_clustering = False,
                 n_clusters_spectral_clustering = None,
                 multiple_model_weighing: bool = False,
                 irrelevant_better_than_random: bool = False,
                 filtering_test_size: float = None,
                 print_times: bool = False):
        self.irrelevant_filter = irrelevant_filter
        self.redundant_filter = redundant_filter
        self.random_state = random_state
        self.filtering_threshold_auc = filtering_threshold_auc
        self.filtering_threshold_corr = filtering_threshold_corr
        self.test_size = filtering_test_size
        self.auc_percentage = auc_percentage
        self.multiple_models_weighing = multiple_model_weighing
        self.irrelevant_better_than_random = irrelevant_better_than_random
        self.print_times = print_times
        self.feature_extractor = feature_extractor
        self.hierarchical_clustering = hierarchical_clustering
        self.spectral_clustering = spectral_clustering
        self.n_clusters_spectral_clustering = n_clusters_spectral_clustering

def get_default_config() -> Config:
    """
    Get the default configuration for the TSelect channel selector.

    Returns
    -------
    config : Config
        The default configuration for the TSelect channel selector.
    """
    return Config()