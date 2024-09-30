# The random seed variable used throughout the package.
SEED = 0


class Keys:
    """
    A class containing strings that are used as dictionary keys throughout the package.
    """
    dataset = 'dataset'
    task_regression = 'regression'
    task_classification = 'classification'
    features = "features"
    splits = "splits"
    y = "y"

    extractor = 'extractor'
    fusion_filter = 'FusionFilter'

    construction = "construction"
    time_series_to_series = "time_series_to_series"
    time_series_filtering = "time_series_filtering"
    time_series_to_attr = "time_series_to_attributes"
    time_attr_to_attr = "time_attributes_to_attributes"
    time_select = "time_select"
    time_construct = 'time_construct'
    fused_series = "fused_series"
    extracted_attr = "extracted_attributes"
    fused_attr = "fused_attributes"
    deleted_attr = "deleted_attributes"
    remaining_attr = "remaining_attributes"
    time_compute = "time_compute"
    time_train = "time_train"
    time_predict = "time_predict"

    average_time_series_to_series = "average_time_series_to_series"
    average_time_series_filtering = "average_time_series_filtering"
    average_time_series_to_attr = "average_time_series_to_attributes"
    average_time_attr_to_attr = "average_time_attributes_to_attributes"
    average_time_select = "average_time_select"
    average_construction_time = "average_construction_time"
    average_fused_series = "average_fused_series"
    average_extracted_attr = "average_extracted_attributes"
    average_fused_attr = "average_fused_attributes"
    average_deleted_attr = "average_deleted_attributes"
    average_remaining_attr = "average_remaining_attributes"

    series_filtering = "series_filtering"
    rank_correlation = 'rank_correlation'
    average_rank_correlation = "average_rank_correlation"
    average_correlation_target_variables = "average_correlation_target_variables"
    average_pvalue_target_variables = "average_pvalue_target_variables"
    average_correlation_all = "average_correlation_all"
    max_correlation_all = "max_correlation_all"
    min_correlation_all = "min_correlation_all"
    average_pvalue_all = "average_pvalue_all"
    max_pvalue_all = "max_pvalue_all"
    min_pvalue_all = "min_pvalue_all"
    removed_series_auc = "removed_series_auc"
    removed_series_corr = "removed_series_corr"
    series_filter = "series_filter"
    num_series_per_cluster = "num_series_per_cluster"
    average_filtered_series = "average_num_folds_series_chosen"
    average_num_selected_channels = "average_num_clusters/num_selected_channels"
    median_series_per_cluster = "median_series_per_cluster"
    max_series_per_cluster = "max_series_per_cluster"
    min_series_per_cluster = "min_series_per_cluster"

    computation_all = "computation_all"
    average_computation_time = "average_time_compute"
    std_computation_time = "std_time_compute"
    total_time = "total_time"
    std_total_time = "std_total_time"

    average_acc_score = "average_accuracy_score"
    average_auc_score = "average_roc_auc_score"
    average_ies_score = "average_ies_score"
    average_time_train = "average_time_train"
    average_time_predict = "average_time_predict"
    acc_score = 'accuracy_score'
    auc_score = 'roc_auc_score'
    ies_score = 'ies_score'
    std_acc_score = "std_accuracy_score"
    std_auc_score = "std_roc_auc_score"
    std_ies_score = "std_ies_score"
    std_time_train = "std_time_train"
    std_time_predict = "std_time_predict"
    std_time_compute = "std_time_compute"
    accuracy_score_train = 'accuracy_score_train'
    roc_auc_train = 'roc_auc_score_train'
    average_accuracy_score_train = 'average_accuracy_score_train'
    average_roc_auc_train = 'average_roc_auc_score_train'

    xgboost_default = 'xgboost_default'
    ensemble = 'ensemble'
    lr_default = 'lr_default'
    hc_model = 'hc_model'

    cv5x2 = '5x2cv'

    scaler = 'scaler'
    statistics = 'statistics'
    minirocket = 'minirocket'

    average_total_time = "average_total_time"
