from  sklearn import metrics
from lifelines.utils import concordance_index


def calculate_clustering_metrics(labels_true, labels_pred):
    metrics_to_evaluate_1 = {
        'c_index': concordance_index,
        'homogeneity': metrics.homogeneity_score,
        'completeness': metrics.completeness_score,
        'v_measure': metrics.v_measure_score,
        'normalized_mutual_info': metrics.normalized_mutual_info_score,
    }

    metrics_to_evaluate_2 = {
        'silhouette': metrics.silhouette_score,
        'davies_bouldin': metrics.davies_bouldin_score,
    }

    p1 = {
        metric_name: metric(labels_true, labels_pred) for
        metric_name, metric in
        metrics_to_evaluate_1.items()
    }
    p2 = {
        metric_name: metric(labels_true.reshape((-1, 1)), labels_pred.reshape((-1, 1))) for
        metric_name, metric in
        metrics_to_evaluate_2.items()
    }

    p1.update(p2)

    return p1