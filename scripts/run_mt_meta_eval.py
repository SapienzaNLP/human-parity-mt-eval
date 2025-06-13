import csv
import itertools
import pickle
from argparse import ArgumentParser
from pathlib import Path
from pprint import pp
from typing import List, Dict, Optional, Set, Literal

import scipy
from mt_metrics_eval import data, stats
from mt_metrics_eval.data import ComputeSigMatrix, AssignRanks

NEW_MQM_ANNOTATION_FILE, ESA_1_ANNOTATION_FILE, ESA_2_ANNOTATION_FILE = (
    "en-de.MQM-1.seg.score",
    "en-de.ESA-1.seg.score",
    "en-de.ESA-2.seg.score",
)

METRICS_OUTPUTS_DIR = Path("data/metrics_results/metrics_outputs")
SEG_SCORES_FILENAME = "seg_scores.pickle"


def read_arguments() -> ArgumentParser:
    parser = ArgumentParser(
        description="Command to run MT meta-evaluation with human judgment scores included."
    )

    parser.add_argument(
        "--wmt-year",
        type=str,
        choices=["wmt20", "wmt22", "wmt23", "wmt24"],
        default="wmt20",
        help="Specifies the WMT test set to use for MT meta-evaluation. Allowed values: 'wmt20', 'wmt22', 'wmt23', "
        "'wmt24'. Default: 'wmt20'.",
    )

    parser.add_argument(
        "--lp",
        type=str,
        choices=["en-de", "zh-en", "en-zh", "en-es"],
        default="en-de",
        help="Specifies the language pair to evaluate within the selected WMT test set. Allowed values: 'en-de', "
        "'zh-en', 'en-zh', 'en-es'. Default: 'en-de'.",
    )

    parser.add_argument(
        "--new-human-annotations-dir",
        type=Path,
        help="Path to the directory containing new human annotations.",
    )

    parser.add_argument(
        "--use-only-google-mqm-and-da-sqm",
        action="store_true",
        help="If set, restricts meta-evaluation to MQM (Google) and DA+SQM annotations. Applicable only to the en-de "
        "language pair in wmt23.",
    )

    parser.add_argument(
        "--gold-name",
        type=str,
        default="mqm",
        help="Which human ratings to use as gold scores. Default: 'mqm'.",
    )

    parser.add_argument(
        "--kwto-k",
        type=int,
        default=1000,
        help="Number of resampling runs for KendallWithTiesOpt statistical significance. Default: 1000.",
    )

    parser.add_argument(
        "--spa-k",
        type=int,
        default=1000,
        help="Number of resampling runs for SPA statistical significance. Default: 1000.",
    )

    parser.add_argument(
        "--new-metrics-path",
        type=Path,
        help="Path to the file containing the info for the new metrics to include.",
    )

    return parser


def get_new_metric2seg_scores(
    new_metrics_path: Path, eval_set_name: str, lp: str = "en-de"
) -> Dict[str, Dict[str, List[float]]]:
    """Read the input file and return a dictionary with the scores for each metric.

    Args:
        new_metrics_path (Path): Path to the file containing the info for the new metrics to include.
        eval_set_name (str): Name of the WMT Metrics evaluation set to consider.
        lp (str): Language pair to consider. Default: 'en-de'.

    Returns:
        Dict[str, Dict[str, List[float]]]: A dictionary where keys are metric names and values are dictionaries
        mapping system names to their segment scores.

    Raises:
        ValueError: If the input file does not contain exactly 2 tab-separated values per line.
    """
    new_metric2seg_scores = dict()
    with open(new_metrics_path, mode="r", encoding="utf-8") as tsv_file:
        reader = csv.reader(tsv_file, delimiter="\t")
        for row in reader:
            if len(row) != 2:
                raise ValueError(
                    f"Expected 2 tab-separated values, got {len(row)}: {row}"
                )
            metric_name, metric_dir = row
            with open(
                METRICS_OUTPUTS_DIR
                / eval_set_name
                / lp
                / metric_dir
                / SEG_SCORES_FILENAME,
                "rb",
            ) as handle:
                new_metric2seg_scores[metric_name] = pickle.load(handle)

    return new_metric2seg_scores


def measure_wmt24_agreement_with_esa(k: int) -> None:
    """Perform the meta-eval on wmt24.en-es adding ESA among the metrics.

    Args:
        k (int): Number of resampling runs for statistical significance.
    """
    wmt24_eval_set = data.EvalSet("wmt24", "en-es", True)

    wmt24_esa_annotation = wmt24_eval_set.Scores("seg", "esa")
    esa_annotated_sys, n_esa_annotated_segs = [], 0
    for sys, esa_seg_scores in wmt24_esa_annotation.items():
        if any(esa is not None for esa in esa_seg_scores) > 0:
            if n_esa_annotated_segs == 0:
                n_esa_annotated_segs = sum(
                    1 for esa in esa_seg_scores if esa is not None
                )
            esa_annotated_sys.append(sys)
    print("\n")
    print(f"# ESA-annotated MT systems in wmt24.en-es = {len(esa_annotated_sys)}.")
    print(f"# ESA-annotated segments in wmt24.en-es = {n_esa_annotated_segs}.")
    print("\n")

    wmt24_mqm_annotation = wmt24_eval_set.Scores("seg", "mqm")
    assert set(wmt24_esa_annotation) == set(wmt24_mqm_annotation)

    mqm_annotated_sys, n_mqm_annotated_segs = [], 0
    for sys, mqm_seg_scores in wmt24_mqm_annotation.items():
        if any(mqm is not None for mqm in mqm_seg_scores):
            if n_mqm_annotated_segs == 0:
                n_mqm_annotated_segs = sum(
                    1 for mqm in mqm_seg_scores if mqm is not None
                )
            mqm_annotated_sys.append(sys)
    print(f"# MQM-annotated MT systems in wmt24.en-es = {len(mqm_annotated_sys)}.")
    print(f"# MQM-annotated segs = {n_mqm_annotated_segs}.")
    print("\n")

    sys_names = set(esa_annotated_sys).intersection(set(mqm_annotated_sys))
    sys_names.discard(wmt24_eval_set.std_ref)  # For ref-based metrics
    sys_names = sorted(sys_names)  # Use a sorted list for consistent order

    print(f"# MT Systems annotated with both ESA and MQM = {len(sys_names)}.")

    filtered_seg_ids = []
    for sys in sys_names:
        esa_seg_scores, mqm_seg_scores = (
            wmt24_esa_annotation[sys],
            wmt24_mqm_annotation[sys],
        )
        assert len(esa_seg_scores) == len(mqm_seg_scores)

        if not any(esa is not None for esa in esa_seg_scores) or not any(
            mqm is not None for mqm in mqm_seg_scores
        ):
            continue

        for seg_idx in range(len(mqm_seg_scores)):
            if (
                esa_seg_scores[seg_idx] is not None
                and mqm_seg_scores[seg_idx] is not None
            ):
                filtered_seg_ids.append(seg_idx)

        break

    print(f"# segs annotated with both ESA and MQM. = {len(filtered_seg_ids)}.")
    print("\n")
    print("\n")
    print("\n")

    gold_filtered_seg_scores = {
        sys: [wmt24_mqm_annotation[sys][seg_idx] for seg_idx in filtered_seg_ids]
        for sys in sys_names
    }
    gold_sys_scores = wmt24_eval_set.Scores("sys", "mqm")
    gold_sys_scores = {sys: gold_sys_scores[sys] for sys in sys_names}

    main_refs = {wmt24_eval_set.std_ref}
    seg_correlations, sys_correlations, sys_correlations_all = (
        dict(),
        dict(),
        dict(),
    )
    for metric_name in wmt24_eval_set.metric_names | {"esa"}:
        assert metric_name != "mqm"

        if metric_name in wmt24_eval_set.metric_names:
            base_name, metric_refs = wmt24_eval_set.ParseMetricName(metric_name)
            if (
                base_name not in wmt24_eval_set.primary_metrics
                and base_name != "CometKiwi-XL"
                and base_name != "CometKiwi-XXL"
            ):
                continue
            if not metric_refs.issubset(main_refs):
                continue
            display_name = wmt24_eval_set.DisplayName(metric_name, "spreadsheet")
        else:
            assert metric_name == "esa"
            display_name = metric_name

        if metric_name == "esa":
            metric_seg_scores = wmt24_esa_annotation
        else:
            metric_seg_scores = wmt24_eval_set.Scores("seg", metric_name)

        if not metric_seg_scores:  # Metric not available at this level.
            continue

        filtered_metric_seg_scores = {
            sys: [metric_seg_scores[sys][seg_idx] for seg_idx in filtered_seg_ids]
            for sys in sys_names
        }
        metric_sys_scores, metric_sys_scores_all = {
            sys: [
                sum(filtered_metric_seg_scores[sys])
                / len(filtered_metric_seg_scores[sys])
            ]
            for sys in sys_names
        }, {
            sys: [
                sum(
                    seg_score
                    for seg_score in metric_seg_scores[sys]
                    if seg_score is not None
                )
                / sum(
                    1 for seg_score in metric_seg_scores[sys] if seg_score is not None
                )
            ]
            for sys in sys_names
        }
        (
            seg_correlations[display_name],
            sys_correlations[display_name],
            sys_correlations_all[display_name],
        ) = (
            wmt24_eval_set.Correlation(
                gold_filtered_seg_scores, filtered_metric_seg_scores, sys_names
            ),
            wmt24_eval_set.Correlation(gold_sys_scores, metric_sys_scores, sys_names),
            wmt24_eval_set.Correlation(
                gold_sys_scores, metric_sys_scores_all, sys_names
            ),
        )

    psd, pvalue = stats.PermutationSigDiffParams(100, 0.02, 0.50), 0.05
    corrs_and_ranks, sig_matrix, draws_index, draws_list = data.CompareMetrics(
        seg_correlations,
        stats.KendallWithTiesOpt,
        "item",
        k,
        psd,
        pvalue,
        False,
        "pairs",
        parallel_file=None,
        sample_rate=1.0,
    )
    print("\n")
    print("Seg-Lev KendallWithTiesOpt:")
    data.PrintMetricComparison(
        corrs_and_ranks,
        sig_matrix,
        pvalue,
    )
    print("\n")

    for corr_fcn in [scipy.stats.pearsonr, scipy.stats.kendalltau]:
        for average_by in ["none", "item", "sys"]:
            (
                corrs_and_ranks,
                sig_matrix,
                draws_index,
                draws_list,
            ) = data.CompareMetrics(
                seg_correlations,
                corr_fcn,
                average_by,
                0,
                psd,
                pvalue,
                False,
                "scores",
                parallel_file=None,
            )
            print("\n")
            print(f"Seg-Lev {corr_fcn.__name__} (average_by={average_by}):")
            data.PrintMetricComparison(
                corrs_and_ranks,
                sig_matrix,
                pvalue,
            )
            print("\n")

    def print_PA_res(corrs: Dict[str, stats.Correlation], print_msg: str) -> None:
        # Compute metric PAs, ordered by decreasing correlation.
        corrs_and_ranks = dict()
        for m, c in corrs.items():
            corrs_and_ranks[m] = [
                pairwise_acc(c.gold_scores, c.metric_scores)[0],
                0,
            ]
        # Use metric name as secondary sort criterion to stabilize ties.
        corrs_and_ranks = dict(
            sorted(corrs_and_ranks.items(), key=lambda x: (-x[1][0], x[0]))
        )

        # Compute significance matrix and determine ranks.
        sig_matrix, draws_index, draws_list = ComputeSigMatrix(
            corrs,
            corrs_and_ranks,
            pairwise_acc,
            "none",
            0,
            psd,
            False,
            "scores",
            None,
        )
        ranks = AssignRanks(sig_matrix, pvalue)
        for i, m in enumerate(corrs_and_ranks):
            corrs_and_ranks[m][1] = ranks[i]
        print("\n")
        print(print_msg)
        pp(corrs_and_ranks)
        print("\n")

    print_PA_res(sys_correlations, "Sys-Lev PA:")
    print_PA_res(sys_correlations_all, "Sys-Lev PA (with all available seg scores):")

    (
        corrs_and_ranks,
        sig_matrix,
        draws_index,
        draws_list,
    ) = data.CompareMetricsWithPairwiseConfidenceError(
        seg_correlations,
        k,
        psd,
        pvalue,
        False,
        "scores",
        parallel_file=None,
    )
    print("\n")
    print("Sys-Lev SPA:")
    data.PrintMetricComparison(
        corrs_and_ranks,
        sig_matrix,
        pvalue,
    )
    print("\n")

    corrs_and_ranks, sig_matrix, draws_index, draws_list = data.CompareMetrics(
        sys_correlations,
        scipy.stats.pearsonr,
        "none",
        0,
        psd,
        pvalue,
        False,
        "scores",
        parallel_file=None,
    )

    print("\n")
    print(f"Sys-Lev Pearson:")
    data.PrintMetricComparison(
        corrs_and_ranks,
        sig_matrix,
        pvalue,
    )
    print("\n")


def measure_wmt20_agreement_raters_ens(
    lp: Literal["en-de", "zh-en"],
    new_human_annotations_dir: Path,
    kwto_k: int,
    spa_k: int,
    new_metrics_path: Optional[Path] = None,
) -> None:
    """Compute and print several meta-evaluation measures considering the additional raters ens in wmt20.

    Args:
        lp (Literal["en-de", "zh-en"]): WMT20 language pair to consider. Allowed values: 'en-de', 'zh-en'.
        new_human_annotations_dir (Path): Path to the directory containing new human annotations.
        kwto_k (int): Number of resampling runs for KendallWithTiesOpt statistical significance.
        spa_k (int): Number of resampling runs for SPA statistical significance.
        new_metrics_path (Optional[Path]): Path to the file containing the info for new metrics. Default: None.
    """
    if lp != "en-de" and lp != "zh-en":
        raise ValueError(
            f"Invalid language pair: {lp}. Allowed values: 'en-de', 'zh-en'."
        )

    print("\n")
    print(f"LP: {lp}.")
    print("\n")

    wmt20_eval_set = data.EvalSet("wmt20", lp, True)

    new_raters_ens_name2seg_scores = dict()
    for new_raters_ens_name in [
        "mqm-col1",
        "mqm-col2",
        "mqm-col3",
        # "psqm-col1", To add the exp for the CR.
        "psqm-col2",
        "psqm-col3",
    ]:
        with open(
            new_human_annotations_dir / lp / f"{new_raters_ens_name}.pickle", "rb"
        ) as handle:
            new_raters_ens_name2seg_scores[new_raters_ens_name] = pickle.load(handle)

    with open(
        new_human_annotations_dir / lp / "psqm-col1.pickle", "rb"
    ) as handle:  # mqm-col1
        wmt20_mqm_annotation = pickle.load(handle)

    mt_systems, filtered_seg_ids = [], []
    for raters_ens_idx, sys2seg_scores in enumerate(
        new_raters_ens_name2seg_scores.values()
    ):
        for sys_idx, (sys, seg_scores) in enumerate(sys2seg_scores.items()):
            assert (
                len(seg_scores)
                == len(wmt20_eval_set.src)
                == len(wmt20_mqm_annotation[sys])
            )
            if raters_ens_idx == 0 and any(score is not None for score in seg_scores):
                mt_systems.append(sys)
            for seg_idx, (seg_score, gt) in enumerate(
                zip(seg_scores, wmt20_mqm_annotation[sys])
            ):
                assert (seg_score is None and gt is None) or (
                    seg_score is not None and gt is not None
                )
                if raters_ens_idx == 0 and sys_idx == 0 and seg_score is not None:
                    filtered_seg_ids.append(seg_idx)

    main_refs = {wmt20_eval_set.std_ref}
    mt_systems = set(mt_systems) - main_refs

    psd, pvalue = stats.PermutationSigDiffParams(100, 0.02, 0.50), 0.05

    print("\n")
    print(f"# segs annotated by all raters ens = {len(filtered_seg_ids)}.")
    print(f"# MT Systems = {len(mt_systems)}.")
    print("\n")

    new_metric2seg_scores = (
        get_new_metric2seg_scores(new_metrics_path, "wmt20", lp)
        if new_metrics_path is not None
        else dict()
    )

    gold_filtered_seg_scores = {
        sys: [wmt20_mqm_annotation[sys][seg_idx] for seg_idx in filtered_seg_ids]
        for sys in mt_systems
    }
    gold_sys_scores = wmt20_eval_set.Scores("sys", "mqm")
    gold_sys_scores = {sys: gold_sys_scores[sys] for sys in mt_systems}

    seg_correlations, sys_correlations = dict(), dict()
    for metric_name in (
        wmt20_eval_set.metric_names
        | set(new_raters_ens_name2seg_scores)
        | set(new_metric2seg_scores)
    ):  # Include new raters ens and new metrics
        assert metric_name != "mqm"

        if metric_name in wmt20_eval_set.metric_names:
            base_name, metric_refs = wmt20_eval_set.ParseMetricName(metric_name)
            if not metric_refs.issubset(main_refs):
                continue
            display_name = wmt20_eval_set.DisplayName(metric_name, "spreadsheet")
        else:
            display_name = metric_name

        if metric_name in new_raters_ens_name2seg_scores:
            metric_seg_scores = new_raters_ens_name2seg_scores[metric_name]
        elif metric_name not in new_metric2seg_scores:
            metric_seg_scores = wmt20_eval_set.Scores("seg", metric_name)
        else:
            metric_seg_scores = new_metric2seg_scores[metric_name]

        if not metric_seg_scores:  # Metric not available at this level.
            continue

        metric_seg_scores = {
            sys: [metric_seg_scores[sys][seg_idx] for seg_idx in filtered_seg_ids]
            for sys in mt_systems
        }
        metric_sys_scores = {
            sys: [
                sum(metric_seg_scores[sys]) / len(metric_seg_scores[sys])
            ]  # Average over segments
            for sys in mt_systems
        }
        (
            seg_correlations[display_name],
            sys_correlations[display_name],
        ) = wmt20_eval_set.Correlation(
            gold_filtered_seg_scores, metric_seg_scores, mt_systems
        ), wmt20_eval_set.Correlation(
            gold_sys_scores, metric_sys_scores, mt_systems
        )

    corrs_and_ranks, sig_matrix, draws_index, draws_list = data.CompareMetrics(
        seg_correlations,
        stats.KendallWithTiesOpt,
        "item",
        kwto_k,
        psd,
        pvalue,
        False,
        "pairs",
        parallel_file=None,
        sample_rate=1.0,
    )
    print("\n")
    print("Seg-Lev KendallWithTiesOpt:")
    data.PrintMetricComparison(
        corrs_and_ranks,
        sig_matrix,
        pvalue,
    )
    print("\n")

    for corr_fcn in [scipy.stats.pearsonr, scipy.stats.kendalltau]:
        for average_by in ["none", "item", "sys"]:
            (
                corrs_and_ranks,
                sig_matrix,
                draws_index,
                draws_list,
            ) = data.CompareMetrics(
                seg_correlations,
                corr_fcn,
                average_by,
                0,
                psd,
                pvalue,
                False,
                "scores",
                parallel_file=None,
            )
            corr_name_and_avg_by = (
                f"Seg-Lev {corr_fcn.__name__} (average_by={average_by})"
            )
            print("\n")
            print(f"{corr_name_and_avg_by}:")
            data.PrintMetricComparison(
                corrs_and_ranks,
                sig_matrix,
                pvalue,
            )
            print("\n")

    # Compute metric PAs, ordered by decreasing correlation.
    corrs_and_ranks = dict()
    for m, c in sys_correlations.items():
        corrs_and_ranks[m] = [
            pairwise_acc(c.gold_scores, c.metric_scores)[0],
            0,
        ]
    # Use metric name as secondary sort criterion to stabilize ties.
    corrs_and_ranks = dict(
        sorted(corrs_and_ranks.items(), key=lambda x: (-x[1][0], x[0]))
    )

    # Compute significance matrix and determine ranks.
    sig_matrix, draws_index, draws_list = ComputeSigMatrix(
        sys_correlations,
        corrs_and_ranks,
        pairwise_acc,
        "none",
        0,
        psd,
        False,
        "scores",
        None,
    )
    ranks = AssignRanks(sig_matrix, pvalue)
    for i, m in enumerate(corrs_and_ranks):
        corrs_and_ranks[m][1] = ranks[i]
    print("\n")
    print("Sys-Lev PA:")
    pp(corrs_and_ranks)
    print("\n")

    (
        corrs_and_ranks,
        sig_matrix,
        draws_index,
        draws_list,
    ) = data.CompareMetricsWithPairwiseConfidenceError(
        seg_correlations,
        spa_k,
        psd,
        pvalue,
        False,
        "scores",
        parallel_file=None,
    )
    print("\n")
    print("Sys-Lev SPA:")
    data.PrintMetricComparison(
        corrs_and_ranks,
        sig_matrix,
        pvalue,
    )
    print("\n")

    corrs_and_ranks, sig_matrix, draws_index, draws_list = data.CompareMetrics(
        sys_correlations,
        scipy.stats.pearsonr,
        "none",
        0,
        psd,
        pvalue,
        False,
        "scores",
        parallel_file=None,
    )

    print("\n")
    print(f"Sys-Lev Pearson:")
    data.PrintMetricComparison(
        corrs_and_ranks,
        sig_matrix,
        pvalue,
    )


def measure_wmt20_agreement(
    lp: Literal["en-de", "zh-en"],
    kwto_k: int,
    spa_k: int,
    wmt20_raters_subset_for_inter_rater: Set[str],
    new_metrics_path: Optional[Path] = None,
    new_human_annotations_dir: Optional[Path] = None,
) -> None:
    """Compute and print several meta-evaluation measures considering the MQM human raters in wmt20.

    Args:
        lp (Literal["en-de", "zh-en"]): WMT20 language pair to consider. Allowed values: 'en-de', 'zh-en'.
        kwto_k (int): Number of resampling runs for KendallWithTiesOpt statistical significance.
        spa_k (int): Number of resampling runs for SPA statistical significance.
        wmt20_raters_subset_for_inter_rater (Set[str]): Subset of WMT20 raters to take into account.
        new_metrics_path (Optional[Path]): Path to the file containing the info for new metrics. Default: None.
        new_human_annotations_dir (Optional[Path]): Path to the dir containing new human annotations. Default: None.
    """
    if lp != "en-de" and lp != "zh-en":
        raise ValueError(
            f"Invalid language pair: {lp}. Allowed values: 'en-de', 'zh-en'."
        )

    print("\n")
    print(f"LP: {lp}.")
    print("\n")
    wmt20_eval_set = data.EvalSet("wmt20", lp, True)

    new_raters_ens_name2seg_scores = dict()
    if new_human_annotations_dir is not None:
        for new_raters_ens_name in [
            "psqm-col1",
            "psqm-col2",
            "psqm-col3",
        ]:
            with open(
                new_human_annotations_dir / lp / f"{new_raters_ens_name}.pickle", "rb"
            ) as handle:
                new_raters_ens_name2seg_scores[new_raters_ens_name] = pickle.load(
                    handle
                )

    # Convert raters set to a sorted list based on the numerical suffix
    sorted_raters = sorted(
        wmt20_raters_subset_for_inter_rater, key=lambda x: int(x.split("mqm-rater")[1])
    )
    # Iterate over raters pairs where the first (gold) has a lower index than the second (metric)
    for gold_rater, metric_rater in [
        (gold_rater, metric_rater)
        for gold_rater, metric_rater in itertools.combinations(sorted_raters, 2)
    ]:
        print("\n")
        print(f"Gold MQM rater = {gold_rater}, Metric MQM rater = {metric_rater}.")
        (
            gold_rater_sys2seg_scores,
            metric_rater_sys2seg_scores,
        ) = wmt20_eval_set.Scores("seg", gold_rater), wmt20_eval_set.Scores(
            "seg", metric_rater
        )
        assert set(gold_rater_sys2seg_scores) == set(metric_rater_sys2seg_scores)
        sys_names = sorted(
            set(gold_rater_sys2seg_scores)
            - {wmt20_eval_set.std_ref}  # For ref-based metrics
        )
        print(f"# MT Systems = {len(sys_names)}.")

        filtered_seg_ids = []
        first_mt_sys = next(iter(sys_names))
        for seg_idx in range(len(gold_rater_sys2seg_scores[first_mt_sys])):
            if (
                gold_rater_sys2seg_scores[first_mt_sys][seg_idx] is not None
                and metric_rater_sys2seg_scores[first_mt_sys][seg_idx] is not None
            ):
                filtered_seg_ids.append(seg_idx)
        print(
            f"# segs annotated by gold MQM rater = "
            f"{sum(1 for seg_score in gold_rater_sys2seg_scores[first_mt_sys] if seg_score is not None)}."
        )
        print(
            f"# segs annotated by metric MQM rater = "
            f"{sum(1 for seg_score in metric_rater_sys2seg_scores[first_mt_sys] if seg_score is not None)}."
        )
        print(f"# segs annotated by both MQM raters = {len(filtered_seg_ids)}.")
        print("\n")

        new_metric2seg_scores = (
            get_new_metric2seg_scores(new_metrics_path, "wmt20", lp)
            if new_metrics_path is not None
            else dict()
        )

        gold_filtered_seg_scores = {
            sys: [
                gold_rater_sys2seg_scores[sys][seg_idx] for seg_idx in filtered_seg_ids
            ]
            for sys in sys_names
        }
        gold_sys_scores = {
            sys: [
                sum(gold_filtered_seg_scores[sys]) / len(gold_filtered_seg_scores[sys])
            ]
            for sys in sys_names
        }
        gold_sys_scores_all = wmt20_eval_set.Scores("sys", gold_rater)
        gold_sys_scores_all = {sys: gold_sys_scores_all[sys] for sys in sys_names}

        main_refs = {wmt20_eval_set.std_ref}
        seg_correlations, sys_correlations, sys_correlations_all = (
            dict(),
            dict(),
            dict(),
        )
        for metric_name in (
            wmt20_eval_set.metric_names
            | {metric_rater}
            | set(new_raters_ens_name2seg_scores)
            | set(new_metric2seg_scores)
        ):  # Add metric rater, raters ens, and new metrics
            if metric_name in wmt20_eval_set.metric_names:
                base_name, metric_refs = wmt20_eval_set.ParseMetricName(metric_name)
                if not metric_refs.issubset(main_refs):
                    continue
                display_name = wmt20_eval_set.DisplayName(metric_name, "spreadsheet")
            else:
                display_name = metric_name

            if metric_name in new_raters_ens_name2seg_scores:
                metric_seg_scores = new_raters_ens_name2seg_scores[metric_name]
            elif metric_name not in new_metric2seg_scores:
                metric_seg_scores = wmt20_eval_set.Scores("seg", metric_name)
            else:
                metric_seg_scores = new_metric2seg_scores[metric_name]

            if not metric_seg_scores:  # Metric not available at this level.
                continue

            filtered_metric_seg_scores = {
                sys: [metric_seg_scores[sys][seg_idx] for seg_idx in filtered_seg_ids]
                for sys in sys_names
            }
            metric_sys_scores, metric_sys_scores_all = {
                sys: [
                    sum(filtered_metric_seg_scores[sys])
                    / len(filtered_metric_seg_scores[sys])
                ]
                for sys in sys_names
            }, {
                sys: [
                    sum(
                        seg_score
                        for seg_score in metric_seg_scores[sys]
                        if seg_score is not None
                    )
                    / sum(
                        1
                        for seg_score in metric_seg_scores[sys]
                        if seg_score is not None
                    )
                ]
                for sys in sys_names
            }
            (
                seg_correlations[display_name],
                sys_correlations[display_name],
                sys_correlations_all[display_name],
            ) = (
                wmt20_eval_set.Correlation(
                    gold_filtered_seg_scores, filtered_metric_seg_scores, sys_names
                ),
                wmt20_eval_set.Correlation(
                    gold_sys_scores, metric_sys_scores, sys_names
                ),
                wmt20_eval_set.Correlation(
                    gold_sys_scores_all, metric_sys_scores_all, sys_names
                ),
            )

        psd, pvalue = stats.PermutationSigDiffParams(100, 0.02, 0.50), 0.05
        corrs_and_ranks, sig_matrix, draws_index, draws_list = data.CompareMetrics(
            seg_correlations,
            stats.KendallWithTiesOpt,
            "item",
            kwto_k,
            psd,
            pvalue,
            False,
            "pairs",
            parallel_file=None,
            sample_rate=1.0,
        )
        print("\n")
        print("Seg-Lev KendallWithTiesOpt:")
        data.PrintMetricComparison(
            corrs_and_ranks,
            sig_matrix,
            pvalue,
        )
        print("\n")

        for corr_fcn in [scipy.stats.pearsonr, scipy.stats.kendalltau]:
            for average_by in ["none", "item", "sys"]:
                (
                    corrs_and_ranks,
                    sig_matrix,
                    draws_index,
                    draws_list,
                ) = data.CompareMetrics(
                    seg_correlations,
                    corr_fcn,
                    average_by,
                    0,
                    psd,
                    pvalue,
                    False,
                    "scores",
                    parallel_file=None,
                )
                print("\n")
                print(f"Seg-Lev {corr_fcn.__name__} (average_by={average_by}):")
                data.PrintMetricComparison(
                    corrs_and_ranks,
                    sig_matrix,
                    pvalue,
                )
                print("\n")

        def print_PA_res(corrs: Dict[str, stats.Correlation], print_msg: str) -> None:
            # Compute metric PAs, ordered by decreasing correlation.
            corrs_and_ranks = dict()
            for m, c in corrs.items():
                corrs_and_ranks[m] = [
                    pairwise_acc(c.gold_scores, c.metric_scores)[0],
                    0,
                ]
            # Use metric name as secondary sort criterion to stabilize ties.
            corrs_and_ranks = dict(
                sorted(corrs_and_ranks.items(), key=lambda x: (-x[1][0], x[0]))
            )

            # Compute significance matrix and determine ranks.
            sig_matrix, draws_index, draws_list = ComputeSigMatrix(
                corrs,
                corrs_and_ranks,
                pairwise_acc,
                "none",
                0,
                psd,
                False,
                "scores",
                None,
            )
            ranks = AssignRanks(sig_matrix, pvalue)
            for i, m in enumerate(corrs_and_ranks):
                corrs_and_ranks[m][1] = ranks[i]
            print("\n")
            print(print_msg)
            pp(corrs_and_ranks)
            print("\n")

        print_PA_res(sys_correlations, "Sys-Lev PA:")
        print_PA_res(
            sys_correlations_all, "Sys-Lev PA (with all available seg scores):"
        )

        (
            corrs_and_ranks,
            sig_matrix,
            draws_index,
            draws_list,
        ) = data.CompareMetricsWithPairwiseConfidenceError(
            seg_correlations,
            spa_k,
            psd,
            pvalue,
            False,
            "scores",
            parallel_file=None,
        )
        print("\n")
        print("Sys-Lev SPA:")
        data.PrintMetricComparison(
            corrs_and_ranks,
            sig_matrix,
            pvalue,
        )
        print("\n")

        corrs_and_ranks, sig_matrix, draws_index, draws_list = data.CompareMetrics(
            sys_correlations,
            scipy.stats.pearsonr,
            "none",
            0,
            psd,
            pvalue,
            False,
            "scores",
            parallel_file=None,
        )

        print("\n")
        print(f"Sys-Lev Pearson:")
        data.PrintMetricComparison(
            corrs_and_ranks,
            sig_matrix,
            pvalue,
        )
        print("\n\n\n")

    print("\n\n\n")


def measure_wmt22_agreement_raters_ens(
    new_human_annotations_dir: Path,
    k: int,
    lp: Literal["en-de", "en-zh"],
    new_metrics_path: Optional[Path] = None,
    gold_name: str = "mqm-col1",
) -> None:
    """Compute and print several meta-evaluation measures considering three raters ensembles in wmt22.

    Args:
        new_human_annotations_dir (Path): Path to the directory containing the raters ensemble columns for wmt22.
        k (int): Number of resampling runs for statistical significance.
        lp (Literal["en-de", "en-zh"]): WMT22 language pair for agreement. Allowed values: 'en-de', 'en-zh'.
        new_metrics_path (Optional[Path): Path to the file containing the info for new metrics. Default: None.
    """
    if lp != "en-de" and lp != "en-zh":
        raise ValueError(
            f"Invalid language pair: {lp}. Allowed values: 'en-de', 'en-zh'."
        )

    wmt22_eval_set = data.EvalSet(
        "wmt22",
        lp,
        True,
    )

    new_raters_ens_name2seg_scores = dict()
    for new_raters_ens_name in ["mqm-col1", "mqm-col2", "mqm-col3"]:
        if new_raters_ens_name == gold_name:
            continue

        with open(
            new_human_annotations_dir / lp / f"{new_raters_ens_name}.pickle", "rb"
        ) as handle:
            new_raters_ens_name2seg_scores[new_raters_ens_name] = pickle.load(handle)

    gold_annotation = None
    if gold_name.startswith("mqm"):
        with open(
            new_human_annotations_dir / lp / f"{gold_name}.pickle", "rb"
        ) as handle:
            gold_annotation = pickle.load(handle)

    wmt22_da_sqm_annotation = wmt22_eval_set.Scores("seg", "wmt-appraise")

    if gold_name == "da-sqm":
        gold_annotation = wmt22_da_sqm_annotation

    assert len(gold_annotation) > 0

    mt_systems, filtered_seg_ids = [], []
    for raters_ens_idx, sys2seg_scores in enumerate(
        new_raters_ens_name2seg_scores.values()
    ):
        for sys_idx, (sys, seg_scores) in enumerate(sys2seg_scores.items()):
            assert (
                len(seg_scores)
                == len(wmt22_eval_set.src)
                == len(gold_annotation[sys])
                == len(wmt22_da_sqm_annotation[sys])
            )
            da_sqm_seg_scores = wmt22_da_sqm_annotation[
                sys
            ]  # DA+SQM in wmt22 contains less annotations.
            if (
                raters_ens_idx == 0
                and any(score is not None for score in seg_scores)
                and any(score is not None for score in da_sqm_seg_scores)
            ):
                mt_systems.append(sys)
            for seg_idx, (seg_score, gt, da_sqm) in enumerate(
                zip(seg_scores, gold_annotation[sys], da_sqm_seg_scores)
            ):
                assert (seg_score is None and gt is None) or (
                    seg_score is not None and gt is not None
                )
                if (
                    raters_ens_idx == 0
                    and sys_idx == 0
                    and seg_score is not None
                    and da_sqm is not None
                ):
                    filtered_seg_ids.append(seg_idx)

    main_refs = {wmt22_eval_set.std_ref}
    mt_systems = set(mt_systems) - main_refs

    psd, pvalue = stats.PermutationSigDiffParams(100, 0.02, 0.50), 0.05

    print("\n")
    print(f"LP = {lp}.")
    print(f"# segs MQM-annotated by all raters ens = {len(filtered_seg_ids)}.")
    print(f"# MT Systems = {len(mt_systems)}.")
    print("\n")

    new_metric2seg_scores = (
        get_new_metric2seg_scores(new_metrics_path, "wmt22", lp)
        if new_metrics_path is not None
        else dict()
    )

    gold_filtered_seg_scores = {
        sys: [gold_annotation[sys][seg_idx] for seg_idx in filtered_seg_ids]
        for sys in mt_systems
    }
    if lp == "to-skip-for-tebuttal":
        gold_sys_scores = wmt22_eval_set.Scores("sys", "mqm")
        gold_sys_scores = {sys: gold_sys_scores[sys] for sys in mt_systems}
    else:
        gold_sys_scores = {
            sys: [
                sum(score for score in gold_annotation[sys] if score is not None)
                / sum(1 for score in gold_annotation[sys] if score is not None)
            ]
            for sys in mt_systems
        }

    seg_correlations, sys_correlations = dict(), dict()
    new_raters_ens_name2seg_scores["da-sqm"] = wmt22_da_sqm_annotation
    for metric_name in (
        wmt22_eval_set.metric_names
        | set(new_raters_ens_name2seg_scores)
        | set(new_metric2seg_scores)
    ):  # Include new raters ens and new metrics
        assert metric_name != "mqm"
        if metric_name == gold_name:
            continue

        if metric_name in wmt22_eval_set.metric_names:
            base_name, metric_refs = wmt22_eval_set.ParseMetricName(metric_name)
            if base_name not in wmt22_eval_set.primary_metrics:
                continue
            if not metric_refs.issubset(main_refs):
                continue
            display_name = wmt22_eval_set.DisplayName(metric_name, "spreadsheet")
        else:
            display_name = metric_name

        if metric_name in new_raters_ens_name2seg_scores:
            metric_seg_scores = new_raters_ens_name2seg_scores[metric_name]
        elif metric_name not in new_metric2seg_scores:
            metric_seg_scores = wmt22_eval_set.Scores("seg", metric_name)
        else:
            metric_seg_scores = new_metric2seg_scores[metric_name]

        if not metric_seg_scores:  # Metric not available at this level.
            continue

        metric_seg_scores = {
            sys: [metric_seg_scores[sys][seg_idx] for seg_idx in filtered_seg_ids]
            for sys in mt_systems
        }
        metric_sys_scores = {
            sys: [
                sum(metric_seg_scores[sys]) / len(metric_seg_scores[sys])
            ]  # Average over segments
            for sys in mt_systems
        }
        (
            seg_correlations[display_name],
            sys_correlations[display_name],
        ) = wmt22_eval_set.Correlation(
            gold_filtered_seg_scores, metric_seg_scores, mt_systems
        ), wmt22_eval_set.Correlation(
            gold_sys_scores, metric_sys_scores, mt_systems
        )

    corrs_and_ranks, sig_matrix, draws_index, draws_list = data.CompareMetrics(
        seg_correlations,
        stats.KendallWithTiesOpt,
        "item",
        k,
        psd,
        pvalue,
        False,
        "pairs",
        parallel_file=None,
        sample_rate=1.0,
    )
    print("\n")
    print("Seg-Lev KendallWithTiesOpt:")
    data.PrintMetricComparison(
        corrs_and_ranks,
        sig_matrix,
        pvalue,
    )
    print("\n")

    for corr_fcn in [scipy.stats.pearsonr, scipy.stats.kendalltau]:
        for average_by in ["none", "item", "sys"]:
            (
                corrs_and_ranks,
                sig_matrix,
                draws_index,
                draws_list,
            ) = data.CompareMetrics(
                seg_correlations,
                corr_fcn,
                average_by,
                0,
                psd,
                pvalue,
                False,
                "scores",
                parallel_file=None,
            )
            corr_name_and_avg_by = (
                f"Seg-Lev {corr_fcn.__name__} (average_by={average_by})"
            )
            print("\n")
            print(f"{corr_name_and_avg_by}:")
            data.PrintMetricComparison(
                corrs_and_ranks,
                sig_matrix,
                pvalue,
            )
            print("\n")

    # Compute metric PAs, ordered by decreasing correlation.
    corrs_and_ranks = dict()
    for m, c in sys_correlations.items():
        corrs_and_ranks[m] = [
            pairwise_acc(c.gold_scores, c.metric_scores)[0],
            0,
        ]
    # Use metric name as secondary sort criterion to stabilize ties.
    corrs_and_ranks = dict(
        sorted(corrs_and_ranks.items(), key=lambda x: (-x[1][0], x[0]))
    )

    # Compute significance matrix and determine ranks.
    sig_matrix, draws_index, draws_list = ComputeSigMatrix(
        sys_correlations,
        corrs_and_ranks,
        pairwise_acc,
        "none",
        0,
        psd,
        False,
        "scores",
        None,
    )
    ranks = AssignRanks(sig_matrix, pvalue)
    for i, m in enumerate(corrs_and_ranks):
        corrs_and_ranks[m][1] = ranks[i]
    print("\n")
    print("Sys-Lev PA:")
    pp(corrs_and_ranks)
    print("\n")

    (
        corrs_and_ranks,
        sig_matrix,
        draws_index,
        draws_list,
    ) = data.CompareMetricsWithPairwiseConfidenceError(
        seg_correlations,
        k,
        psd,
        pvalue,
        False,
        "scores",
        parallel_file=None,
    )
    print("\n")
    print("Sys-Lev SPA:")
    data.PrintMetricComparison(
        corrs_and_ranks,
        sig_matrix,
        pvalue,
    )
    print("\n")

    corrs_and_ranks, sig_matrix, draws_index, draws_list = data.CompareMetrics(
        sys_correlations,
        scipy.stats.pearsonr,
        "none",
        0,
        psd,
        pvalue,
        False,
        "scores",
        parallel_file=None,
    )

    print("\n")
    print(f"Sys-Lev Pearson:")
    data.PrintMetricComparison(
        corrs_and_ranks,
        sig_matrix,
        pvalue,
    )
    print("\n\n\n")


def measure_wmt22_agreement(
    lp: Literal["en-de", "en-zh"],
    kwto_k: int,
    spa_k: int,
    raters_subset_for_inter_rater: Set[str],
    new_metrics_path: Optional[Path] = None,
) -> None:
    """Compute and print several meta-evaluation measures considering the MQM human raters in wmt22.

    Args:
        lp (Literal["en-de", "en-zh"]): WMT20 language pair to consider. Allowed values: 'en-de', 'en-zh'.
        kwto_k (int): Number of resampling runs for KendallWithTiesOpt statistical significance.
        spa_k (int): Number of resampling runs for SPA statistical significance.
        raters_subset_for_inter_rater (Set[str]): Subset of WMT22 raters to take into account.
        new_metrics_path (Optional[Path]): Path to the file containing the info for new metrics. Default: None.
    """
    if lp != "en-de" and lp != "en-zh":
        raise ValueError(
            f"Invalid language pair: {lp}. Allowed values: 'en-de', 'en-zh'."
        )

    print("\n")
    print(f"LP: {lp}.")
    print("\n")
    wmt22_eval_set = data.EvalSet("wmt22", lp, True)

    # Convert raters set to a sorted list based on the numerical suffix
    sorted_raters = sorted(
        raters_subset_for_inter_rater, key=lambda x: int(x.split("mqm-rater")[1])
    )
    # Iterate over raters pairs where the first (gold) has a lower index than the second (metric)
    for gold_rater, metric_rater in [
        (gold_rater, metric_rater)
        for gold_rater, metric_rater in itertools.combinations(sorted_raters, 2)
    ]:
        print("\n")
        print(f"Gold MQM rater = {gold_rater}, Metric MQM rater = {metric_rater}.")

        gold_mqm_rater_ratings, metric_mqm_rater_ratings = wmt22_eval_set.Ratings(
            gold_rater
        ), wmt22_eval_set.Ratings(metric_rater)
        assert (
            set(gold_mqm_rater_ratings)
            == set(metric_mqm_rater_ratings)
            == wmt22_eval_set.sys_names
        )

        rater2mqm_scores = dict()
        for mqm_rater, mqm_rater_ratings in [
            ("gold", gold_mqm_rater_ratings),
            ("metric", metric_mqm_rater_ratings),
        ]:
            rater2mqm_scores[mqm_rater] = dict()
            for sys, error_spans in mqm_rater_ratings.items():
                rater2mqm_scores[mqm_rater][sys] = []
                for seg_idx, rating in enumerate(error_spans):
                    if rating is None:
                        rater2mqm_scores[mqm_rater][sys].append(None)
                        continue

                    seg_score = 0
                    for error in rating.errors:
                        score_to_sum = -error.score
                        if error.category == "source_error":
                            score_to_sum = -1 if error.severity == "minor" else -5
                        seg_score += score_to_sum

                    rater2mqm_scores[mqm_rater][sys].append(seg_score)

        (
            gold_rater_sys2seg_scores,
            metric_rater_sys2seg_scores,
        ) = rater2mqm_scores.pop("gold"), rater2mqm_scores.pop("metric")
        assert set(gold_rater_sys2seg_scores) == set(metric_rater_sys2seg_scores)

        sys_names = sorted(
            set(gold_rater_sys2seg_scores)
            - {wmt22_eval_set.std_ref}  # For ref-based metrics
        )
        print(f"# MT Systems = {len(sys_names)}.")

        filtered_seg_ids = []
        first_mt_sys = next(iter(sys_names))
        for seg_idx in range(len(gold_rater_sys2seg_scores[first_mt_sys])):
            if (
                gold_rater_sys2seg_scores[first_mt_sys][seg_idx] is not None
                and metric_rater_sys2seg_scores[first_mt_sys][seg_idx] is not None
            ):
                filtered_seg_ids.append(seg_idx)
        print(
            f"# segs annotated by gold MQM rater = "
            f"{sum(1 for seg_score in gold_rater_sys2seg_scores[first_mt_sys] if seg_score is not None)}."
        )
        print(
            f"# segs annotated by metric MQM rater = "
            f"{sum(1 for seg_score in metric_rater_sys2seg_scores[first_mt_sys] if seg_score is not None)}."
        )
        print(f"# segs annotated by both MQM raters = {len(filtered_seg_ids)}.")
        print("\n")

        new_metric2seg_scores = (
            get_new_metric2seg_scores(new_metrics_path, "wmt22", lp)
            if new_metrics_path is not None
            else dict()
        )

        gold_filtered_seg_scores = {
            sys: [
                gold_rater_sys2seg_scores[sys][seg_idx] for seg_idx in filtered_seg_ids
            ]
            for sys in sys_names
        }
        gold_sys_scores, gold_sys_scores_all = {
            sys: [
                sum(gold_filtered_seg_scores[sys]) / len(gold_filtered_seg_scores[sys])
            ]
            for sys in sys_names
        }, {
            sys: [
                sum(
                    seg_score
                    for seg_score in gold_rater_sys2seg_scores[sys]
                    if seg_score is not None
                )
                / sum(
                    1
                    for seg_score in gold_rater_sys2seg_scores[sys]
                    if seg_score is not None
                )
            ]
            for sys in sys_names
        }

        main_refs = {wmt22_eval_set.std_ref}
        seg_correlations, sys_correlations, sys_correlations_all = (
            dict(),
            dict(),
            dict(),
        )
        for metric_name in (
            wmt22_eval_set.metric_names | {metric_rater} | set(new_metric2seg_scores)
        ):  # Add metric rater, raters ens, and new metrics
            assert metric_name != "mqm"

            if metric_name in wmt22_eval_set.metric_names:
                base_name, metric_refs = wmt22_eval_set.ParseMetricName(metric_name)
                if base_name not in wmt22_eval_set.primary_metrics:
                    continue
                if not metric_refs.issubset(main_refs):
                    continue
                display_name = wmt22_eval_set.DisplayName(metric_name, "spreadsheet")
            else:
                display_name = metric_name

            if metric_name == metric_rater:
                metric_seg_scores = metric_rater_sys2seg_scores
            elif metric_name not in new_metric2seg_scores:
                metric_seg_scores = wmt22_eval_set.Scores("seg", metric_name)
            else:
                metric_seg_scores = new_metric2seg_scores[metric_name]

            if not metric_seg_scores:  # Metric not available at this level.
                continue

            filtered_metric_seg_scores = {
                sys: [metric_seg_scores[sys][seg_idx] for seg_idx in filtered_seg_ids]
                for sys in sys_names
            }
            metric_sys_scores, metric_sys_scores_all = {
                sys: [
                    sum(filtered_metric_seg_scores[sys])
                    / len(filtered_metric_seg_scores[sys])
                ]
                for sys in sys_names
            }, {
                sys: [
                    sum(
                        seg_score
                        for seg_score in metric_seg_scores[sys]
                        if seg_score is not None
                    )
                    / sum(
                        1
                        for seg_score in metric_seg_scores[sys]
                        if seg_score is not None
                    )
                ]
                for sys in sys_names
            }
            (
                seg_correlations[display_name],
                sys_correlations[display_name],
                sys_correlations_all[display_name],
            ) = (
                wmt22_eval_set.Correlation(
                    gold_filtered_seg_scores, filtered_metric_seg_scores, sys_names
                ),
                wmt22_eval_set.Correlation(
                    gold_sys_scores, metric_sys_scores, sys_names
                ),
                wmt22_eval_set.Correlation(
                    gold_sys_scores_all, metric_sys_scores_all, sys_names
                ),
            )

        psd, pvalue = stats.PermutationSigDiffParams(100, 0.02, 0.50), 0.05
        corrs_and_ranks, sig_matrix, draws_index, draws_list = data.CompareMetrics(
            seg_correlations,
            stats.KendallWithTiesOpt,
            "item",
            kwto_k,
            psd,
            pvalue,
            False,
            "pairs",
            parallel_file=None,
            sample_rate=1.0,
        )
        print("\n")
        print("Seg-Lev KendallWithTiesOpt:")
        data.PrintMetricComparison(
            corrs_and_ranks,
            sig_matrix,
            pvalue,
        )
        print("\n")

        for corr_fcn in [scipy.stats.pearsonr, scipy.stats.kendalltau]:
            for average_by in ["none", "item", "sys"]:
                (
                    corrs_and_ranks,
                    sig_matrix,
                    draws_index,
                    draws_list,
                ) = data.CompareMetrics(
                    seg_correlations,
                    corr_fcn,
                    average_by,
                    0,
                    psd,
                    pvalue,
                    False,
                    "scores",
                    parallel_file=None,
                )
                print("\n")
                print(f"Seg-Lev {corr_fcn.__name__} (average_by={average_by}):")
                data.PrintMetricComparison(
                    corrs_and_ranks,
                    sig_matrix,
                    pvalue,
                )
                print("\n")

        def print_PA_res(corrs: Dict[str, stats.Correlation], print_msg: str) -> None:
            # Compute metric PAs, ordered by decreasing correlation.
            corrs_and_ranks = dict()
            for m, c in corrs.items():
                corrs_and_ranks[m] = [
                    pairwise_acc(c.gold_scores, c.metric_scores)[0],
                    0,
                ]
            # Use metric name as secondary sort criterion to stabilize ties.
            corrs_and_ranks = dict(
                sorted(corrs_and_ranks.items(), key=lambda x: (-x[1][0], x[0]))
            )

            # Compute significance matrix and determine ranks.
            sig_matrix, draws_index, draws_list = ComputeSigMatrix(
                corrs,
                corrs_and_ranks,
                pairwise_acc,
                "none",
                0,
                psd,
                False,
                "scores",
                None,
            )
            ranks = AssignRanks(sig_matrix, pvalue)
            for i, m in enumerate(corrs_and_ranks):
                corrs_and_ranks[m][1] = ranks[i]
            print("\n")
            print(print_msg)
            pp(corrs_and_ranks)
            print("\n")

        print_PA_res(sys_correlations, "Sys-Lev PA:")
        print_PA_res(
            sys_correlations_all, "Sys-Lev PA (with all available seg scores):"
        )

        (
            corrs_and_ranks,
            sig_matrix,
            draws_index,
            draws_list,
        ) = data.CompareMetricsWithPairwiseConfidenceError(
            seg_correlations,
            spa_k,
            psd,
            pvalue,
            False,
            "scores",
            parallel_file=None,
        )
        print("\n")
        print("Sys-Lev SPA:")
        data.PrintMetricComparison(
            corrs_and_ranks,
            sig_matrix,
            pvalue,
        )
        print("\n")

        corrs_and_ranks, sig_matrix, draws_index, draws_list = data.CompareMetrics(
            sys_correlations,
            scipy.stats.pearsonr,
            "none",
            0,
            psd,
            pvalue,
            False,
            "scores",
            parallel_file=None,
        )

        print("\n")
        print(f"Sys-Lev Pearson:")
        data.PrintMetricComparison(
            corrs_and_ranks,
            sig_matrix,
            pvalue,
        )
        print("\n\n\n")

    print("\n\n\n")


def run_mt_meta_eval_command() -> None:
    """Command to run MT meta-evaluation with human judgment scores included."""
    parser = read_arguments()
    args = parser.parse_args()

    if args.wmt_year == "wmt24":
        measure_wmt24_agreement_with_esa(args.kwto_k, args.spa_k)
    elif args.wmt_year == "wmt20":
        measure_wmt20_agreement_raters_ens(
            args.lp,
            args.new_human_annotations_dir,
            args.kwto_k,
            args.spa_k,
            args.new_metrics_path,
        )
    elif args.wmt_year == "wmt22":
        measure_wmt22_agreement_raters_ens(
            args.new_human_annotations_dir,
            args.k,
            args.wmt22_lp,
            args.new_metrics_path,
            args.gold_name,
        )
    else:
        new_raters_ens_name2seg_scores = dict()
        for new_raters_ens_name in ["mqm-col1", "mqm-col2", "mqm-col3"]:
            if new_raters_ens_name == args.gold_name:
                continue

            with open(
                args.new_human_annotations_dir
                / "wmt23"
                / args.lp
                / f"{new_raters_ens_name}.pickle",
                "rb",
            ) as handle:
                new_raters_ens_name2seg_scores[new_raters_ens_name] = pickle.load(
                    handle
                )
                new_raters_ens_name2seg_scores[new_raters_ens_name].pop("synthetic_ref")

        if "mqm-col" in args.gold_name:
            with open(
                args.new_human_annotations_dir
                / "wmt23"
                / args.lp
                / f"{args.gold_name}.pickle",
                "rb",
            ) as handle:
                wmt23_mqm_annotation = pickle.load(handle)
                wmt23_mqm_annotation.pop("synthetic_ref")
        else:
            wmt23_mqm_annotation = new_raters_ens_name2seg_scores["mqm-col1"]

        if args.lp == "en-de" and not args.use_only_google_mqm_and_da_sqm:
            new_mqm_annotation, esa_1_annotation, esa_2_annotation = (
                data.ReadScoreFile(
                    args.new_human_annotations_dir / NEW_MQM_ANNOTATION_FILE
                ),
                data.ReadScoreFile(
                    args.new_human_annotations_dir / ESA_1_ANNOTATION_FILE
                ),
                data.ReadScoreFile(
                    args.new_human_annotations_dir / ESA_2_ANNOTATION_FILE
                ),
            )
            n_annotated_segs = sum(
                1
                for score in next(iter(new_mqm_annotation.values()))
                if score is not None
            )
            print("\n")
            print(
                f"# annotated segments in new human annotations = {n_annotated_segs}."
            )

        wmt23_eval_set = data.EvalSet("wmt23", args.lp, True)
        wmt23_da_sqm_annotation = wmt23_eval_set.Scores("seg", "da-sqm")

        if args.lp == "en-de" and not args.use_only_google_mqm_and_da_sqm:
            # The three Google MQM annotations are all structured in the same way.
            assert sorted(list(new_mqm_annotation)) == sorted(
                list(esa_1_annotation)
            ) == sorted(list(esa_2_annotation)) == sorted(
                list(wmt23_mqm_annotation)
            ) == sorted(
                list(wmt23_da_sqm_annotation)
            ) and len(
                next(iter(new_mqm_annotation.values()))
            ) == len(
                next(iter(esa_1_annotation.values()))
            ) == len(
                next(iter(esa_2_annotation.values()))
            ) == len(
                next(iter(wmt23_mqm_annotation.values()))
            ) == len(
                next(iter(wmt23_da_sqm_annotation.values()))
            )
        else:
            assert sorted(list(wmt23_mqm_annotation)) == sorted(
                list(wmt23_da_sqm_annotation)
            ) and len(next(iter(wmt23_mqm_annotation.values()))) == len(
                next(iter(wmt23_da_sqm_annotation.values()))
            )

        sys_names = set(wmt23_mqm_annotation.keys())
        sys_names.discard(wmt23_eval_set.std_ref)  # For ref-based metrics
        sys_names = sorted(sys_names)  # Use a sorted list for consistent order

        print(f"# MT Systems = {len(sys_names)}.")

        filtered_seg_ids = []
        first_mt_sys = next(iter(sys_names))
        for seg_idx in range(len(wmt23_mqm_annotation[first_mt_sys])):
            if args.lp == "en-de" and not args.use_only_google_mqm_and_da_sqm:
                # The three Google MQM annotations are all structured in the same way.
                if (
                    wmt23_mqm_annotation[first_mt_sys][seg_idx] is not None
                    and wmt23_da_sqm_annotation[first_mt_sys][seg_idx] is not None
                    and new_mqm_annotation[first_mt_sys][seg_idx] is not None
                    and esa_1_annotation[first_mt_sys][seg_idx] is not None
                    and esa_2_annotation[first_mt_sys][seg_idx] is not None
                ):
                    filtered_seg_ids.append(seg_idx)
            else:
                if (
                    wmt23_mqm_annotation[first_mt_sys][seg_idx] is not None
                    and wmt23_da_sqm_annotation[first_mt_sys][seg_idx] is not None
                ):
                    filtered_seg_ids.append(seg_idx)
        print(
            f"# segs annotated in all human annotations (including wmt23 ones) = {len(filtered_seg_ids)}."
        )

        human_annotation2seg_scores = (
            {
                args.gold_name: wmt23_mqm_annotation,
                "da-sqm": wmt23_da_sqm_annotation,
                "new_mqm": new_mqm_annotation,
                "esa_1": esa_1_annotation,
                "esa_2": esa_2_annotation,
            }
            if args.lp == "en-de" and not args.use_only_google_mqm_and_da_sqm
            else {
                args.gold_name: wmt23_mqm_annotation,
                "da-sqm": wmt23_da_sqm_annotation,
            }
        )
        for new_mqm_col_name, annotations in new_raters_ens_name2seg_scores.items():
            human_annotation2seg_scores[new_mqm_col_name] = annotations

        gold_scores = human_annotation2seg_scores.get(args.gold_name)
        if gold_scores is None:
            wmt23_gold_metric_name = f"{args.gold_name}-refA"
            if wmt23_gold_metric_name not in wmt23_eval_set.metric_names:
                wmt23_gold_metric_name = f"{args.gold_name}-src"
            if wmt23_gold_metric_name not in wmt23_eval_set.metric_names:
                raise ValueError(
                    f"Gold name {args.gold_name} corresponds neither to a metric nor to human scores!"
                )
            gold_scores = wmt23_eval_set.Scores("seg", wmt23_gold_metric_name)

        print(
            f"# segs annotated in gold scores ({args.gold_name}) = "
            f"{sum(1 for seg_score in next(iter(gold_scores.values())) if seg_score is not None)}."
        )
        print("\n")

        gold_filtered_seg_scores = {
            sys: [gold_scores[sys][seg_idx] for seg_idx in filtered_seg_ids]
            for sys in sys_names
        }

        new_metric2seg_scores = get_new_metric2seg_scores(
            args.new_metrics_path, "wmt23", args.lp
        )

        if args.logging_file is not None:
            metrics_to_log = [
                "mqm",
                "da-sqm",
                "new_mqm",
                "esa_1",
                "esa_2",
                "XCOMET-XL-refA",
                "MetricX-23-QE-src",
                "MetricX-23-refA",
                "GEMBA-MQM-src",
                "CometKiwi-src",
                "COMET-refA",
                "DIFF-REGRESSION-METRIC-MQM-LIN-MBR",
                "DIFF-REGRESSION-METRIC-DA-LIN-MBR",
            ]
            with open(args.logging_file, "w") as file:
                for seg_idx in filtered_seg_ids:
                    for sys in sys_names:
                        file.write(f"SEGMENT INDEX = {seg_idx}.\n")
                        file.write(f"SOURCE TEXT: {wmt23_eval_set.src[seg_idx]}.\n")
                        file.write(
                            f"CANDIDATE TRANSLATION: {wmt23_eval_set.sys_outputs[sys][seg_idx]}.\n"
                        )
                        file.write(
                            f"REFERENCE TRANSLATION: {wmt23_eval_set.all_refs[wmt23_eval_set.std_ref][seg_idx]}.\n"
                        )
                        file.write("\n")
                        for metric_name in metrics_to_log:
                            if metric_name in human_annotation2seg_scores:
                                file.write(
                                    f"{metric_name}\t{human_annotation2seg_scores[metric_name][sys][seg_idx]}\n"
                                )
                            elif metric_name in new_metric2seg_scores:
                                file.write(
                                    f"{metric_name}\t{new_metric2seg_scores[metric_name][sys][seg_idx]}\n"
                                )
                            else:
                                file.write(
                                    f"{metric_name}\t{wmt23_eval_set.Scores('seg', metric_name)[sys][seg_idx]}\n"
                                )
                        file.write("\n\n")
                    file.write("\n\n\n")

        main_refs = {wmt23_eval_set.std_ref}
        seg_correlations = dict()
        for metric_name in (
            wmt23_eval_set.metric_names
            | set(human_annotation2seg_scores)
            | set(new_metric2seg_scores)
        ):
            if (
                metric_name == args.gold_name
                or metric_name == f"{args.gold_name}-refA"
                or metric_name == f"{args.gold_name}-src"
            ):
                continue

            if metric_name in wmt23_eval_set.metric_names:
                base_name, metric_refs = wmt23_eval_set.ParseMetricName(metric_name)
                if (
                    (
                        base_name not in wmt23_eval_set.primary_metrics
                        and base_name != "CometKiwi-XL"
                        and base_name != "CometKiwi-XXL"
                    )
                    or base_name == "MetricX-23"
                    or base_name == "MetricX-23-QE"
                ):
                    continue
                if not metric_refs.issubset(main_refs):
                    continue
                display_name = wmt23_eval_set.DisplayName(metric_name, "spreadsheet")
            else:
                display_name = metric_name

            if metric_name in new_metric2seg_scores:
                metric_seg_scores = new_metric2seg_scores[metric_name]
            else:
                metric_seg_scores = (
                    human_annotation2seg_scores[metric_name]
                    if metric_name in human_annotation2seg_scores
                    else wmt23_eval_set.Scores("seg", metric_name)
                )
            if not metric_seg_scores:  # Metric not available at this level.
                continue
            filtered_metric_seg_scores = {
                sys: [metric_seg_scores[sys][seg_idx] for seg_idx in filtered_seg_ids]
                for sys in sys_names
            }
            seg_correlations[display_name] = wmt23_eval_set.Correlation(
                gold_filtered_seg_scores, filtered_metric_seg_scores, sys_names
            )

        psd, pvalue = stats.PermutationSigDiffParams(100, 0.02, 0.50), 0.05
        corrs_and_ranks, sig_matrix, draws_index, draws_list = data.CompareMetrics(
            seg_correlations,
            stats.KendallWithTiesOpt,
            "item",
            args.kwto_k,
            psd,
            pvalue,
            False,
            "pairs",
            parallel_file=None,
            sample_rate=1.0,
        )
        print("\n")
        print("Seg-Lev KendallWithTiesOpt:")
        data.PrintMetricComparison(
            corrs_and_ranks,
            sig_matrix,
            pvalue,
        )
        print("\n")

        for corr_fcn in [scipy.stats.pearsonr, scipy.stats.kendalltau]:
            for average_by in ["none", "item", "sys"]:
                (
                    corrs_and_ranks,
                    sig_matrix,
                    draws_index,
                    draws_list,
                ) = data.CompareMetrics(
                    seg_correlations,
                    corr_fcn,
                    average_by,
                    0,
                    psd,
                    pvalue,
                    False,
                    "scores",
                    parallel_file=None,
                )
                print("\n")
                print(f"Seg-Lev {corr_fcn.__name__} (average_by={average_by}):")
                data.PrintMetricComparison(
                    corrs_and_ranks,
                    sig_matrix,
                    pvalue,
                )
                print("\n")

        (
            corrs_and_ranks,
            sig_matrix,
            draws_index,
            draws_list,
        ) = data.CompareMetricsWithPairwiseConfidenceError(
            seg_correlations,
            args.spa_k,
            psd,
            pvalue,
            False,
            "scores",
            parallel_file=None,
        )
        print("\n")
        print("Sys-Lev SPA:")
        data.PrintMetricComparison(
            corrs_and_ranks,
            sig_matrix,
            pvalue,
        )
        print("\n")


if __name__ == "__main__":
    run_mt_meta_eval_command()
