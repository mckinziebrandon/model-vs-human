# /!usr/bin/env python3

"""
Plotting functionality
"""

import copy
import logging
import os
from os.path import join as pjoin

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import analyses as a
from . import decision_makers as dm
from .. import constants as consts
from ..helper import plotting_helper as ph
from ..utils import load_dataset

logger = logging.getLogger(__name__)

# global default boundary settings for thin gray transparent
# boundaries to avoid not being able to see the difference
# between two partially overlapping datapoints of the same color:
PLOTTING_EDGE_COLOR = (0.3, 0.3, 0.3, 0.3)
PLOTTING_EDGE_WIDTH = 0.02

METRICS = {"OOD accuracy": (a.SixteenClassAccuracy(), "16-class-accuracy"),
           "accuracy difference": (a.SixteenClassAccuracyDifference(),
                                   "16-class-accuracy-difference"),
           "observed consistency": (a.ErrorConsistency(), "observed-consistency"),
           "error consistency": (a.ErrorConsistency(), "error-consistency")}

# exclusion criteria:
# - not OOD: control condition without manipulation (e.g. 100% contrast)
# - mean human accuracy < 0.2 (error consistency etc. not meaningful)
EXCLUDE_CONDITIONS = {
    "colour": ["cr"],
    "contrast": ["c100", "c03", "c01"],
    "high-pass": ["inf", "0.55", "0.45", "0.4"],
    "low-pass": ["0", "15", "40"],
    "phase-scrambling": ["0", "150", "180"],
    "power-equalisation": ["0"],
    "false-colour": ["True"],
    "rotation": ["0"],
    "eidolonI": ["1-10-10", "64-10-10", "128-10-10"],
    "eidolonII": ["1-3-10", "32-3-10", "64-3-10", "128-3-10"],
    "eidolonIII": ["1-0-10", "16-0-10", "32-0-10", "64-0-10", "128-0-10"],
    "uniform-noise": ["0.0", "0.6", "0.9"]
}

EXCLUDE = True


def get_datasets(dataset_names, *args, **kwargs):
    dataset_list = []
    for dataset in dataset_names:
        dataset = load_dataset(dataset, *args, **kwargs)
        dataset_list.extend(dataset) if isinstance(dataset, list) else dataset_list.append(dataset)
    return dataset_list


def get_dataset_names(plot_type):
    """Given plot_type, return suitable dataset(s).

    In this regard, 'suitable' means:
    - valid plot_type <-> dataset combination
    - data is available
    """

    dataset_names = []
    dataset_candidates = consts.PLOT_TYPE_TO_DATASET_MAPPING[plot_type]

    for candidate in dataset_candidates:
        if os.path.exists(pjoin(consts.RAW_DATA_DIR, candidate)):
            dataset_names.append(candidate)

    if len(dataset_names) == 0:
        raise ValueError("No data found for the specified plot_types.")

    return dataset_names


def get_permutations(elements):
    """Return permutation of elements.

    Return value: list of tuples, where tuples are
    unique combinations of elements
    """

    permutations = []
    for i, elem1 in enumerate(elements):
        for j, elem2 in enumerate(elements):
            if i < j:
                permutations.append((elem1, elem2))
    return permutations


def exclude_conditions(dataset):
    dataset = copy.deepcopy(dataset)
    if len(dataset.experiments) > 0:
        assert dataset.name in EXCLUDE_CONDITIONS.keys()
        for c in EXCLUDE_CONDITIONS[dataset.name]:
            assert len(dataset.experiments) == 1
            assert c in dataset.experiments[0].data_conditions, f"{c} not found for {dataset.name}"
            idx = dataset.experiments[0].data_conditions.index(c)
            dataset.experiments[0].data_conditions.remove(c)
            del dataset.experiments[0].plotting_conditions[idx]
            #print(f"Dataset {dataset.name}: removing condition {c}")
    return dataset


def log(plot_type, dataset_name):
    """Print logging info for plotting to console"""

    logging_info = f"Plotting {plot_type} for dataset {dataset_name}"
    logger.info(logging_info)
    print(logging_info)


def get_human_and_CNN_subjects(subjects):
    """Split subjects into 2 lists: human, CNNs subjects."""

    assert type(subjects) is list
    human_subjects = []
    CNN_subjects = []
    for s in subjects:
        if s.startswith("subject-"):
            human_subjects.append(s)
        else:
            CNN_subjects.append(s)
    return human_subjects, CNN_subjects


def get_raw_matrix(dataset,
                   decision_maker_fun,
                   analysis,
                   value="error-consistency"):
    """Return NxN data frame of error consistencies."""

    df = ph.get_experimental_data(dataset)
    decision_makers = decision_maker_fun(df)
    subjects = dm.get_individual_decision_makers(decision_makers)

    num_subjects = len(subjects)
    matrix = np.ones([num_subjects, num_subjects])
    for i in tqdm(range(num_subjects)):
        s1 = subjects[i]
        df1 = df.loc[(df["subj"] == s1)]
        for j in range(i, num_subjects):
            s2 = subjects[j]
            df2 = df.loc[(df["subj"] == s2)]
            a = analysis.analysis(df1, df2)[value]
            matrix[i, j] = a
            matrix[j, i] = a

    plotting_names = []
    colors = []
    for s in subjects:
        attr = dm.decision_maker_to_attributes(s, decision_makers)
        if s.startswith("subject-"):
            plotting_names.append(s)
        else:
            plotting_names.append(attr["plotting_name"])
        if attr["color"] == (230 / 255.0, 230 / 255.0, 230 / 255.0):
            # supervised models: too bright for error consistency matrices
            colors.append((150 / 255.0, 150 / 255.0, 150 / 255.0))
        else:
            colors.append(attr["color"])

    assert len(colors) == matrix.shape[0] == matrix.shape[1]

    return {"matrix": pd.DataFrame(data=matrix,
                                   columns=plotting_names,
                                   index=plotting_names),
            "colors": colors}


def plotting_names_to_data_subjects(plotting_names,
                                    decision_makers):
    subjects = []
    plotting_set = set()
    for d in decision_makers:
        if d.plotting_name in plotting_names:
            subjects += d.decision_makers
            plotting_set.add(d.plotting_name)
    missing_subjects = plotting_set.symmetric_difference(set(plotting_names))
    if len(missing_subjects) > 0:
        print("Missing subjects: ")
        print(missing_subjects)
        raise ValueError("subjects missing")
    return subjects


def get_mean_over_datasets(colname,
                           metric_fun,
                           metric_name,
                           datasets,
                           decision_maker_fun):
    """Compute the mean result of metric_fun applied to datasets.

    Returns data frame with columns as follows:
    <plotting_name> (name of decision maker)
    <colname> (name of numerical column with metric results)
    <color> (plotting color)
    """

    result_df = pd.DataFrame(columns=['model', 'plotting_name', 'dataset', colname, 'color'])
    for d_orig in datasets:
        df_raw = ph.get_experimental_data(d_orig)
        if EXCLUDE:
            d = exclude_conditions(d_orig)
        else:
            d = d_orig
        for dmaker in decision_maker_fun(df_raw):
            if len(d.experiments) == 1:
                df_selection = df_raw.loc[(df_raw["subj"].isin(dmaker.decision_makers)) &
                                          (df_raw["condition"].isin(d.experiments[0].data_conditions))]
            elif len(d.experiments) == 0:
                df_selection = df_raw.loc[(df_raw["subj"].isin(dmaker.decision_makers))]
            else:
                raise ValueError("unknown")
            r1 = metric_fun.analysis(df=df_selection)
            result_df = result_df.append([{"plotting_name": dmaker.plotting_name,
                                           "dataset": d.name,
                                           colname: r1[metric_name],
                                           "color": dmaker.color}],
                                         ignore_index=True)

    # average over datasets
    result_df = result_df.groupby(['plotting_name', 'color'], as_index=False)[colname].mean()
    return result_df
