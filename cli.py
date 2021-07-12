"""Command Line Interface"""

import sys
import argparse
from KB import KnowledgeBase
from SpatialDB import SpatialDB
from object_reasoner import ObjectReasoner
from sklearn.model_selection import StratifiedKFold
import copy
import os
import json
import time
from crossval_sampling import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_data', help='Base path to raw spatial data', default='./data')
    parser.add_argument('--path_to_pred', help='Base path to ML predictions', default='./data/logged-predictions')
    parser.add_argument('--dbname', help='Name for PostGRE SQL spatial database', default='gis_database')
    parser.add_argument('--scenario', nargs='?',
                        choices=['best', 'worst', 'selected'],
                        default="selected",
                        help="Hybrid correction scenario."
                             "best: correct all which need correction, based on ground truth"
                             "worst: correct all ML predictions indiscriminately"
                             "selected: apply selection based on ML confidence")
    parser.add_argument('--baseline', nargs='?',
                        choices=['k-net', 'n-net', 'two-stage'],
                        default="k-net",
                        help="Baseline ML method to retrieve predictions from.")
    parser.add_argument('--set', default='lab', choices=['lab'], help="Chosen dataset. Only Lab set supported at the moment.")
    parser.add_argument('--nsplits', type=int, default=7,
                        help="Number of folds for Kfold cross validation. Defaults to 7.")
    parser.add_argument('--rm', nargs='?',
                        choices=['spatial', 'size', 'size_spatial'],
                        default="spatial",
                        help="Reasoning method to use after applying ML baseline ")
    parser.add_argument('--ql', nargs='?',
                        choices=['gold', 'ML', 'hybrid', 'sizevalidated'],
                        default="gold",
                        help="Which labels to use for nearby objects when validating QSRs"
                             "gold: ground truth labels for all objects but the one to predict"
                             "ML: ML predicted labels"
                            "hybrid: ML predicted if above confidence threshold, size-validated otherwise")

    args = parser.parse_args()

    overall_res = {m: {} for m in ['MLonly', args.rm]} # dictionary of ablations under test
    overall_res[args.rm]['processingTime'] = []
    KB = KnowledgeBase(args)

    print("Init VG data and batch-compute 3D geometries for spatial DB.. ")
    start = time.time()
    spatialDB = SpatialDB(KB, args)
    spatialDB.db_session()
    print("Took % fseconds." % float(time.time() - start))

    reasoner = ObjectReasoner(args,spatialDB.ids_tbprocessed)

    # Nfold stratified cross-validation for test results
    # subsample test set to devote a small portion to param tuning
    skf = StratifiedKFold(n_splits=args.nsplits)
    allclasses = reasoner.mapper.values()
    for test1_index, test2_index in skf.split(reasoner.predictions, reasoner.labels):
        sampled_reasoner = subsample(copy.deepcopy(reasoner), test1_index, test2_index, allclasses)
        overall_res = sampled_reasoner.run(overall_res,spatialDB, KB.size_auto)

    """Compute mean and stdev of eval results across test runs
     and output eval report as json file"""
    avg_res = {}
    for method, subdict in overall_res.items():
        print("---Cross-fold eval results for method %s----" % method)
        avg_res[method] = {}
        for metric_name, metric_array in subdict.items():
            meanm = statistics.mean(metric_array)
            print("Mean %s: %f" % (metric_name, meanm))
            stdm = statistics.stdev(metric_array)
            print("Stdev %s: %f" % (metric_name, stdm))
            avg_res[method][metric_name] = str(meanm) + "+-" + str (stdm)
    with open(os.path.join(args.path_to_data, 'eval_results_%s_%s_%s' % (args.baseline, args.set, args.rm)), 'w') as jout:
        json.dump(avg_res, jout)
    return 0

if __name__ == '__main__':
    sys.exit(main())
