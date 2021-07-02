"""
Autom Sampling & Param Tuning module

Based on kfold cross val indices computed through scikit-learn,
splits the predictions in two sets for each run. The larger split is used at test set
(i.e., for evaluation); the smaller set is used to estimate confidence thresholds to decide which ML
predictions to correct.
"""
import statistics

def subsample(Reasonerobj, test1_index, test2_index, allclasses):
    # generate test splits so that all contain the same distribution of classes, or as close as possible
    # retain larger split as test set, smaller split is for tuning the epsilon params
    Reasonerobj.labels = [lbl for i, lbl in enumerate(Reasonerobj.labels) if i in test1_index]
    Reasonerobj.fnames = [nm for i, nm in enumerate(Reasonerobj.fnames) if i in test1_index]
    Reasonerobj.crops = [(c,d) for i, (c,d) in enumerate(Reasonerobj.crops) if i in test1_index]

    fullpredictions, fullpredictions2 = Reasonerobj.predictions.copy(), Reasonerobj.predictions.copy()
    Reasonerobj.predictions = fullpredictions[test1_index]
    predictions2 = fullpredictions2[test2_index]
    Reasonerobj.epsilon_set = estimate_epsilon(predictions2, allclasses) # None because only one ML baseline here
    return Reasonerobj

def estimate_epsilon(subsample_preds_algo1, classlist, subsample_preds_algo2=None):
    """
    Input:  - predictions on test subset by ML algorithm 1
            - list of N classes
            - OPTIONAL: predictions on test subset by ML algorithm 2, if a second one is leveraged
    Output: a 3xN list, with values of the epsilon param for each class and for each algorithm
            + indication of the class label those value refer to
    """
    min_classwise1, min_classwise2 = [],[]
    for classl in classlist:
        min_predwise1,min_predwise2 = [],[]
        if subsample_preds_algo2 is None: #only one baseline algorithm
            for pred in subsample_preds_algo1:
                try:
                    min_predwise1.append(min([score for l_, score in pred if l_ == classl]))
                except: continue
        else:
            for pred,pred2 in list(zip(subsample_preds_algo1,subsample_preds_algo2)):
                try:
                    min_predwise1.append(min([score for l_,score in pred if l_ ==classl]))
                    min_predwise2.append(min([score for l_,score in pred2 if l_ ==classl]))
                except: continue
            min_classwise2.append(min(min_predwise2))
        min_classwise1.append(min(min_predwise1))
    if subsample_preds_algo2 is None: epsilon_set= (statistics.mean(min_classwise1),None)
    else: epsilon_set = (statistics.mean(min_classwise1),statistics.mean(min_classwise2))
    return epsilon_set
