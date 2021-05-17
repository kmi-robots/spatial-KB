from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import math

def eval_singlemodel(ReasonerObj,eval_d,method, K=1):
    if K==1:
        # eval top-1 of each ranking
        y_pred = ReasonerObj.predictions[:, 0, 0].astype('int').astype('str').tolist()
        y_true = ReasonerObj.labels
        global_acc = accuracy_score(y_true, y_pred)
        print(classification_report(y_true, y_pred, digits=4))
        print(global_acc)
        Pu,Ru, F1u, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
        Pw, Rw, F1w, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        for k,metr in [('accuracy',global_acc),('Punweighted',Pu),('Runweighted',Ru),('F1unweighted',F1u), ('Pweighted',Pw),('Rweighted',Rw),('F1weighted',F1w)]:
            try:eval_d[method][k].append(metr)
            except KeyError:
                eval_d[method][k] =[]
                eval_d[method][k].append(metr)
        return eval_d
    else:#eval quality of top-K ranking
        return eval_ranking(ReasonerObj, K, eval_d,method)

def eval_ranking(ReasonerObj,K,eval_d,method):
    """
    Prints mean Precision@K, mean nDCG@K and hit ratio @ K
    """
    y_pred = ReasonerObj.predictions[:, :K, 0].astype('int').astype('str').tolist()
    y_true = ReasonerObj.labels
    precisions = []
    ndcgs = []
    hits = 0
    IDCG = 0.  # Ideal DCG
    for n in range(2, K + 2):
        IDCG += float(1 / math.log(n, 2))
    for z, (ranking, gt_label) in enumerate(zip(y_pred, y_true)):
        pred_rank = [1 if r == gt_label else 0 for r in ranking]
        dis_scores = [float(1 / math.log(i + 2, 2)) for i, r in enumerate(ranking) if r == gt_label]
        no_hits = pred_rank.count(1)
        precisions.append(float(no_hits / K))
        if no_hits >= 1:
            hits += 1  # increment if at least one hit in the ranking
            nDCG = float(sum(dis_scores) / IDCG)  # compute nDCG for ranking
            ndcgs.append(nDCG)
    print("Avg ranking Precision@%i: %f " % (K, float(sum(precisions) / len(precisions))))
    print("Avg Normalised DCG @%i: %f" % (K, float(sum(ndcgs) / len(precisions))))
    print("Hit ratio @%i: %f" % (K, float(hits / len(precisions))))

    for k,metr in [('meanP@K', float(sum(precisions) / len(precisions))), ('meannDCG@K', float(sum(ndcgs) / len(precisions))) \
        , ('hitratio', float(hits / len(precisions)))]:
        try: eval_d[method][k].append(metr)
        except KeyError:
            eval_d[method][k] = []
            eval_d[method][k].append(metr)
    return eval_d
