# deal with 2d - 2d
import heapq


def recall_k(true, predict, k=10):
    recall = list()
    for i in range(len(true)):
        # TP
        tp = true_positive(true[i], predict[i])
        # 挑選預測數值前10大的item項的進行計算
        predict_max_index = map(predict.index, heapq.nlargest(k, predict))
        tp_k = true_positive(true[i][:k], predict[i][:k])
        # recall value
        recall.append(tp_k/tp)
    return sum(recall)/len(recall)

def true_positive(t, p):
    tp = 0
    for i in range(len(t)):
        if t[i] == p[i]:
            tp+=1
    
    return tp


if __name__ == "__main__":
    true = [1,2,3,4]
    predict = [1,3,2,4]
    re = recall_k(true, predict, 3)
