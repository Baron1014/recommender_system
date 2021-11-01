# deal with 2d - 2d
import heapq


def recall_k(true_array, predict_array, k=10):
    recall = list()
    for i in range(len(true_array)):
        total_rating_number = get_real_items_number(true_array[i])
        # 挑選預測數值前10大的item項的進行計算
        max_data = heapq.nlargest(k, enumerate(predict_array[i]), key=lambda x:x[1])
        predict_max_index, _ = zip(*max_data)
        top_real_interactive = caculate_toptp(true_array[i], list(predict_max_index))

        # recall value
        if total_rating_number != 0:
            recall.append(top_real_interactive/total_rating_number)
        else:
            print(f"User {i} no rating!")
            recall.append(0)

    return sum(recall)/len(recall)

# 取得總共有幾個互動過的items數量
def get_real_items_number(ground_truth):
    return len(ground_truth[ground_truth!=0])

# 計算前10個推薦是否真的有互動的個數
def caculate_toptp(true, top_index):
    tp = 0
    for ix in top_index:
        if true[ix] > 0:
            tp+= 1
    
    return tp




if __name__ == "__main__":
    true = [1,2,3,4]
    predict = [1,3,2,4]
    re = recall_k(true, predict, 3)
