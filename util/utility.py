import numpy as np 
from tqdm import tqdm

# 計算cosine & pcc sim
# target: users or items
def get_sim(target, target_matrix):
    # init 目標相似度名單
    cos_dict = dict()
    pcc_dict = dict()
    for u in tqdm(range(len(target)), desc='caculator u & v similar'):
        # init 目標 u 跟 v 的相似度
        uv_cos = list()
        uv_pcc = list()
        for v in range(len(target)):
            if u != v:
                # 計算使用者u、v的cosine
                uv_cos.append(cos_sim(target_matrix[u], target_matrix[v]))
                # 計算使用者u、v的pcc
                uv_pcc.append(pcc_sim(target_matrix[u], target_matrix[v]))
            else:
                # 為了保持index不會跑掉，因此在自己的位置不做計算且補0
                uv_cos.append(0)
                uv_pcc.append(0)
        cos_dict[u] = uv_cos
        pcc_dict[u] = uv_pcc

    return cos_dict, pcc_dict


# 計算兩個向量的 cosine 相似度
def cos_sim(a, b):
    s = norm(a)*norm(b)
    if s == 0:
        return 0
    return np.inner(a,b) / s

# 計算兩個向量的 Pearson Correlation Coefficient 相似度
def pcc_sim(a, b):
    mean_a = non_zero_vec_mean(a)
    mean_b = non_zero_vec_mean(b)
    s = np.sqrt(np.sum(np.power((a-mean_a), 2))) * np.sqrt(np.sum(np.power((b-mean_b), 2)))
    if s == 0:
        return 0
    
    return np.inner(a-mean_a, b-mean_b) / s


# 計算bias
def get_bias(user_matrix, users, movies):
    # 計算bias
    bias_matrix = np.zeros((len(users), len(movies)))
    mean = get_u(user_matrix)

     # init u + bu
    for u in range(bias_matrix.shape[0]):
        bias_matrix[u] = get_ubias(user_matrix, u)

    # Bias = u + bu + bi
    for i in range(bias_matrix.shape[1]):
        bias_matrix[:,i] += get_ibias(user_matrix, i) - mean

    # 刪除原本沒有評分的bias
    for i in range(user_matrix.shape[0]):
        for j in range(user_matrix.shape[1]):
            if user_matrix[i,j] == 0:
                bias_matrix[i,j] = 0

    return bias_matrix


# 取得整體平均
def get_u(matrix):
    return np.mean(non_zero_mean(matrix)) 


# 針對向量非0地方做計算
def non_zero_mean(arr):
    exist = arr != 0
    total = arr.sum(axis = 1)
    exist_number = exist.sum(axis=1)

    return np.reshape(total/exist_number, (-1, 1))


# item bias
def get_ibias(user_matrix, i):
    return non_zero_vec_mean(user_matrix[:,i]) 

# user bias
def get_ubias(user_matrix, u):
    return non_zero_vec_mean(user_matrix[u]) 


# 針對單一向量非0地方做計算
def non_zero_vec_mean(vec):
    exist = vec != 0
    total = vec.sum()
    if total == 0: return 0
    exist_number = exist.sum()

    return total/exist_number

# square error
def se(t, p):
    return (p-t)**2

# rmse
def rmse(delta_list):
    if len(delta_list)==0:
        return None
    
    return (sum(delta_list)/len(delta_list))**0.5

# 計算向量長度
def norm(v):
    return np.sqrt(np.sum(np.power(v, 2)))

