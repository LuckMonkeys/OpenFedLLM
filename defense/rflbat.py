import torch
import logging
import os
import numpy as np
import sklearn.metrics.pairwise as smp
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

logger = logging.getLogger('logger')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
logging.getLogger('matplotlib.font_manager').disabled = True
import math
import random
from .base import Aggregate

def gap_statistics(data, num_sampling, K_max, n):
        num_cluster = 0
        data = np.reshape(data, (data.shape[0], -1))
        # Linear transformation
        data_c = np.ndarray(shape=data.shape)
        for i in range(data.shape[1]):
            data_c[:,i] = (data[:,i] - np.min(data[:,i])) / \
                 (np.max(data[:,i]) - np.min(data[:,i]))
        gap = []
        s = []
        for k in range(1, K_max + 1):
            k_means = KMeans(n_clusters=k, init='k-means++').fit(data_c)
            predicts = (k_means.labels_).tolist()
            centers = k_means.cluster_centers_
            # v_k = 0
            v_k = 1e-5
            for i in range(k):
                for predict in predicts:
                    if predict == i:
                        v_k += np.linalg.norm(centers[i] - \
                                 data_c[predicts.index(predict)])
            # perform clustering on fake data
            v_kb = []
            for _ in range(num_sampling):
                data_fake = []
                for i in range(n):
                    temp = np.ndarray(shape=(1,data.shape[1]))
                    for j in range(data.shape[1]):
                        temp[0][j] = random.uniform(0,1)
                    data_fake.append(temp[0])
                k_means_b = KMeans(n_clusters=k, init='k-means++').fit(data_fake)
                predicts_b = (k_means_b.labels_).tolist()
                centers_b = k_means_b.cluster_centers_
                # v_kb_i = 0
                v_kb_i = 1e-5
                for i in range(k):
                    for predict in predicts_b:
                        if predict == i:
                            v_kb_i += np.linalg.norm(centers_b[i] - \
                                    data_fake[predicts_b.index(predict)])
                v_kb.append(v_kb_i)
            # gap for k
            v = 0
            for v_kb_i in v_kb:
                # print(v_kb_i)
                # print(math.log(v_kb_i))
                v += math.log(v_kb_i)
            v /= num_sampling
            gap.append(v - math.log(v_k))
            sd = 0
            for v_kb_i in v_kb:
                sd += (math.log(v_kb_i) - v)**2
            sd = math.sqrt(sd / num_sampling)
            s.append(sd * math.sqrt((1 + num_sampling) / num_sampling))
        # select smallest k
        try:
            for k in range(1, K_max + 1):
                print(gap[k - 1] - gap[k] + s[k - 1])
                if k == K_max:
                    num_cluster = K_max
                    break
                if gap[k - 1] - gap[k] + s[k - 1] > 0:
                    num_cluster = k
                    break
        except Exception as e:
            print(f">>>>>>>>>>>>>>>Error Occur: k_max: {K_max}, gap: {gap}, s:{s}")
            print(e)

            num_cluster = 1
            # breakpoint()
        return num_cluster




class RFLBAT(Aggregate):

    def __init__(self, eps1=10, eps2=4, num_adv=1, folder_path = "./figs", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.eps1 = eps1
        self.eps2 = eps2
        self.num_adv = 1
        self.folder_path = folder_path

        self.n_clients = None
        self.n_features = None
        self.n_classes = None

    def exec(self, inputs, clients_this_round, num_dps, global_dict, key_order, device, *args, **kwargs):
        
        if not isinstance(inputs, torch.Tensor):
            delta = torch.stack(inputs)
        else:
            delta = inputs

        dataAll = delta.cpu()
        num_total = delta.shape[0]
        
        pca = PCA(n_components=2) #instantiate
        pca = pca.fit(dataAll)
        X_dr = pca.transform(dataAll)

        # Save figure
        plt.figure()
        plt.scatter(X_dr[0:self.num_adv,0], 
            X_dr[0:self.num_adv,1], c='red')
        plt.scatter(X_dr[self.num_adv:num_total,0], 
            X_dr[self.num_adv:num_total,1], c='green')
        # plt.scatter(X_dr[self.params.fl_total_participants:,0], X_dr[self.params.fl_total_participants:,1], c='black')
        folderpath = '{0}/RFLBAT'.format(self.folder_path)
        if not os.path.exists(folderpath):
            os.makedirs(folderpath)
        figname = '{0}/PCA_E{1}.jpg'.format(folderpath, round)
        plt.savefig(figname)
        logger.info(f"RFLBAT: Save figure {figname}.")

        # Compute sum eu distance
        eu_list = []
        for i in range(len(X_dr)):
            eu_sum = 0
            for j in range(len(X_dr)):
                if i==j:
                    continue
                eu_sum += np.linalg.norm(X_dr[i]-X_dr[j])
            eu_list.append(eu_sum)
        accept = []
        x1 = []
        for i in range(len(eu_list)):
            if eu_list[i] < self.eps1 * np.median(eu_list):
                accept.append(i)
                x1 = np.append(x1, X_dr[i])
            else:
                logger.info("RFLBAT: discard update {0}".format(i))
        x1 = np.reshape(x1, (-1, X_dr.shape[1]))
        # breakpoint()
        num_clusters = gap_statistics(x1, \
            num_sampling=5, K_max=len(x1)//2+1, n=len(x1))
        logger.info("RFLBAT: the number of clusters is {0}"\
            .format(num_clusters))
        k_means = KMeans(n_clusters=num_clusters, \
            init='k-means++').fit(x1)
        predicts = k_means.labels_
        logger.info(f"RFLBAT: cluster label {predicts}")

        # select the most suitable cluster
        v_med = []
        for i in range(num_clusters):
            temp = []
            for j in range(len(predicts)):
                if predicts[j] == i:
                    temp.append(dataAll[accept[j]])
            if len(temp) <= 1:
                v_med.append(1)
                continue
            v_med.append(np.median(np.average(smp\
                .cosine_similarity(temp), axis=1)))
        temp = []
        for i in range(len(accept)):
            if predicts[i] == v_med.index(min(v_med)):
                temp.append(accept[i])
        accept = temp

        # compute eu list again to exclude outliers
        temp = []
        for i in accept:
            temp.append(X_dr[i])
        X_dr = temp
        eu_list = []
        for i in range(len(X_dr)):
            eu_sum = 0
            for j in range(len(X_dr)):
                if i==j:
                    continue
                eu_sum += np.linalg.norm(X_dr[i]-X_dr[j])
            eu_list.append(eu_sum)
        temp = []
        for i in range(len(eu_list)):
            if eu_list[i] < self.eps2 * np.median(eu_list):
                temp.append(accept[i])
            else:
                logger.info("RFLBAT: discard update {0}"\
                    .format(i))
        accept = temp
        logger.info("RFLBAT: the final clients accepted are {0}"\
            .format(accept))
        
        
        selected_num_dps = []
        for idx, i in enumerate(clients_this_round):
            if idx in accept:
                selected_num_dps.append(num_dps[i])
            else:
                selected_num_dps.append(0)
                
        n_freq = torch.tensor([snd/sum(selected_num_dps) for snd in selected_num_dps]).to(device)
        aggregated_input = torch.sum(delta * n_freq[:, None], dim=0)
        
        return self.server_step(global_dict, aggregated_input, key_order)
        
