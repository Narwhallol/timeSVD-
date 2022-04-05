import numpy as np

from dataset import load_dataset


class timeSVDpp:
    def __init__(self, mat, f=20):
        self.mat = np.array(mat)
        self.f = f  # 维度
        self.bi = {}
        self.bu = {}
        # self.bu_t = {}
        self.qi = {}
        self.pu = {}
        # self.pu_t = {}
        self.avg = 3.529860
        self.y = {}
        self.u_dict = {}  # R(u)
        self.tu = {}
        self.alpha_u = {}
        self.bin_num = 10
        # self.bpu_t_mun = 40
        self.bi_bin = {}
        self.min_time = 874724710
        self.span = 18561928  # 时间跨度(s)
        for i in range(self.mat.shape[0]):
            user_id = self.mat[i, 0]
            item_id = self.mat[i, 1]
            time = int(self.mat[i, 3])
            self.u_dict.setdefault(user_id, [])
            self.u_dict[user_id].append(item_id)
            self.bi.setdefault(item_id, 0)
            self.bu.setdefault(user_id, 0)
            # self.bu_t.setdefault(user_id, np.zeros(self.bpu_t_mun))
            self.alpha_u.setdefault(user_id, 0)
            self.tu.setdefault(user_id, [])
            self.tu[user_id].append(time)
            self.qi.setdefault(item_id, np.random.random((self.f, 1)) / 10 * np.sqrt(self.f))
            self.pu.setdefault(user_id, np.random.random((self.f, 1)) / 10 * np.sqrt(self.f))
            # self.pu_t.setdefault(user_id, np.zeros(self.bpu_t_mun))
            self.y.setdefault(item_id, np.zeros((self.f, 1)) + .1)
            self.bi_bin.setdefault(item_id, np.zeros(self.bin_num))

    def dev_u(self, user_id, time):
        tu = np.mean(self.tu[user_id])
        if time - tu < 0:
            return -pow(abs(time - tu), 0.015)  # beta = 0.015
        else:
            return pow(abs(time - tu), 0.015)

    def get_bin(self, time):  # 求bi,bin(t)
        bin_size = self.span / self.bin_num
        return np.minimum(self.bin_num - 1, int((time - self.min_time) / bin_size))

    # def get_bpu_t(self, time):
    #     size = self.span / self.bpu_t_mun
    #     return int((time - self.min_time) / size)

    def predict(self, user_id, item_id, time):
        self.bi.setdefault(item_id, 0)
        self.bu.setdefault(user_id, 0)
        # self.bu_t.setdefault(user_id, np.zeros(self.bpu_t_mun))  # 平均用40个参数来描述用户每天的偏见
        self.alpha_u.setdefault(user_id, 0)
        self.tu.setdefault(user_id, []).append(time)
        self.qi.setdefault(item_id, np.zeros((self.f, 1)))  # 防止有未出现过的user或item
        self.pu.setdefault(user_id, np.zeros((self.f, 1)))
        # self.pu_t.setdefault(user_id, np.zeros(self.bpu_t_mun))  # 平均用40个参数来描述用户每天的偏好
        self.y.setdefault(user_id, np.zeros((self.f, 1)))
        self.u_dict.setdefault(user_id, [])
        self.bi_bin.setdefault(item_id, np.zeros(self.bin_num))
        user_impl_prf, sqrt_Ru = self.getY(user_id)

        dev = self.dev_u(user_id, time)
        bin_index = self.get_bin(time)
        # param_index = self.get_bpu_t(time)
        # bi = self.bi[item_id] +
        # bu = self.bu[user_id] + self.alpha_u[user_id] * dev +

        # rating = self.avg + self.bi[item_id] + self.bu[user_id] + np.sum(
        #     self.qi[item_id] * (self.pu[user_id] + user_impl_prf))
        # todo:add bin , bu,t , pu,t and 𝜶*dev_u(t)
        rating = self.avg + self.bi[item_id] + self.bi_bin[item_id][bin_index] + self.bu[user_id] + (
                self.alpha_u[user_id] * dev) + self.qi[item_id].T.dot((self.pu[user_id] + user_impl_prf))
        # 当分数大于5或小于1时，返回5或1
        if rating > 5:
            rating = 5.0
        if rating < 1:
            rating = 1.0
        return rating

    def getY(self, user_id):
        Ru = self.u_dict[user_id]
        I_Ru = len(Ru)
        sqrt_Ru = np.sqrt(I_Ru)
        y_u = np.zeros((self.f, 1))
        if I_Ru == 0:  # 用户没有任何评分
            user_impl_prf = y_u
        else:
            for i in Ru:
                y_u += self.y[i]
            user_impl_prf = y_u / sqrt_Ru

        return user_impl_prf, sqrt_Ru

    def train(self, steps=30, gamma=0.007, Lambda=0.15):  # gamma是学习速率
        print('train data size', self.mat.shape[0])
        for step in range(steps):
            print('step', step + 1, 'is running')
            KK = np.random.permutation(self.mat.shape[0])  # 随机梯度下降算法
            rmse = 0.0
            for i in range(self.mat.shape[0]):
                j = KK[i]
                user_id = self.mat[j, 0]
                item_id = self.mat[j, 1]
                rating = self.mat[j, 2]
                time = int(self.mat[j, 3])

                dev = self.dev_u(user_id, time)
                bin_index = self.get_bin(time)
                # param_index = self.get_bpu_t(time)

                predict = self.predict(user_id, item_id, time)
                user_impl_prf, sqrt_Ru = self.getY(user_id)
                eui = rating - predict
                rmse += eui ** 2
                self.bu[user_id] += gamma * (eui - Lambda * self.bu[user_id])
                self.bi[item_id] += gamma * (eui - Lambda * self.bi[item_id])
                P_u = self.pu[user_id][:]
                self.pu[user_id] += gamma * (eui * self.qi[item_id] - Lambda * self.pu[user_id])
                self.qi[item_id] += gamma * (eui * (P_u + user_impl_prf) - Lambda * self.qi[item_id])
                for j in self.u_dict[user_id]:
                    self.y[j] += gamma * (eui * self.qi[j] / sqrt_Ru - Lambda * self.y[j])
                self.bi_bin[item_id][bin_index] += gamma * (eui - Lambda * self.bi_bin[item_id][bin_index])
                # self.bu_t[user_id][param_index] += 2 * gamma * (eui - Lambda * self.bu_t[user_id][param_index])
                # self.pu_t[user_id][param_index] += 2 * gamma * (
                #         eui * self.qi[item_id] - Lambda * self.pu_t[user_id][param_index])
                self.alpha_u[user_id] += gamma * (
                        eui * dev - Lambda * self.alpha_u[user_id])

            gamma = 0.93 * gamma  # gamma以0.93的速率降低
            print('rmse is', np.sqrt(rmse / self.mat.shape[0]))

    def test(self, test_data):

        test_data = np.array(test_data)
        print('test data size', test_data.shape[0])
        rmse = 0.0
        for i in range(test_data.shape[0]):
            user_id = test_data[i, 0]
            item_id = test_data[i, 1]
            rating = float(test_data[i, 2])
            time = int(test_data[i, 3])
            eui = rating - self.predict(user_id, item_id, time)
            rmse += eui ** 2
        print('rmse of test data is', np.sqrt(rmse / test_data.shape[0]))


train_data, test_data, data = load_dataset()
a = timeSVDpp(train_data, 10)
a.train()
a.test(test_data)

# f=10; step=20; gamma=0.04; rmse = 1.03234706
# f=20; step=30; gamma=0.04; rmse = 1.04029142
# f=10; step=30; gamma=0.01; rmse = 1.01328479
# f=10; step=50; gamma=0.005; rmse = 0.99921006
# f=10; step=30; gamma=0.007; rmse = 0.99539091（2w）
# f=10; step=30; gamma=0.007; rmse = 0.95800567（5w）

