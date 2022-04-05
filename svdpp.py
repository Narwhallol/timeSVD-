import numpy as np

from dataset import load_dataset


class SVDpp:
    def __init__(self, mat, f=20):
        self.mat = np.array(mat)
        self.f = f  # 维度
        self.bi = {}
        self.bu = {}
        self.qi = {}
        self.pu = {}
        self.avg = np.mean(self.mat[:, 2].astype(float))
        self.y = {}
        self.u_dict = {}  # R(u)
        # self.tu = {}
        # self.alpha_u = {}
        # self.bin_num = 30
        # self.max_time = 2243
        for i in range(self.mat.shape[0]):
            user_id = self.mat[i, 0]
            item_id = self.mat[i, 1]
            time = int(self.mat[i, 3])
            self.u_dict.setdefault(user_id, [])
            self.u_dict[user_id].append(item_id)
            self.bi.setdefault(item_id, 0)
            self.bu.setdefault(user_id, 0)
            # self.alpha_u.setdefault(user_id, 0)
            # self.tu.setdefault(user_id, []).append(time)
            self.qi.setdefault(item_id, np.random.random((self.f, 1)) / 10 * np.sqrt(self.f))
            self.pu.setdefault(user_id, np.random.random((self.f, 1)) / 10 * np.sqrt(self.f))
            self.y.setdefault(item_id, np.zeros((self.f, 1)) + .1)

    # def dev_u(self, user_id, time):
    #     sign = 1
    #     tu = np.mean(self.tu[user_id])
    #     if time - tu < 0:
    #         sign = -1
    #     dev = sign * pow(abs(time - tu), 0.4)  # beta = 0.4
    #     return dev

    # def get_bin(self, time):  # 求bi,bin(t)
    #     bin_size = (self.max_time - 1) / self.bin_num
    #     return time / bin_size

    def predict(self, user_id, item_id, time):
        self.bi.setdefault(item_id, 0)
        self.bu.setdefault(user_id, 0)
        self.qi.setdefault(item_id, np.zeros((self.f, 1)))  # 初始化qi,pu,y向量全0
        self.pu.setdefault(user_id, np.zeros((self.f, 1)))
        self.y.setdefault(user_id, np.zeros((self.f, 1)))
        self.u_dict.setdefault(user_id, [])
        user_impl_prf, sqrt_Ru = self.getY(user_id)

        rating = self.avg + self.bi[item_id] + self.bu[user_id] + np.sum(
            self.qi[item_id] * (self.pu[user_id] + user_impl_prf))
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

    def train(self, steps=30, gamma=0.01, Lambda=0.15):  # gamma是学习速率
        print('train data size', self.mat.shape[0])
        for step in range(steps):
            print('step', step + 1, 'is running')
            KK = np.random.permutation(self.mat.shape[0])  # 随机梯度下降算法
            rmse = 0.0
            for i in range(self.mat.shape[0]):
                j = KK[i]
                user_id = self.mat[j, 0]
                item_id = self.mat[j, 1]
                rating = float(self.mat[j, 2])
                time = self.mat[j, 3]
                predict = self.predict(user_id, item_id, time)
                user_impl_prf, sqrt_Ru = self.getY(user_id)
                eui = rating - predict
                rmse += eui ** 2
                self.bu[user_id] += 2 * gamma * (eui - Lambda * self.bu[user_id])
                self.bi[item_id] += 2 * gamma * (eui - Lambda * self.bi[item_id])
                self.pu[user_id] += 2 * gamma * (eui * self.qi[item_id] - Lambda * self.pu[user_id])
                self.qi[item_id] += 2 * gamma * (eui * (self.pu[user_id] + user_impl_prf) - Lambda * self.qi[item_id])
                for j in self.u_dict[user_id]:
                    self.y[j] += 2 * gamma * (eui * self.qi[j] / sqrt_Ru - Lambda * self.y[j])

            gamma = 0.93 * gamma  # gamma以0.93的速率降低
            print('rmse is', np.sqrt(rmse / self.mat.shape[0]))

    def test(self, test_data):

        test_data = np.array(test_data)
        print('test data size', test_data.shape)
        rmse = 0.0
        for i in range(test_data.shape[0]):
            user_id = test_data[i, 0]
            item_id = test_data[i, 1]
            rating = float(test_data[i, 2])
            time = test_data[i, 3]
            eui = rating - self.predict(user_id, item_id, time)
            rmse += eui ** 2
        print('rmse of test data is', np.sqrt(rmse / test_data.shape[0]))


train_data, test_data, data = load_dataset()
a = SVDpp(train_data, 30)
a.train()
a.test(test_data)

# f=10; step=20; gamma=0.04; rmse = 1.032357167596428
# f=20; step=30; gamma=0.04; rmse = 1.0420413654784164
# f=10; step=30; gamma=0.01; rmse = 1.0386664212137127
