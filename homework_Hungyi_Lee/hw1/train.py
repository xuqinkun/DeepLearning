import csv
import time


def read_csv(csv_path, data_classes):
    csv_file = open(csv_path, "r", encoding="gb18030")
    csv_reader = csv.reader(csv_file)
    data = []
    key_index = {}
    idx = -1
    for raw in csv_reader:
        line = raw[3:27]
        if len(key_index) <= data_classes:
            key_index[raw[2]] = idx
            idx += 1
        for i in range(0, len(line)):
            if line[i] == "NR":
                line[i] = '0.0'
        data.append(line)
    return data, key_index


def multiply_matrix(mat1, mat2):
    ret = []
    for i in range(len(mat1)):
        ret.append([])
        for col_k in range(len(mat2[0])):
            prod = 0
            for row_j in range(len(mat2)):
                prod += mat1[i][row_j] * mat2[row_j][col_k]
            ret[i].append(prod)
    return ret


def transpose(mat):
    ret = []
    for col in range(len(mat[0])):
        ret.append([])
    for row in range(len(mat)):
        for col in range(len(mat[row])):
            ret[col].append(mat[row][col])
    return ret


def add_matrix(mat1, mat2):
    c = []
    for i in range(len(mat1)):
        c.append([])
        for j in range(len(mat1[0])):
            c[i].append(mat1[i][j] + mat2[i][j])
    return c


def feature_scaling(feature=[]):
    s = 0
    n = len(feature)
    for i in range(n):
        s += feature[i]
    mean = s / n
    s_square = sum([(i - mean) ** 2 for i in feature]) / (n - 1)
    deviation = s_square ** 0.5
    if deviation != 0.0:
        for i in range(n):
            feature[i] = (feature[i] - mean) / deviation
    return feature

class Regression:

    def __init__(self, train_csv, test_csv, data_item_class, data_width, learning_rate):
        self.data = []
        self.label = []
        self.data_class_num = data_item_class
        self.train_csv = train_csv
        self.test_csv = test_csv
        self.data_width = data_width
        self.weight = []
        self.bias = 0
        for i in range(data_item_class * data_width + 1):
            self.weight.append([])
            self.weight[i].append(1)
        self.learning_rate = learning_rate

    def pre_process(self):
        raw_data, key_index = read_csv(self.train_csv, self.data_class_num)

        # Remove head
        raw_data.remove(raw_data[0])
        # How many days for observing data
        days = int(len(raw_data) / self.data_class_num)

        temp_data = []
        days_of_month = 20
        months = int(days / days_of_month)

        for i in range(months):
            month_data = []
            for t in range(self.data_class_num):
                month_data.append([])
            for j in range(days_of_month):
                for k in range(0, self.data_class_num):
                    offset = i * self.data_class_num * days_of_month + j * self.data_class_num + k
                    month_data[k] += raw_data[offset]
            temp_data.append(month_data)

        for i in range(len(temp_data)):
            for j in range(len(temp_data[i])):
                for k in range(len(temp_data[i][j])):
                    temp_data[i][j][k] = float(temp_data[i][j][k])

        # Feature Scaling
        for i in range(len(temp_data)):
            features = temp_data[i]
            for feature in features:
                feature_scaling(feature)

        for i in range(len(temp_data)):
            data_size = len(temp_data[i][0]) - self.data_width
            month_data = temp_data[i]
            for j in range(data_size):
                item = []
                for k in range(self.data_class_num):
                    item.append(month_data[k][j:j + self.data_width])
                # self.data.append(month_data[key_index['PM2.5']][j:j + self.data_width])
                self.data.append(item)
            # Save label data by month
            self.label += month_data[key_index['PM2.5']][self.data_width:]
        pass

    def train(self):
        self.pre_process()
        data_size = len(self.data)
        offset = int(data_size * 0.8)
        x_train = self.data[0:offset]
        y_train = transpose([self.label[0:offset]])
        x_test = self.data[offset:]
        y_test = self.label[offset:]

        X = []
        reg_rate = 0.011
        train_size = len(x_train)
        for i in range(train_size):
            X.append([])
            X[i].append(1)
            # X[i] += x_train[i]
            for j in range(len(x_train[i])):
                X[i] += x_train[i][j]

        times = 0
        autograd = 0
        start_time = time.time()

        loss = float('inf')
        while loss > 0.18:
            times += 1

            distant = add_matrix(multiply_matrix(X, self.weight), multiply_matrix(y_train, [[-1]]))
            grad = multiply_matrix(transpose(X), distant)

            norm_2 = multiply_matrix(transpose(distant), distant)[0][0] / train_size

            for i in range(len(grad)):
                for j in range(len(grad[i])):
                    grad[i][j] /= data_size

            reg_items = multiply_matrix(self.weight, [[reg_rate * 2]])
            reg_items[0][0] = 0
            grad = add_matrix(grad, reg_items)

            reg_item = multiply_matrix(transpose(self.weight), self.weight)[0][0] * reg_rate

            autograd += norm_2

            # Update weight
            neg_grad = multiply_matrix(grad, [[-self.learning_rate / (autograd ** 0.5)]])
            self.weight = add_matrix(self.weight, neg_grad)

            loss = norm_2 + reg_item
            if times % 100 == 0:
                end_time = time.time()
                print("Training time: %ds, iteration times:%d loss: %.5f" % (end_time - start_time, times, loss))
                start_time = end_time
        weight_file = open("data/weight.csv", "w")
        weights = [str(w[0]) for w in self.weight]

        weight_file.write(",".join(weights))


if __name__ == '__main__':
    reg = Regression("data/train.csv", "data/test.csv", 18, 9, 100)
    reg.train()
