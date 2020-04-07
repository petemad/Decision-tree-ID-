import math
import numpy as np
import pandas as pd
import time


def calculate_entropy(p):
    e = 0
    for pi in p:
        try:
            e = e + -pi * math.log(pi, 2)
        except ValueError:
            e = e + 0
    return e


def calculate_probability(value, list_of_values):
    counter = 0
    for v in list_of_values:
        if v == value:
            counter = counter + 1
    return counter / len(list_of_values)


def get_class_column(dataset):
    return dataset[dataset.columns[-1]]


def get_features(dataset):
    return dataset.columns.values[:-1]


def calculate_global_entropy(dataset):
    classes = get_class_column(dataset)
    unique_classes = classes.unique()
    p = []
    for c in unique_classes:
        p.append(calculate_probability(c, classes))
    return calculate_entropy(p)


def feature_value_information_map(feature_values, class_values):
    classes = class_values.unique()
    unique_feature_values = feature_values.unique()
    feature_info_map = dict()
    for f in unique_feature_values:
        feature_info_map[f] = dict()
    for f in unique_feature_values:
        for c in classes:
            _myMap = feature_info_map[f]
            _myMap[c] = 0
    for f, c in zip(feature_values, class_values):
        _myMap = feature_info_map[f]
        _myMap[c] = _myMap[c] + 1
    return feature_info_map


def calculate_information_gain(dataset, feature_name):
    feature_values = dataset[feature_name]
    class_values = get_class_column(dataset)
    unique_feature_values = feature_values.unique()
    classes = class_values.unique()
    f_v_info_map = feature_value_information_map(feature_values, class_values)
    entropy = []
    summ = []
    for f in unique_feature_values:
        summation = 0
        p = []
        for c in classes:
            _myMap = f_v_info_map[f]
            summation = summation + _myMap[c]
        for c in classes:
            _myMap = f_v_info_map[f]
            val = _myMap[c]
            p.append(val / summation)
        summ.append(summation)
        entropy.append(calculate_entropy(p))
    weighted_sum = 0
    for i in range(len(entropy)):
        weighted_sum = weighted_sum + (entropy[i] * (summ[i] / len(feature_values)))
    global_entropy = calculate_global_entropy(dataset)
    information_gain = global_entropy - weighted_sum
    return information_gain


def choose_root(dataset):
    features = get_features(dataset)
    information_gain = []
    for f in features:
        information_gain.append(calculate_information_gain(dataset, f))
    ind = information_gain.index(np.max(information_gain))
    return np.asarray(features)[ind]


def value_data_map(dataset, feature):
    class_column_list = dataset[feature]
    classes = class_column_list.unique()
    class_column_list = list(class_column_list)
    class_data_map = dict()
    for c in classes:
        class_data_map[c] = []
    for c in classes:
        for i in range(len(class_column_list)):
            if class_column_list[i] == c:
                class_data_map[c].append(dataset.iloc[i])
    for c in classes:
        class_data_map[c] = pd.DataFrame(class_data_map[c])
        class_data_map[c] = class_data_map[c].drop(feature, axis=1)
    return class_data_map


def can_decide(record, val_data_map, root):
    c = np.asarray(record[root])[0]
    data = val_data_map[c]
    classes = get_class_column(data)
    features = get_features(data)
    if len(features) == 0:
        return 'prob'
    elif len(classes.unique()) > 1:
        return False
    else:
        return True


def predict(dataset, record):
    root = choose_root(dataset)
    v_d_map = value_data_map(dataset, root)
    c = np.asarray(record[root])[0]
    data = v_d_map[c]
    if can_decide(record, v_d_map, root) == 'prob':
        prob = calculate_probability(1, get_class_column(data))
        return prob
    if can_decide(record, v_d_map, root):
        return list(get_class_column(data))[0]
    else:
        return predict(data, record)


# Apply the algorithm to the cardio dataset
cardio_data = pd.read_csv('./cardio_train.csv', delimiter=';')
cardio_data = cardio_data.drop('id', axis=1)
age = cardio_data['age']
max_age = np.max(age)
min_age = np.min(age)
diff_age = (max_age - min_age) / 5
# transform the age column to discrete values
for i in range(len(age)):
    if min_age <= cardio_data['age'][i] < (min_age + diff_age):
        update = min_age + (diff_age / 2)
    elif (min_age + diff_age) <= cardio_data['age'][i] < (min_age + (2 * diff_age)):
        update = ((2 * min_age) + (3 * diff_age)) / 2
    elif (min_age + (2 * diff_age)) <= cardio_data['age'][i] < (min_age + (3 * diff_age)):
        update = ((2 * min_age) + (5 * diff_age)) / 2
    elif (min_age + (3 * diff_age)) <= cardio_data['age'][i] < (min_age + (4 * diff_age)):
        update = ((2 * min_age) + (5 * diff_age)) / 2
    elif (min_age + (4 * diff_age)) <= cardio_data['age'][i] < max_age:
        update = (min_age + max_age + (4 * diff_age)) / 2
    cardio_data.at[i, 'age'] = update


height = cardio_data['height']
max_h = np.max(height)
min_h = np.min(height)
diff_h = (max_h - min_h) / 4
for i in range(len(height)):
    if min_h <= cardio_data['height'][i] < (min_h + diff_h):
        update = min_h + (diff_h / 2)
    elif (min_h + diff_h) <= cardio_data['height'][i] < (min_h + (2 * diff_h)):
        update = ((2 * min_h) + (3 * diff_h)) / 2
    elif (min_h + (2 * diff_h)) <= cardio_data['height'][i] < (min_h + (3 * diff_h)):
        update = ((2 * min_h) + (5 * diff_h)) / 2
    elif (min_h + (3 * diff_h)) <= cardio_data['height'][i] < (min_h + (4 * diff_h)):
        update = ((2 * min_h) + (5 * diff_h)) / 2
    cardio_data.at[i, 'height'] = update


weight = cardio_data['weight']
max_w = np.max(weight)
min_w = np.min(weight)
diff_w = (max_w - min_w) / 4
for i in range(len(weight)):
    if min_w <= cardio_data['weight'][i] < (min_w + diff_w):
        update = min_w + (diff_w / 2)
    elif (min_w + diff_w) <= cardio_data['weight'][i] < (min_w + (2 * diff_w)):
        update = ((2 * min_w) + (3 * diff_w)) / 2
    elif (min_w + (2 * diff_w)) <= cardio_data['weight'][i] < (min_w + (3 * diff_w)):
        update = ((2 * min_w) + (5 * diff_w)) / 2
    elif (min_w + (3 * diff_w)) <= cardio_data['weight'][i] < (min_w + (4 * diff_w)):
        update = ((2 * min_w) + (5 * diff_w)) / 2
    cardio_data.at[i, 'weight'] = update

ap_hi = cardio_data['ap_hi']
for i in range(len(ap_hi)):
    # edit negative values
    if ap_hi[i] < 0:
        cardio_data.at[i, 'ap_hi'] = cardio_data.at[i, 'ap_hi'] * -1
    # The range of the not acceptable values ranges from 1 to 24
    if 5 < ap_hi[i] < 25:
        cardio_data.at[i, 'ap_hi'] = cardio_data.at[i, 'ap_hi'] * 10
    # There are exactly two values that equal to 1
    if ap_hi[i] == 1:
        cardio_data.at[i, 'ap_hi'] = 100
    # ranges fro 11020 to 16020
    if ap_hi[i] > 10000:
        cardio_data.at[i, 'ap_hi'] = int(cardio_data.at[i, 'ap_hi'] / 100)
    if 700 < ap_hi[i] < 2100:
        cardio_data.at[i, 'ap_hi'] = int(cardio_data.at[i, 'ap_hi'] / 10)
    if ap_hi[i] == 309 or ap_hi[i] == 401:
        cardio_data.at[i, 'ap_hi'] = 240

ap_lo = cardio_data['ap_lo']
for i in range(len(ap_lo)):
    # edit negative values
    if ap_lo[i] < 0:
        cardio_data.at[i, 'ap_lo'] = cardio_data.at[i, 'ap_lo']*-1
    if ap_lo[i] > 5000:
        cardio_data.at[i, 'ap_lo'] = cardio_data.at[i, 'ap_lo']/100
    if ap_lo[i] > 500:
        cardio_data.at[i, 'ap_lo'] = cardio_data.at[i, 'ap_lo']/10
ap_lo_meam = int(np.mean(ap_lo))
for i in range(len(ap_lo)):
    if ap_lo[i] == 0 or ap_lo[i] < 20:
        cardio_data.at[i, 'ap_lo'] = ap_lo_meam
    if 0 < ap_lo[i] < 100:
        cardio_data.at[i, 'ap_lo'] = (ap_lo[i] // 10)*10

cardio_data_train = cardio_data.copy()
train_set = cardio_data.sample(frac=0.9, random_state=0)
test_set = cardio_data_train.drop(train_set.index)
test_set_labels = test_set.pop('cardio')
result_confidence_level = []
start_time = time.time()
for i in range(len(test_set)):
    result_confidence_level.append(predict(train_set, test_set.iloc[[i]]))
result_labels = []
for cl in result_confidence_level:
    if cl < 0.5:
        result_labels.append(0)
    else:
        result_labels.append(1)
accuracy = 0
for i in range(len(result_labels)):
    if result_labels[i] == test_set_labels[i]:
        accuracy = accuracy + 1
print("--- accuracy = % " % (accuracy/7000))
result_df = pd.DataFrame(zip(result_labels, result_confidence_level, test_set_labels),
                         columns=['Result labels', 'Confidence level', 'Actual labels'])
compression_opts = dict(method='zip', archive_name='result.csv')
result_df.to_csv('results.zip', index=False, compression=compression_opts)
print("--- %s seconds ---" % (time.time() - start_time))


