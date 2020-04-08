from decision_tree_ID3 import predict
import numpy as np
import time


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
        cardio_data.at[i, 'ap_lo'] = cardio_data.at[i, 'ap_lo'] * -1
    if ap_lo[i] > 5000:
        cardio_data.at[i, 'ap_lo'] = cardio_data.at[i, 'ap_lo'] / 100
    if ap_lo[i] > 500:
        cardio_data.at[i, 'ap_lo'] = cardio_data.at[i, 'ap_lo'] / 10
ap_lo_mean = int(np.mean(ap_lo))
for i in range(len(ap_lo)):
    if ap_lo[i] == 0 or ap_lo[i] < 20:
        cardio_data.at[i, 'ap_lo'] = ap_lo_mean
    if 0 < ap_lo[i] < 100:
        cardio_data.at[i, 'ap_lo'] = (ap_lo[i] // 10) * 10


train_set_1 = cardio_data[1000:3000]
train_set_2 = cardio_data[3000:5000]
train_set_3 = cardio_data[5000:7000]
train_set_4 = cardio_data[7000:9000]
train_set_5 = cardio_data[9000:11000]
test_set = cardio_data[0:999]
test_set_labels = np.asarray(test_set.pop('cardio'))
result_labels = []
start_time = time.time()
for i in range(len(test_set)):
    decision = 0
    p1 = predict(train_set_1, test_set.iloc[[i]]))
    if p1 < 0.5:
        decision = decision - 1
    else :
        decision = decision + 1
    p2 = predict(train_set_2, test_set.iloc[[i]]))
    if p2 < 0.5:
        decision = decision - 1
    else :
        decision = decision + 1
    p3 = predict(train_set_3, test_set.iloc[[i]]))
    if p3 < 0.5:
        decision = decision - 1
    else :
        decision = decision + 1
    p4 = predict(train_set_4, test_set.iloc[[i]]))
    if p4 < 0.5:
        decision = decision - 1
    else :
        decision = decision + 1
    p5 = predict(train_set_5, test_set.iloc[[i]]))
    if p5 < 0.5:
        decision = decision - 1
    else :
        decision = decision + 1
    if decision < 0:
        result_labels.append(0)
    else :
        result_labels.append(1)
accuracy = 0
for i in range(len(result_labels)):
    if result_labels[i] == test_set_labels[i]:
        accuracy = accuracy + 1
print(accuracy/len(result_labels))
result_df = pd.DataFrame(zip(result_labels, test_set_labels),
                         columns=['Result labels', 'Actual labels'])
compression_opts = dict(method='zip', archive_name='result.csv')
result_df.to_csv('results.zip', index=False, compression=compression_opts)
print("--- %s seconds ---" % (time.time() - start_time))