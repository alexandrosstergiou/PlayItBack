import pandas as pd
import os
from sklearn.utils import shuffle

df_balanced = pd.read_csv('balanced_train_segments.csv')
df_labels = pd.read_csv('class_labels_indices.csv')
df_eval = pd.read_csv('eval_segments.csv')
df_unbalanced = pd.read_csv('unbalanced_train_segments.csv')

df_pkled_balanced = pd.DataFrame()
df_pkled_unbalanced = pd.DataFrame()
df_pkled_as500k = pd.DataFrame()
df_pkled_eval = pd.DataFrame()


# balanced
if not os.path.isfile("balanced_train.pkl"):
    video_names = []
    video_labels = []
    video_ids = []
    for index, row in df_balanced.iterrows():
        v_id = row['YTID']+'_'+str(int(row['start_seconds']))+'_'+str(int(row['end_seconds']))+'.mp4'
        l = row['positive_labels']
        if ',' in l:
            l = l.split(',')
        else:
            l = [l]
        labels = [df_labels.loc[df_labels.mid == ls,'display_name'].item() for ls in l]
        ids = [df_labels.loc[df_labels.mid == ls,'index'].item() for ls in l]
        video_names.append(v_id)
        video_labels.append(labels)
        video_ids.append(ids)

    df_pkled_balanced['video'] = video_names
    df_pkled_balanced['classes'] = video_labels
    df_pkled_balanced['class_ids'] = video_ids
    df_pkled_balanced.to_pickle("balanced_train.pkl")


# unbalanced
if not os.path.isfile("unbalanced_train.pkl"):
    video_names = []
    video_labels = []
    video_ids = []
    for index, row in df_unbalanced.iterrows():
        v_id = row['YTID']+'_'+str(int(row['start_seconds']))+'_'+str(int(row['end_seconds']))+'.mp4'
        l = row['positive_labels']
        if ',' in l:
            l = l.split(',')
        else:
            l = [l]
        labels = [df_labels.loc[df_labels.mid == ls,'display_name'].item() for ls in l]
        ids = [df_labels.loc[df_labels.mid == ls,'index'].item() for ls in l]
        video_names.append(v_id)
        video_labels.append(labels)
        video_ids.append(ids)


    df_pkled_unbalanced['video'] = video_names
    df_pkled_unbalanced['classes'] = video_labels
    df_pkled_unbalanced['class_ids'] = video_ids
    df_pkled_unbalanced.to_pickle("unbalanced_train.pkl")


# as500k
classes_indices = {}
if not os.path.isfile("as500k_train.pkl"):
    video_names = []
    video_labels = []
    video_ids = []
    for index, row in df_unbalanced.iterrows():
        v_id = row['YTID']+'_'+str(int(row['start_seconds']))+'_'+str(int(row['end_seconds']))+'.mp4'
        l = row['positive_labels']
        if ',' in l:
            l = l.split(',')
        else:
            l = [l]
        labels = [df_labels.loc[df_labels.mid == ls,'display_name'].item() for ls in l]
        ids = [df_labels.loc[df_labels.mid == ls,'index'].item() for ls in l]

        for id in ids:
            if not id in classes_indices.keys():
                classes_indices[id] = 1
            else:
                classes_indices[id] = classes_indices[id] + 1

        for l,d in zip(labels,ids):
            if classes_indices[d] > 500000:
                labels.remove(l)
                ids.remove(d)

        if not len(labels) == 0:
            video_names.append(v_id)
            video_labels.append(labels)
            video_ids.append(ids)


    df_pkled_as500k['video'] = video_names
    df_pkled_as500k['classes'] = video_labels
    df_pkled_as500k['class_ids'] = video_ids
    df_pkled_as500k.to_pickle("as500k_train.pkl")

# eval
if not os.path.isfile("test.pkl"):
    video_names = []
    video_labels = []
    video_ids = []
    for index, row in df_eval.iterrows():
        v_id = row['YTID']+'_'+str(int(row['start_seconds']))+'_'+str(int(row['end_seconds']))+'.mp4'
        l = row['positive_labels']
        if ',' in l:
            l = l.split(',')
        else:
            l = [l]
        labels = [df_labels.loc[df_labels.mid == ls,'display_name'].item() for ls in l]
        ids = [df_labels.loc[df_labels.mid == ls,'index'].item() for ls in l]
        video_names.append(v_id)
        video_labels.append(labels)
        video_ids.append(ids)

    df_pkled_eval['video'] = video_names
    df_pkled_eval['classes'] = video_labels
    df_pkled_eval['class_ids'] = video_ids
    df_pkled_eval.to_pickle("test.pkl")
