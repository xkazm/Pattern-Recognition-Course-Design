import numpy as np
import pandas as pd
import random
import math
import collections

class DTree(object):
    def __init__(self):
        self.split_feature=None
        self.split_value=None
        self.leaf_value=None
        self.tree_left=None
        self.tree_right=None

    def calc_predict_value(self,dataset):
        if self.leaf_value is None:
            return self.leaf_value
        elif dataset[self.split_feature]<=self.split_value:
            return self.tree_left.calc_predict_value(dataset)
        else:
            return self.tree_right.calc_predict_value(dataset)

    def describe_tree(self):
        if not self.tree_left and not self.tree_right:
            leaf_info="{leaf_value:"+str(self.leaf_value)+"}"
            return leaf_info
        left_info=self.tree_left.describe_tree()
        right_info=self.tree_left.describe_tree()
        tree_structure="{split_feature:"+str(self.split_feature)+\
                       ",split_value:"+str(self.split_value)+\
                       ",left_tree:"+left_info+\
                       ",right_tree:"+right_info+"}"
        return tree_structure

class RandomForestClassifier(object):
    def __init__(self,n_estimators=10,max_depth=-1,min_sample_split=2,min_samples_leaf=1,
                 min_split_gain=0.0,colsample_bytree="sqrt",subsample=1.0,random_state=None):
        self.n_estimators=n_estimators
        self.max_depth=max_depth if max_depth!=-1 else float('inf')
        self.min_samples_split=min_sample_split
        self.min_samples_leaf=min_samples_leaf
        self.min_split_gain=min_split_gain
        self.colsample_bytree=colsample_bytree
        self.subsample=subsample
        self.random_state=random_state
        self.trees=dict()
        self.feature_importances=dict()

    def fit(self,dataset,targets):
        targets=targets.to_frame(name='label')

        if self.random_state:
            random.seed(self.random_state)
        random_state_stages=random.sample(range(self.n_estimators),self.n_estimators)

        if self.colsample_bytree=='sqrt':
            self.colsample_bytree=int(len(dataset.columns)**0.5)
        elif self.colsample_bytree=='log2':
            self.colsample_bytree=int(math.log(len(dataset.columns)))
        else:
            self.colsample_bytree=len(dataset.columns)

        for stage in range(self.n_estimators):
            print(("step:"+str(stage+1)).center(80,'='))

            random.seed(random_state_stages[stage])
            subset_index=random.sample(range(len(dataset)),int(self.subsample*len(dataset)))
            subcol_index=random.sample(dataset.columns.tolist(),self.colsample_bytree)
            dataset_copy=dataset.loc[subset_index,subcol_index].reset_index(drop=True)
            targets_copy=targets.loc[subset_index,:].reset_index(drop=True)

            tree=self._fit(dataset_copy,targets_copy,depth=0)
            self.trees[stage]=tree
            print(tree.describe_tree())

    def _fit(self,dataset,targets,depth):
        if len(targets['label'].unique())<=1 or dataset.__len__()<=self.min_samples_split:
            tree=DTree()
            tree.leaf_value=self.calc_leaf_value(targets['label'])
            return tree

        if depth<self.max_depth:
            best_split_feature,best_split_value,best_split_gain=self.choose_best_feature(dataset,targets)
            left_dataset,right_dataset,left_targets,right_targets=self.split_dataset(dataset,targets,best_split_feature,best_split_value)

            tree=DTree()

            if left_dataset.__len__()<=self.min_samples_leaf or right_dataset.__len__()<=self.min_samples_leaf or best_split_gain<=self.min_split_gain:
                tree.leaf_value=self.calc_leaf_value(targets['label'])
                return tree
            else:
                self.feature_importances[best_split_feature]=self.feature_importances.get(best_split_feature,0)+1

                tree.split_feature=best_split_feature
                tree.split_value=best_split_value
                tree.tree_left=self._fit(left_dataset,left_targets,depth+1)
                tree.tree_right=self._fit(right_dataset,right_targets,depth+1)
                return tree
        else:
            tree=DTree()
            tree.leaf_value=self.calc_leaf_value(targets['label'])
            return tree

    def choose_best_feature(self,dataset,targets):
        best_split_gain=1
        best_split_feature=None
        best_split_value=None

        for feature in dataset.columns:
            if dataset[feature].unique().__len__()<=100:
                unique_values=sorted(dataset[feature].unique().tolist())
            else:
                unique_values=np.unique([np.percentile(dataset[feature],x) for x in np.linspace(0,100,100)])

            for split_value in unique_values:
                left_targets=targets[dataset[feature]<=split_value]
                right_targets=targets[dataset[feature]>split_value]
                split_gain=self.calc_gini(left_targets['label'],right_targets['label'])

                if split_gain<best_split_gain:
                    best_split_feature=feature
                    best_split_value=split_value
                    best_split_gain=split_gain
        return best_split_feature,best_split_value,best_split_gain

    def calc_leaf_value(self,targets):
        label_counts=collections.Counter(targets)
        major_label=max(zip(label_counts.values(),label_counts.keys()))
        return major_label[1]

    def calc_gini(self,left_targets,right_targets):
        split_gain=0
        for targets in [left_targets,right_targets]:
            gini=1
            label_counts=collections.Counter(targets)
            for key in label_counts:
                prob=label_counts[key]*1.0/len(targets)
                gini-=prob**2
            split_gain+=len(targets)*1.0/(len(left_targets)+len(right_targets))*gini
        return split_gain

    def split_dataset(self,dataset,targets,split_feature,split_value):
        left_dataset=dataset[dataset[split_feature]<=split_value]
        left_targets=targets[dataset[split_feature]<=split_value]
        right_dataset=dataset[dataset[split_feature]>split_value]
        right_targets=targets[dataset[split_feature]>split_value]
        return left_dataset,right_dataset,left_targets,right_targets

    def predict(self,dataset):
        res=[]
        for index,row in dataset.iterrows():
            pred_list=[]
            for stage,tree in self.trees.items():
                pred_list.append(tree.calc_predict_value(row))

            pred_label_counts=collections.Counter(pred_list)
            pred_label=max(zip(pred_label_counts.values(),pred_label_counts.keys()))
            res.append(pred_label[1])
        return np.array(res)