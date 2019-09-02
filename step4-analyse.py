
import pandas as pd
from sklearn import tree
from sklearn import metrics
import warnings



def analyse_decisiontree(X_train, y_train, X_test, y_test):
    warnings.filterwarnings('ignore')
    # Modeling
    tree_clf = tree.DecisionTreeClassifier(criterion = "gini", splitter = 'random', max_leaf_nodes = 10, min_samples_leaf = 5, max_depth= 5)
    tree_clf = tree_clf.fit(X_train,y_train)

    # Visualize
    #dot_data = StringIO()
    #tree.export_graphviz(tree_clf, out_file=dot_data, feature_names=features)
    #graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    #Image(graph.create_png())
    #graph.write_png("tree.png")

    #Predict the response for test dataset
    tree_y_pred = tree_clf.predict(X_test)

    # Model Accuracy, how often is the classifier correct?
    #https://www.datacamp.com/community/tutorials/decision-tree-classification-python
    #print(pd.crosstab(y_test, tree_y_pred, rownames = ['y_test'], colnames = ['tree_y_pred']))
    #print("Accuracy:",metrics.accuracy_score(y_test, tree_y_pred))
    #print("Precision:",metrics.average_precision_score(y_test, tree_y_pred))
    #print("recall:",metrics.recall_score(y_test, tree_y_pred))
    #print("f1 score:",metrics.f1_score(y_test, tree_y_pred))

    # Importance
    #tree_imp = pd.DataFrame({'Var' : X_train.columns,'Imp' : tree_clf.feature_importances_ })
    #tree_imp = tree_imp.sort_values(by = 'Imp', ascending=False)[:10]
    #tree_imp.plot.bar(x='Var', y='Imp')
    #plt.show()
    return{#'method': 'Decision Tree',
           'cross' : pd.crosstab(y_test, tree_y_pred, rownames = ['y_test'], colnames = ['tree_y_pred']),
           "Accuracy" : metrics.accuracy_score(y_test, tree_y_pred),
           "Precision" : metrics.average_precision_score(y_test, tree_y_pred),
           "recall" : metrics.recall_score(y_test, tree_y_pred),
           "f1 score" :metrics.f1_score(y_test, tree_y_pred)};



def analyse_randomforest(X_train, y_train, X_test, y_test):
    from sklearn.ensemble import RandomForestClassifier
    warnings.filterwarnings('ignore')
    rf_clf = RandomForestClassifier(n_estimators=300)
    rf_clf = rf_clf.fit(X_train,y_train)

    #Predict the response for test dataset
    rf_y_pred = rf_clf.predict(X_test)

    # Importance
    #rf_feature_imp = pd.Series(rf_clf.feature_importances_,index=X_train.columns).sort_values(ascending=False)[1:10]
    #rf_feature_imp.plot.bar()
    #plt.show()
    return{#'method': 'Random Forest',
           'cross' : pd.crosstab(y_test, rf_y_pred, rownames = ['y_test'], colnames = ['tree_y_pred']),
           "Accuracy" : metrics.accuracy_score(y_test, rf_y_pred),
           "Precision" : metrics.average_precision_score(y_test, rf_y_pred),
           "recall" : metrics.recall_score(y_test, rf_y_pred),
           "f1 score" :metrics.f1_score(y_test, rf_y_pred)};



def analyse_xgboost(X_train, y_train, X_test, y_test):
    import xgboost as xgb
    warnings.filterwarnings('ignore')
    from xgboost import XGBClassifier
    from xgboost.sklearn import XGBClassifier
    from xgboost import plot_importance

    dtrain = xgb.DMatrix(X_train, label= y_train)
    dtest = xgb.DMatrix(X_test, label= y_test)

    param = {'max_depth': 4, 'eta': 1, 'objective': 'binary:logistic','silent':1}
    #param = {'max_depth': 4, 'eta': 1, 'objective': 'multi:softprob','silent':1, 'num_class': 10}



    num_round = 100
    evallist = [(dtest, 'eval'), (dtrain, 'train')]

    bst = xgb.train(param, dtrain, num_round, evallist, verbose_eval=False)

    #Prediction
    xg_y_pred = bst.predict(dtest)
    xg_y_pred  = xg_y_pred > 0.5
    #print(pd.crosstab(y_test, xg_y_pred, rownames = ['y_test'], colnames = ['xg_y_pred']))
    #print("Accuracy:",metrics.accuracy_score(y_test, xg_y_pred))
    #print("Precision:",metrics.average_precision_score(y_test, xg_y_pred))
    #print("recall:",metrics.recall_score(y_test, xg_y_pred))
    #print("f1 score:",metrics.f1_score(y_test, xg_y_pred))

    #Importances
    #xgb.plot_importance(bst, max_num_features=10)
    #plt.show()
    return{#'method': 'Xgboost',
           'cross' : pd.crosstab(y_test, xg_y_pred, rownames = ['y_test'], colnames = ['tree_y_pred']),
           "Accuracy" : metrics.accuracy_score(y_test, xg_y_pred),
           "Precision" : metrics.average_precision_score(y_test, xg_y_pred),
           "recall" : metrics.recall_score(y_test, xg_y_pred),
           "f1 score" :metrics.f1_score(y_test, xg_y_pred)};

