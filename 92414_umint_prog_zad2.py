#-imports main-----------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
#-init--------------------------------------
df = pd.read_csv('z2_data_1y.csv', sep=';')
pd.set_option('future.no_silent_downcasting', True)
pd.set_option('display.max_columns',None)
pd.set_option('display.width',None)
silent = 0
do_3,do_41,do_42 = 1,0,0
do_univar,do_bivar = 0,0
#-constants---------------------------
NUMERIC_X_COLS = ['instant', 'year', 'holiday', 'workingday', 'weather', 'temperature', 'humidity', 'windspeed']
CATEG_X_COLS = ['month', 'day', 'hour', 'weekday']
NOTHING = 'nic'
EDA_MAPPINGS = {
    'instant':     {np.nan: NOTHING},                                                                       #numer, free range
    'year':        {np.nan: NOTHING},                                                                       #numer, free range
    'month':       {np.nan: NOTHING, 'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12},                         #categ, cap4resolver
    'day':         {np.nan: NOTHING, **{f'd{i}': i for i in range(1, 32)}},                                 #categ, cap4resolver
    'hour':        {np.nan: NOTHING, **{f'h{i}': i for i in range(0, 24)}},                                 #categ, cap4resolver
    'holiday':     {np.nan: NOTHING},                                                                       #numer
    'weekday':     {np.nan: NOTHING, 'sun': 0, 'mon': 1, 'tue': 2, 'wed': 3, 'thu': 4, 'fri': 5, 'sat': 6}, #categ, cap4resolver
    'workingday':  {np.nan: NOTHING, 'False': 0, 'True': 1},                                                #numer
    'weather':     {np.nan: NOTHING, 'clear': 0, 'cloudy': 1, 'light rain/snow': 2, 'heavy rain/snow': 3},  #numer
    'temperature': {np.nan: NOTHING},                                                                       #numer, free range
    'humidity':    {np.nan: NOTHING},                                                                       #numer, free range
    'windspeed':   {np.nan: NOTHING},                                                                       #numer, free range
    'count':       {np.nan: NOTHING}                                                                        #numer, free range
}
KEPT_X_COLS = []
#KEPT_X_COLS = ['temperature','humidity','hour'] #5.1a
#KEPT_X_COLS = ['weather','windspeed','month'] #5.1b
#KEPT_X_COLS = ['hour','temperature','workingday'] #5.2a
#KEPT_X_COLS = ['holiday','weekday','windspeed'] #5.2b
#-functions----------------------
def resolver():
    global df
    for col in df.columns:
        try:
            float(col.rsplit('_', 1)[-1])
            cap, ix=col.rsplit('_', 1)
            df.rename(columns={col:resolverlogic(cap,ix)}, inplace=True)
        except ValueError:
            continue
def resolverlogic(cap, ix):
    for k, v in EDA_MAPPINGS[cap].items():
        if v == float(ix):
            return k
def edamap():
    global df
    df.drop(columns=['month'], inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    df.insert(1, 'year', df['date'].dt.year)
    df.insert(2, 'month', df['date'].dt.month)
    df.insert(3, 'day', df['date'].dt.day)
    df.drop(columns=['date'], inplace=True)

    for col, mapping in EDA_MAPPINGS.items():
        df[col] = df[col].replace(mapping)
    if KEPT_X_COLS != []:
        df = df.loc[:, KEPT_X_COLS+['count']]
    if not silent:
        print('0.0 Eda mapped : date->year,month,day; np.nan->',NOTHING,', words->numbers; cols kept:',list(df.columns))
def cullid():
    global df
    try:
        df.drop(columns=['instant'], inplace=True)
    except KeyError:
        pass
    if not silent:
        print('1.1 Culled id col instant : (ENTIRE COL DELETED)')
def cullnull(null_toomany):
    global df
    for col in df.columns:
        count = (df[col] == NOTHING).sum() 
        if count >= null_toomany:
            df.drop(columns=[col], inplace=True)
            if not silent:
                print('1.2 Culled nulls under',col,': (ENTIRE COL DELETED) cuz has '+str(count)+'x',NOTHING,'rows')
        elif 0 < count < null_toomany:
            df = df[df[col] != NOTHING]
            if not silent:
                print('1.2 Culled nulls under',col,': '+str(count)+'x',NOTHING,'rows removed')
    df=df.astype(np.float64)
def culldupl():
    global df
    count = df.duplicated().sum()
    df=df.drop_duplicates()
    if not silent:
        print('1.3 Culled duplicates : '+str(count)+'x duplicated rows removed')
def culloutl(outl_bounds):
    global df
    for col in df.columns:
        if col not in outl_bounds:
            continue
        before = len(df)
        df = df[(df[col] >= outl_bounds[col][0]) & (df[col] <= outl_bounds[col][1])]
        after = len(df)
        if not silent:
            print('1.4 Culled outliers under',col,': ',before-after,'x rows removed')
def dummify():
    global df
    df = pd.get_dummies(df, columns=[col for col in CATEG_X_COLS if col in df.columns]).astype(np.float64)
    resolver()
    if not silent:
        print('(BIVAR/PCA.ONLY) Performed dummification:',CATEG_X_COLS,'dummified')
def dtdl(dt, tree_depth):
    plt.figure(figsize=(1.5*2**tree_depth,1.5*tree_depth))
    plot_tree(
        dt, feature_names=list(df.drop(columns=['count']).columns),
        filled=True, fontsize=7
    )
    plt.subplots_adjust(left=0,right=1,bottom=0,top=1)
    plt.savefig('tree_high_dpi.png', dpi=100)
def errvis(y_train, y_test,
           y_dt_train_pred, y_dt_test_pred,
           y_ens_train_pred, y_ens_test_pred,
           y_svm_train_pred, y_svm_test_pred):
    fig, axes = plt.subplots(2, 3, figsize=(12,5), sharey=True)
    axes[0][0].scatter(y_dt_train_pred, y_train-y_dt_train_pred, alpha=0.6, color='red')
    axes[0][0].axhline(0, color='black', linestyle='--')
    axes[0][0].set_title(f'DT Train Residuals, MSE={mean_squared_error(y_train, y_dt_train_pred):.4f}, R2={r2_score(y_train, y_dt_train_pred):.4f}')
    axes[0][0].set_xlabel('Predicted')
    axes[0][0].set_ylabel('Residuals')
    axes[1][0].scatter(y_dt_test_pred, y_test-y_dt_test_pred, alpha=0.6, color='red')
    axes[1][0].axhline(0, color='black', linestyle='--')
    axes[1][0].set_title(f'DT Test Residuals, MSE={mean_squared_error(y_test, y_dt_test_pred):.4f}, R2={r2_score(y_test, y_dt_test_pred):.4f}')
    axes[1][0].set_xlabel('Predicted')
    axes[1][0].set_ylabel('Residuals')
    axes[0][1].scatter(y_ens_train_pred, y_train-y_ens_train_pred, alpha=0.6, color='green')
    axes[0][1].axhline(0, color='black', linestyle='--')
    axes[0][1].set_title(f'ENS Train Residuals, MSE={mean_squared_error(y_train, y_ens_train_pred):.4f}, R2={r2_score(y_train, y_ens_train_pred):.4f}')
    axes[0][1].set_xlabel('Predicted')
    axes[1][1].scatter(y_ens_test_pred, y_test-y_ens_test_pred, alpha=0.6, color='green')
    axes[1][1].axhline(0, color='black', linestyle='--')
    axes[1][1].set_title(f'ENS Test Residuals, MSE={mean_squared_error(y_test, y_ens_test_pred):.4f}, R2={r2_score(y_test, y_ens_test_pred):.4f}')
    axes[1][1].set_xlabel('Predicted')
    axes[0][2].scatter(y_svm_train_pred, y_train-y_svm_train_pred, alpha=0.6, color='blue')
    axes[0][2].axhline(0, color='black', linestyle='--')
    axes[0][2].set_title(f'SVM Train Residuals, MSE={mean_squared_error(y_train, y_svm_train_pred):.4f}, R2={r2_score(y_train, y_svm_train_pred):.4f}')
    axes[0][2].set_xlabel('Predicted')
    axes[1][2].scatter(y_svm_test_pred, y_test-y_svm_test_pred, alpha=0.6, color='blue')
    axes[1][2].axhline(0, color='black', linestyle='--')
    axes[1][2].set_title(f'SVM Test Residuals, MSE={mean_squared_error(y_test, y_svm_test_pred):.4f}, R2={r2_score(y_test, y_svm_test_pred):.4f}')
    axes[1][2].set_xlabel('Predicted')
    plt.subplots_adjust(left=0.04,right=0.99,bottom=0.05,top=0.97)
    win = tk.Toplevel()
    win.title('errvis')
    canvas = FigureCanvasTkAgg(fig, master=win)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    tk.Button(win, text='Close', command=win.destroy).pack(pady=10)
    win.geometry('800x600')
def _3(x_train, x_test, y_train, y_test, dt_mxd, dt_mnsl, ens_mxd, ens_ne, svm_c, svm_g):
    #3.1
    dt = DecisionTreeRegressor(max_depth=dt_mxd, min_samples_leaf=dt_mnsl, random_state=42)
    dt.fit(x_train, y_train)
    y_dt_train_pred = dt.predict(x_train)
    y_dt_test_pred = dt.predict(x_test)

    dtdl(dt, dt_mxd)
    #3.2
    ens = AdaBoostRegressor(
        estimator=DecisionTreeRegressor(max_depth=ens_mxd), n_estimators=ens_ne, random_state=42
    )
    ens.fit(x_train, y_train)
    y_ens_train_pred = ens.predict(x_train)
    y_ens_test_pred = ens.predict(x_test)
    print('_3 reports ens feature importances:')
    if len(df.columns)-1 == len(ens.feature_importances_):
        pair=list(zip(df.columns,ens.feature_importances_))
    else:
        pair=list(zip(range(0,len(df.columns)),ens.feature_importances_))
    for cap,imp in pair:
         print(f"{cap}: {imp:.4f}")
    #3.3
    svm = SVR(kernel='rbf', C=svm_c, gamma=svm_g) #200,13
    svm.fit(x_train, y_train)
    y_svm_train_pred = svm.predict(x_train)
    y_svm_test_pred = svm.predict(x_test)
    #3.4
    errvis(y_train, y_test, y_dt_train_pred, y_dt_test_pred, y_ens_train_pred, y_ens_test_pred, y_svm_train_pred, y_svm_test_pred)
def _3dvis(i, j, k, q1, q2, q3):
    global df
    c = (df['count']-df['count'].min())/(df['count'].max()-df['count'].min())
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(i,j,k, c=c, cmap='coolwarm', s=10)
    ax.set_xlabel(q1)
    ax.set_ylabel(q2)
    ax.set_zlabel(q3)
    plt.subplots_adjust(left=0,right=1,top=1,bottom=0)
    plt.show()
def _41(q1, q2, q3):
    #4.1
    global df
    i,j,k = df[q1],df[q2],df[q3]
    _3dvis(i, j, k, q1, q2, q3)
def _42(x_train, x_test, y_train, y_test, dt_mxd, dt_mnsl, ens_mxd, ens_ne, svm_c, svm_g):
    #4.2
    pca_3d = PCA(n_components=3, random_state=42)
    x_train_3d = pca_3d.fit_transform(x_train)
    x_test_3d = pca_3d.transform(x_test)
    
    q1,q2,q3 = '1d', '2d', '3d'
    x_3d=np.concatenate((x_train_3d, x_test_3d))
    i,j,k = x_3d.T[0],x_3d.T[1],x_3d.T[2]
    _3dvis(i, j, k, q1, q2, q3)

    _3(x_train_3d, x_test_3d, y_train, y_test, dt_mxd, dt_mnsl, ens_mxd, ens_ne, svm_c, svm_g)
#-runtime--------------------------------------------------
#0.0
edamap()
#1.1
cullid()
#1.2
cullnull(2000)
#1.3
culldupl()
#1.4
culloutl({'temperature': [-40, 40], 'humidity': [0, 100], 'windspeed': [0, 110]})
#(BIVAR/PCA.ONLY)
if do_bivar or do_42:
    dummify()
#2
x = copy.deepcopy(df.drop(columns=['count']).to_numpy())
y = copy.deepcopy(df['count'].to_numpy())

tn, tt = 8,2
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=(tt)/(tn+tt), random_state=42
)
if do_3 or do_42:
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test  = scaler.transform(x_test)
if do_3:
    _3(x_train, x_test, y_train, y_test, 8, 1, 12, 45, 340, 15)
if do_41:
    _41('hour','temperature','workingday')
if do_42:
    _42(x_train, x_test, y_train, y_test, 8, 1, 12, 45, 340, 15)
#-export-----------------
x=np.concatenate((x_train, x_test))
y=np.concatenate((y_train, y_test))
new_dfxfrag = pd.DataFrame(x, columns=df.drop(columns=['count']).columns)
new_dfyfrag = pd.Series(y, name='count')
df = pd.concat([new_dfxfrag, new_dfyfrag], axis=1)
df.to_csv('out.csv', sep=';', index=False)
print('OK')
    
if do_univar:
    axes=df.hist(bins=20)
    for ax in axes.ravel():
        ax.grid(False)
    plt.get_current_fig_manager().window.state('zoomed')
    plt.subplots_adjust(left=0.03,right=.97,top=.97,bottom=0.03)
    plt.show()
if do_bivar:
    corr=pd.concat([df.drop(columns=['count']),df['count']],axis=1).corr()
    ax = sns.heatmap(corr,annot=True,fmt='.1f',cmap='coolwarm',cbar=False,
                     xticklabels=corr.columns,yticklabels=corr.columns,
                     annot_kws={"size": 8})
    plt.get_current_fig_manager().window.state('zoomed')
    plt.subplots_adjust(left=0.05,right=1,top=1,bottom=0.11)
    plt.show()

    
