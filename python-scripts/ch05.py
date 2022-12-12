#!/usr/bin/env python
# coding: utf-8

# In[1]:


#! ipython id=8423516c796d4fcba7176ba452b0c79c
import numpy as np
import pandas as pd


# In[2]:


#! ipython id=a0cfab72569c438a8f28a9ef5d6fc1bf
from pandas import Series, DataFrame


# In[3]:


#! ipython suppress id=cf3b99eb2a384927b92a30f4ac81d2b6
get_ipython().run_line_magic('pushd', 'book-materials')
import numpy as np
np.random.seed(12345)
import matplotlib.pyplot as plt
plt.rc("figure", figsize=(10, 6))
PREVIOUS_MAX_ROWS = pd.options.display.max_rows
pd.options.display.max_rows = 20
pd.options.display.max_columns = 20
pd.options.display.max_colwidth = 80
np.set_printoptions(precision=4, suppress=True)


# In[4]:


#! ipython id=109a9769701946a1807122ae4f49db36
obj = pd.Series([4, 7, -5, 3])
obj


# In[5]:


#! ipython id=cac09267b2504d2fb925cb8ce2b0e72b
obj.array
obj.index


# In[6]:


#! ipython id=1c154eb24cd24f2c9c384792a58a1246
obj2 = pd.Series([4, 7, -5, 3], index=["d", "b", "a", "c"])
obj2
obj2.index


# In[7]:


#! ipython id=225898d625a34be5ae18d996b120fd33
obj2["a"]
obj2["d"] = 6
obj2[["c", "a", "d"]]


# In[8]:


#! ipython id=17a19a84e4b041d5864a687f6fa33bae
obj2[obj2 > 0]
obj2 * 2
import numpy as np
np.exp(obj2)


# In[9]:


#! ipython id=109088e212e74524ba05e1594b0e5c42
"b" in obj2
"e" in obj2


# In[10]:


#! ipython id=0e370847eb3d4d2d9d487fc1d1ec42a7
sdata = {"Ohio": 35000, "Texas": 71000, "Oregon": 16000, "Utah": 5000}
obj3 = pd.Series(sdata)
obj3


# In[11]:


#! ipython id=a5730643a732486ead3db1e8ae3ed8c7
obj3.to_dict()


# In[12]:


#! ipython id=2d7b52717b68416e8f634ffeafb6fc33
states = ["California", "Ohio", "Oregon", "Texas"]
obj4 = pd.Series(sdata, index=states)
obj4


# In[13]:


#! ipython id=b4c9af663a264adaa1ae65f78045d575
pd.isna(obj4)
pd.notna(obj4)


# In[14]:


#! ipython id=41cd0cd1434f457e882fbf21b95f69e5
obj4.isna()


# In[15]:


#! ipython id=f47d1cbe7ccf4d229f0d995b4e81ea25
obj3
obj4
obj3 + obj4


# In[16]:


#! ipython id=9a5faf04218f4c27a4941ca90e2bc998
obj4.name = "population"
obj4.index.name = "state"
obj4


# In[17]:


#! ipython id=39259f38cc1c4acaaf780a3ef38529af
obj
obj.index = ["Bob", "Steve", "Jeff", "Ryan"]
obj


# In[18]:


#! ipython verbatim id=d2c7e7e6d89e40539c24dbcfbe953e1d
data = {"state": ["Ohio", "Ohio", "Ohio", "Nevada", "Nevada", "Nevada"],
        "year": [2000, 2001, 2002, 2001, 2002, 2003],
        "pop": [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]}
frame = pd.DataFrame(data)


# In[19]:


#! ipython id=d8d1e434c0a343cab04f1ebb8d52209e
frame


# In[20]:


#! ipython id=cdb23cb4601f48ea9dce3edd0365b4c8
frame.head()


# In[21]:


#! ipython id=cc54053e5311453ebe9230b1a8a7cf8e
frame.tail()


# In[22]:


#! ipython id=9ab873d64bd84603aba1df2ad164f563
pd.DataFrame(data, columns=["year", "state", "pop"])


# In[23]:


#! ipython id=12523ffe5f1d49dbb513abdb03a31d2b
frame2 = pd.DataFrame(data, columns=["year", "state", "pop", "debt"])
frame2
frame2.columns


# In[24]:


#! ipython id=6e3dd5278f434cd599d2666468e4577c
frame2["state"]
frame2.year


# In[25]:


#! ipython id=3f92b0c9864f432d97881dd72a29b0df
frame2.loc[1]
frame2.iloc[2]


# In[26]:


#! ipython id=41b7b40d14b644e0ac7ee9304509a5fb
frame2["debt"] = 16.5
frame2
frame2["debt"] = np.arange(6.)
frame2


# In[27]:


#! ipython id=8febc7d638974bb4b25e85b7b7912630
val = pd.Series([-1.2, -1.5, -1.7], index=["two", "four", "five"])
frame2["debt"] = val
frame2


# In[28]:


#! ipython id=234b8ef877fc416e87c2473afc3efdd3
frame2["eastern"] = frame2["state"] == "Ohio"
frame2


# In[29]:


#! ipython id=a21013e8edae4e0284876cc17a4f0763
del frame2["eastern"]
frame2.columns


# In[30]:


#! ipython id=0b3c30df8ab14ab1a3486d88b9f23890
populations = {"Ohio": {2000: 1.5, 2001: 1.7, 2002: 3.6},
               "Nevada": {2001: 2.4, 2002: 2.9}}


# In[31]:


#! ipython id=ce53aa5d1ba149ebbad6692ba2b5b4bb
frame3 = pd.DataFrame(populations)
frame3


# In[32]:


#! ipython id=151a9260166c40b2b8037406372b1c8d
frame3.T


# In[33]:


#! ipython id=20aa5034aa9c4aaa86ca519b520f0df0
pd.DataFrame(populations, index=[2001, 2002, 2003])


# In[34]:


#! ipython id=1f91fd2d6c124e3fa22ba5a4820d089c
pdata = {"Ohio": frame3["Ohio"][:-1],
         "Nevada": frame3["Nevada"][:2]}
pd.DataFrame(pdata)


# In[35]:


#! ipython id=87564c9c64c74ad68a9e9ae09ada922a
frame3.index.name = "year"
frame3.columns.name = "state"
frame3


# In[36]:


#! ipython id=25a04ae8186c4456a19a526612717349
frame3.to_numpy()


# In[37]:


#! ipython id=188cda2461e74d2c8eeae893001486b0
frame2.to_numpy()


# In[38]:


#! ipython id=8d6570394d6c41519a0f53fca2898f37
obj = pd.Series(np.arange(3), index=["a", "b", "c"])
index = obj.index
index
index[1:]


# In[39]:


#! ipython id=5049c22f3b92417eab90b3edf07bbec3
labels = pd.Index(np.arange(3))
labels
obj2 = pd.Series([1.5, -2.5, 0], index=labels)
obj2
obj2.index is labels


# In[40]:


#! ipython id=ae6e0ad0026347e2a624f864873bdcc2
frame3
frame3.columns
"Ohio" in frame3.columns
2003 in frame3.index


# In[41]:


#! ipython id=63bf8fa08ef54a05b3b099a61cc89743
pd.Index(["foo", "foo", "bar", "bar"])


# In[42]:


#! ipython id=328712376009457e8a7d83146296bccf
obj = pd.Series([4.5, 7.2, -5.3, 3.6], index=["d", "b", "a", "c"])
obj


# In[43]:


#! ipython id=5980e55c4b1f4943a6bf013a1898544a
obj2 = obj.reindex(["a", "b", "c", "d", "e"])
obj2


# In[44]:


#! ipython id=fbc0d3b6adc64be781a6d9cbb7132352
obj3 = pd.Series(["blue", "purple", "yellow"], index=[0, 2, 4])
obj3
obj3.reindex(np.arange(6), method="ffill")


# In[45]:


#! ipython id=3a2e62814cda43e6924c63d519d47591
frame = pd.DataFrame(np.arange(9).reshape((3, 3)),
                     index=["a", "c", "d"],
                     columns=["Ohio", "Texas", "California"])
frame
frame2 = frame.reindex(index=["a", "b", "c", "d"])
frame2


# In[46]:


#! ipython id=899fe1bb88214dd4ab543c8242adf628
states = ["Texas", "Utah", "California"]
frame.reindex(columns=states)


# In[47]:


#! ipython id=b9b8180a14764143adb17041231fdf08
frame.reindex(states, axis="columns")


# In[48]:


#! ipython id=6d8400ea97fd4bd8bf704f18835cada3
frame.loc[["a", "d", "c"], ["California", "Texas"]]


# In[49]:


#! ipython id=21e5021229c34d05ac9ec0dec36c506d
obj = pd.Series(np.arange(5.), index=["a", "b", "c", "d", "e"])
obj
new_obj = obj.drop("c")
new_obj
obj.drop(["d", "c"])


# In[50]:


#! ipython id=d6ee7bd35ad844b8b31a02cc2345e8dc
data = pd.DataFrame(np.arange(16).reshape((4, 4)),
                    index=["Ohio", "Colorado", "Utah", "New York"],
                    columns=["one", "two", "three", "four"])
data


# In[51]:


#! ipython id=59357e8a21f64a9fabdd36f5643dbb4f
data.drop(index=["Colorado", "Ohio"])


# In[52]:


#! ipython id=d55b792ba8924524857e51d43b177489
data.drop(columns=["two"])


# In[53]:


#! ipython id=4645ed672f144c4fb368d30c774f906a
data.drop("two", axis=1)
data.drop(["two", "four"], axis="columns")


# In[54]:


#! ipython id=85f497478a9d4b5e97aa8012912d1d2f
obj = pd.Series(np.arange(4.), index=["a", "b", "c", "d"])
obj
obj["b"]
obj[1]
obj[2:4]
obj[["b", "a", "d"]]
obj[[1, 3]]
obj[obj < 2]


# In[55]:


#! ipython id=655B753DAA154E1FB24039C26C83060B
obj.loc[["b", "a", "d"]]


# In[56]:


#! ipython id=FE6AE0EAF21741D8A054BEF2C9CC57C8
obj1 = pd.Series([1, 2, 3], index=[2, 0, 1])
obj2 = pd.Series([1, 2, 3], index=["a", "b", "c"])
obj1
obj2
obj1[[0, 1, 2]]
obj2[[0, 1, 2]]


# In[57]:


#! ipython id=82C31C77659740BF9AF1B87B9A392659
obj1.iloc[[0, 1, 2]]
obj2.iloc[[0, 1, 2]]


# In[58]:


#! ipython id=faaaebe3b978418f88cde6c84b3f7025
obj2.loc["b":"c"]


# In[59]:


#! ipython id=b1ef8c6f13f24aec959f541900023322
obj2.loc["b":"c"] = 5
obj2


# In[60]:


#! ipython id=a8c96c31128a41658f14d2442c891713
data = pd.DataFrame(np.arange(16).reshape((4, 4)),
                    index=["Ohio", "Colorado", "Utah", "New York"],
                    columns=["one", "two", "three", "four"])
data
data["two"]
data[["three", "one"]]


# In[61]:


#! ipython id=dd77a40e483f4eb18dcbf506fbcc5d2a
data[:2]
data[data["three"] > 5]


# In[62]:


#! ipython id=9511846494a343e99600772bda5cc9e1
data < 5


# In[63]:


#! ipython id=c8be1582b25543df91106bb41cc5e7f1
data[data < 5] = 0
data


# In[64]:


#! ipython id=68f97d0a40964d4084df3e6c69c7750c
data
data.loc["Colorado"]


# In[65]:


#! ipython id=0f49b78471e0479b9fe9aed7b3de7c3d
data.loc[["Colorado", "New York"]]


# In[66]:


#! ipython id=5003cd2010a5468daf398166ef509b78
data.loc["Colorado", ["two", "three"]]


# In[67]:


#! ipython id=abea7d531ec64961b975355fad10ed23
data.iloc[2]
data.iloc[[2, 1]]
data.iloc[2, [3, 0, 1]]
data.iloc[[1, 2], [3, 0, 1]]


# In[68]:


#! ipython id=729735e3373e433a97a215dc545059b3
data.loc[:"Utah", "two"]
data.iloc[:, :3][data.three > 5]


# In[69]:


#! ipython id=f3d7c4a30f744a2f9a01745b8fae3472
data.loc[data.three >= 2]


# In[70]:


#! ipython allow_exceptions id=06fa3619ab814c8a805051d3dc07a14d
ser = pd.Series(np.arange(3.))
ser
ser[-1]


# In[71]:


#! ipython id=67e50e342e664361802d030ed79cb05d
ser


# In[72]:


#! ipython id=6b6026d6df544f24b4db09fb1c10e3bb
ser2 = pd.Series(np.arange(3.), index=["a", "b", "c"])
ser2[-1]


# In[73]:


#! ipython allow_exceptions id=f519b83b2d0e4d1fae95a7b27069b69d
ser.iloc[-1]


# In[74]:


#! ipython id=ce162f7e552f402e88b968b48d1b4e8c
ser[:2]


# In[75]:


#! ipython id=b52f3eac12614d848a86489c34c763b3
data.loc[:, "one"] = 1
data
data.iloc[2] = 5
data
data.loc[data["four"] > 5] = 3
data


# In[76]:


#! ipython suppress id=a66680e4e4944ed080d63272cc49a0fc
data.loc[data.three == 5]["three"] = 6


# In[77]:


#! ipython id=596c80a9a2b04827a21d6bbc850ca254
data


# In[78]:


#! ipython id=5a0fe0249caf4832bbca24428ac384c2
data.loc[data.three == 5, "three"] = 6
data


# In[79]:


#! ipython id=8e1ec27bfff145398d40159c0274a17d
s1 = pd.Series([7.3, -2.5, 3.4, 1.5], index=["a", "c", "d", "e"])
s2 = pd.Series([-2.1, 3.6, -1.5, 4, 3.1],
               index=["a", "c", "e", "f", "g"])
s1
s2


# In[80]:


#! ipython id=968ac5a22c6b4dc99def5fbbc7bc5dad
s1 + s2


# In[81]:


#! ipython id=67d3d91915da4a38a5c048a14be15dd2
df1 = pd.DataFrame(np.arange(9.).reshape((3, 3)), columns=list("bcd"),
                   index=["Ohio", "Texas", "Colorado"])
df2 = pd.DataFrame(np.arange(12.).reshape((4, 3)), columns=list("bde"),
                   index=["Utah", "Ohio", "Texas", "Oregon"])
df1
df2


# In[82]:


#! ipython id=901b4e8f9e6c4d089d62839f975084da
df1 + df2


# In[83]:


#! ipython id=df97a438f4a843d99c3368c57b80860c
df1 = pd.DataFrame({"A": [1, 2]})
df2 = pd.DataFrame({"B": [3, 4]})
df1
df2
df1 + df2


# In[84]:


#! ipython id=1edb825335d94eb9a4a80836a75e4329
df1 = pd.DataFrame(np.arange(12.).reshape((3, 4)),
                   columns=list("abcd"))
df2 = pd.DataFrame(np.arange(20.).reshape((4, 5)),
                   columns=list("abcde"))
df2.loc[1, "b"] = np.nan
df1
df2


# In[85]:


#! ipython id=78588cf651234834908f8023f96f677f
df1 + df2


# In[86]:


#! ipython id=ae29de7160e54c1ca33fabdc91ad44ee
df1.add(df2, fill_value=0)


# In[87]:


#! ipython id=c37b4e0701a343aba7236622cb3ecee0
1 / df1
df1.rdiv(1)


# In[88]:


#! ipython id=c47367ccc10648e29bf8e1b9a0ce21b6
df1.reindex(columns=df2.columns, fill_value=0)


# In[89]:


#! ipython id=702c34a6688847919fa2fadc17946eb4
arr = np.arange(12.).reshape((3, 4))
arr
arr[0]
arr - arr[0]


# In[90]:


#! ipython id=77a1cd84b9334c3e8bc12e79ca7c37b7
frame = pd.DataFrame(np.arange(12.).reshape((4, 3)),
                     columns=list("bde"),
                     index=["Utah", "Ohio", "Texas", "Oregon"])
series = frame.iloc[0]
frame
series


# In[91]:


#! ipython id=defba2a16851407cbf7d19dc036cfb6d
frame - series


# In[92]:


#! ipython id=949f3f67d1ac48d3a0e32eb51611e649
series2 = pd.Series(np.arange(3), index=["b", "e", "f"])
series2
frame + series2


# In[93]:


#! ipython id=29ad65971c53424aaeb21f009454414c
series3 = frame["d"]
frame
series3
frame.sub(series3, axis="index")


# In[94]:


#! ipython id=034aa9e1976846a888eb7f4d16586081
frame = pd.DataFrame(np.random.standard_normal((4, 3)),
                     columns=list("bde"),
                     index=["Utah", "Ohio", "Texas", "Oregon"])
frame
np.abs(frame)


# In[95]:


#! ipython id=33d3e9f69e734a63b4126e27b8a3621a
def f1(x):
    return x.max() - x.min()

frame.apply(f1)


# In[96]:


#! ipython id=4c55244879094bf5811d05244d2a230d
frame.apply(f1, axis="columns")


# In[97]:


#! ipython id=2d02bdcf56984d359a678eda51cfc90a
def f2(x):
    return pd.Series([x.min(), x.max()], index=["min", "max"])
frame.apply(f2)


# In[98]:


#! ipython id=03714284e3a74941a7a81c9f8225dad2
def my_format(x):
    return f"{x:.2f}"

frame.applymap(my_format)


# In[99]:


#! ipython id=8ff61d5965c44373b2fbd3d6c6d055e1
frame["e"].map(my_format)


# In[100]:


#! ipython id=d0e3fb6710a04be7b65557895820e8ee
obj = pd.Series(np.arange(4), index=["d", "a", "b", "c"])
obj
obj.sort_index()


# In[101]:


#! ipython id=e6fb8d18326d4d73bb700eb7a5dab7dd
frame = pd.DataFrame(np.arange(8).reshape((2, 4)),
                     index=["three", "one"],
                     columns=["d", "a", "b", "c"])
frame
frame.sort_index()
frame.sort_index(axis="columns")


# In[102]:


#! ipython id=7f3bf86fc9dd4cbfa517949a64a71146
frame.sort_index(axis="columns", ascending=False)


# In[103]:


#! ipython id=b0d33ccb4dbc4ec8852f688a921f880b
obj = pd.Series([4, 7, -3, 2])
obj.sort_values()


# In[104]:


#! ipython id=427d6873c73b4c738385e2121309c51b
obj = pd.Series([4, np.nan, 7, np.nan, -3, 2])
obj.sort_values()


# In[105]:


#! ipython id=e1a336ca986046c8b0aa1eeb8177f7c0
obj.sort_values(na_position="first")


# In[106]:


#! ipython id=d18ce999bf1d435381d8049084ad1572
frame = pd.DataFrame({"b": [4, 7, -3, 2], "a": [0, 1, 0, 1]})
frame
frame.sort_values("b")


# In[107]:


#! ipython id=e8d81f94d83d41c6a58ae00f2e4b7479
frame.sort_values(["a", "b"])


# In[108]:


#! ipython id=848608e226a74f93919a4e2142671521
obj = pd.Series([7, -5, 7, 4, 2, 0, 4])
obj.rank()


# In[109]:


#! ipython id=b1493f954289435bbb9407bf167522f8
obj.rank(method="first")


# In[110]:


#! ipython id=9c90dab0cadc4eaebcf98eba4989d952
obj.rank(ascending=False)


# In[111]:


#! ipython id=16017d1637de4b5ba9c49a268b6bee22
frame = pd.DataFrame({"b": [4.3, 7, -3, 2], "a": [0, 1, 0, 1],
                      "c": [-2, 5, 8, -2.5]})
frame
frame.rank(axis="columns")


# In[112]:


#! ipython id=72b1b246fbfa43dbaf69cf8cf27c0955
obj = pd.Series(np.arange(5), index=["a", "a", "b", "b", "c"])
obj


# In[113]:


#! ipython id=de8dc594c6aa49f78d50be960f071703
obj.index.is_unique


# In[114]:


#! ipython id=c46c8ff26d6748f4b58fc9f011016596
obj["a"]
obj["c"]


# In[115]:


#! ipython id=d88c0cebdc6c415f9189e69148daa4b1
df = pd.DataFrame(np.random.standard_normal((5, 3)),
                  index=["a", "a", "b", "b", "c"])
df
df.loc["b"]
df.loc["c"]


# In[116]:


#! ipython id=be356bb6c69b42a8b08bb64a882be75a
df = pd.DataFrame([[1.4, np.nan], [7.1, -4.5],
                   [np.nan, np.nan], [0.75, -1.3]],
                  index=["a", "b", "c", "d"],
                  columns=["one", "two"])
df


# In[117]:


#! ipython id=9667d27e90654379827c6ec2f4279767
df.sum()


# In[118]:


#! ipython id=9c79fbba54b94ff1b8221179c035326b
df.sum(axis="columns")


# In[119]:


#! ipython id=39cfbf6aadb94a95b34e474d1c0ac79c
df.sum(axis="index", skipna=False)
df.sum(axis="columns", skipna=False)


# In[120]:


#! ipython id=D35C0B9048F14176B6E5D0D8E5C77AC7
df.mean(axis="columns")


# In[121]:


#! ipython id=0e594ab9fbac4f03a872ce850303250c
df.idxmax()


# In[122]:


#! ipython id=a496b59741e14566aa71f41813e209d4
df.cumsum()


# In[123]:


#! ipython id=05b39485471d4aebb3ddc860d8278662
df.describe()


# In[124]:


#! ipython id=65e9049764c74b3dbea8926c6943f0cf
obj = pd.Series(["a", "a", "b", "c"] * 4)
obj.describe()


# In[125]:


#! ipython id=e97bd8c699eb46259b5bc4d9a50b918b
price = pd.read_pickle("examples/yahoo_price.pkl")
volume = pd.read_pickle("examples/yahoo_volume.pkl")


# In[126]:


#! ipython id=199d47b63f174740850946cb9d8c09bd
returns = price.pct_change()
returns.tail()


# In[127]:


#! ipython id=acc082d6d7764e0ebfedaf4a92aa5773
returns["MSFT"].corr(returns["IBM"])
returns["MSFT"].cov(returns["IBM"])


# In[128]:


#! ipython id=816399a611f64102ade1a2922528a35c
returns["MSFT"].corr(returns["IBM"])


# In[129]:


#! ipython id=631a9370002647c4ba7ffbf73e239c5f
returns.corr()
returns.cov()


# In[130]:


#! ipython id=668730d660dd4811aff520a5e870b532
returns.corrwith(returns["IBM"])


# In[131]:


#! ipython id=07109c0f2f1a442f80c3cd9212519342
returns.corrwith(volume)


# In[132]:


#! ipython id=90d33c8d9c4543029cad491971f29316
obj = pd.Series(["c", "a", "d", "a", "a", "b", "b", "c", "c"])


# In[133]:


#! ipython id=18c2f28cf84c4223bb3a96445e2c53e4
uniques = obj.unique()
uniques


# In[134]:


#! ipython id=fb21f3506cb848c1a81c78348cb25847
obj.value_counts()


# In[135]:


#! ipython id=fcb65a23a9b749048c6c1d292392c8c5
pd.value_counts(obj.to_numpy(), sort=False)


# In[136]:


#! ipython id=045ee2714cf3417b98224a725cb9af71
obj
mask = obj.isin(["b", "c"])
mask
obj[mask]


# In[137]:


#! ipython id=e2794101c4db4b8aae1a9d50f0e3b200
to_match = pd.Series(["c", "a", "b", "b", "c", "a"])
unique_vals = pd.Series(["c", "b", "a"])
indices = pd.Index(unique_vals).get_indexer(to_match)
indices


# In[138]:


#! ipython id=43b21608636349b881aadf2a75987ec7
data = pd.DataFrame({"Qu1": [1, 3, 4, 3, 4],
                     "Qu2": [2, 3, 1, 2, 3],
                     "Qu3": [1, 5, 2, 4, 4]})
data


# In[139]:


#! ipython id=9ba487d2333f4a509874619fa6ee6026
data["Qu1"].value_counts().sort_index()


# In[140]:


#! ipython id=0e370e3e65534ed7938ffe5e7d19bb58
result = data.apply(pd.value_counts).fillna(0)
result


# In[141]:


#! ipython id=2AF29992875D411F801CAC28587E96D9
data = pd.DataFrame({"a": [1, 1, 1, 2, 2], "b": [0, 0, 1, 0, 0]})
data
data.value_counts()


# In[142]:


#! ipython suppress id=1ab79407b26d4ee9b72f448cb4ea2fe1
get_ipython().run_line_magic('popd', '')


# In[143]:


#! ipython suppress id=ed35ffdee5f94d9c88d2a466c0526684
pd.options.display.max_rows = PREVIOUS_MAX_ROWS

