#!/usr/bin/env python
# coding: utf-8

# In[1]:


#! ipython suppress id=a3ea397f76e34983a5af4ea3f35454d5
get_ipython().run_line_magic('pushd', 'book-materials')
import numpy as np
import pandas as pd
pd.options.display.max_rows = 20
pd.options.display.max_colwidth = 80
pd.options.display.max_columns = 20
np.random.seed(12345)
import matplotlib.pyplot as plt
plt.rc("figure", figsize=(10, 6))
np.set_printoptions(precision=4, suppress=True)


# In[2]:


#! ipython id=621650ab20cf419491e0c47ca60e550e
data = pd.Series(np.random.uniform(size=9),
                 index=[["a", "a", "a", "b", "b", "c", "c", "d", "d"],
                        [1, 2, 3, 1, 3, 1, 2, 2, 3]])
data


# In[3]:


#! ipython id=4f2fe3c58dd54150862310bad365e326
data.index


# In[4]:


#! ipython id=28ad22c60acd495f90afdcf494f4cc62
data["b"]
data["b":"c"]
data.loc[["b", "d"]]


# In[5]:


#! ipython id=0cb3aaeee9a94a528395c72925306253
data.loc[:, 2]


# In[6]:


#! ipython id=eb341288b398456aa88e5eda52320bcd
data.unstack()


# In[7]:


#! ipython id=943f16ce2a3d450d8cbb28ab4635e90d
data.unstack().stack()


# In[8]:


#! ipython id=40c41d98fcdc4afa9af2e14878e97f8d
frame = pd.DataFrame(np.arange(12).reshape((4, 3)),
                     index=[["a", "a", "b", "b"], [1, 2, 1, 2]],
                     columns=[["Ohio", "Ohio", "Colorado"],
                              ["Green", "Red", "Green"]])
frame


# In[9]:


#! ipython id=83062f500f574272b3d06be6b2e5b3fc
frame.index.names = ["key1", "key2"]
frame.columns.names = ["state", "color"]
frame


# In[10]:


#! ipython id=562fcf1e41aa416e83bba08459357ca0
frame.index.nlevels


# In[11]:


#! ipython id=ed6816dc1d774d23a55b14bf89565bdc
frame["Ohio"]


# In[12]:


#! ipython id=772991d3f3924148b80fe0eba5409490
frame.swaplevel("key1", "key2")


# In[13]:


#! ipython id=58da2287120d4d31b512ab1d54fb11a9
frame.sort_index(level=1)
frame.swaplevel(0, 1).sort_index(level=0)


# In[14]:


#! ipython id=846b2e10f00744479d2a9aeec5df02ba
frame.groupby(level="key2").sum()
frame.groupby(level="color", axis="columns").sum()


# In[15]:


#! ipython id=04fc9900e5144be48f18eb3c2ff9f2b1
frame = pd.DataFrame({"a": range(7), "b": range(7, 0, -1),
                      "c": ["one", "one", "one", "two", "two",
                            "two", "two"],
                      "d": [0, 1, 2, 0, 1, 2, 3]})
frame


# In[16]:


#! ipython id=2da1465c53d3417ca1307be17bf16ee1
frame2 = frame.set_index(["c", "d"])
frame2


# In[17]:


#! ipython id=f5eb808eaa3c4205a92c92efb781c871
frame.set_index(["c", "d"], drop=False)


# In[18]:


#! ipython id=00d179bed5214d068a0ff6f2c9c73d60
frame2.reset_index()


# In[19]:


#! ipython id=9c7c866c38f0426baacebb7aa9a7f800
df1 = pd.DataFrame({"key": ["b", "b", "a", "c", "a", "a", "b"],
                    "data1": pd.Series(range(7), dtype="Int64")})
df2 = pd.DataFrame({"key": ["a", "b", "d"],
                    "data2": pd.Series(range(3), dtype="Int64")})
df1
df2


# In[20]:


#! ipython id=a7edac90d9404e05a3817e5a891cf800
pd.merge(df1, df2)


# In[21]:


#! ipython id=44769f49c44e4b55b39e3c5ad5d15c6f
pd.merge(df1, df2, on="key")


# In[22]:


#! ipython id=3fe1a0e0afee4d2d9723f49517398e3b
df3 = pd.DataFrame({"lkey": ["b", "b", "a", "c", "a", "a", "b"],
                    "data1": pd.Series(range(7), dtype="Int64")})
df4 = pd.DataFrame({"rkey": ["a", "b", "d"],
                    "data2": pd.Series(range(3), dtype="Int64")})
pd.merge(df3, df4, left_on="lkey", right_on="rkey")


# In[23]:


#! ipython id=36b3b25412c2474c8ac3ba94e43b83bc
pd.merge(df1, df2, how="outer")
pd.merge(df3, df4, left_on="lkey", right_on="rkey", how="outer")


# In[24]:


#! ipython id=aecc234c40094e10a7b92d8c1df2c2e0
df1 = pd.DataFrame({"key": ["b", "b", "a", "c", "a", "b"],
                    "data1": pd.Series(range(6), dtype="Int64")})
df2 = pd.DataFrame({"key": ["a", "b", "a", "b", "d"],
                    "data2": pd.Series(range(5), dtype="Int64")})
df1
df2
pd.merge(df1, df2, on="key", how="left")


# In[25]:


#! ipython id=1039644542aa4f5eb77954754b0bb837
pd.merge(df1, df2, how="inner")


# In[26]:


#! ipython id=c72b7dea085b4c788b7fe9e64e5251a7
left = pd.DataFrame({"key1": ["foo", "foo", "bar"],
                     "key2": ["one", "two", "one"],
                     "lval": pd.Series([1, 2, 3], dtype='Int64')})
right = pd.DataFrame({"key1": ["foo", "foo", "bar", "bar"],
                      "key2": ["one", "one", "one", "two"],
                      "rval": pd.Series([4, 5, 6, 7], dtype='Int64')})
pd.merge(left, right, on=["key1", "key2"], how="outer")


# In[27]:


#! ipython id=093f1002955746f9b7dc75cde68bec0d
pd.merge(left, right, on="key1")


# In[28]:


#! ipython id=b576e79738b04814bca42851e3307031
pd.merge(left, right, on="key1", suffixes=("_left", "_right"))


# In[29]:


#! ipython id=b15d2833cf8c4df9bf11e165d32e4f63
left1 = pd.DataFrame({"key": ["a", "b", "a", "a", "b", "c"],
                      "value": pd.Series(range(6), dtype="Int64")})
right1 = pd.DataFrame({"group_val": [3.5, 7]}, index=["a", "b"])
left1
right1
pd.merge(left1, right1, left_on="key", right_index=True)


# In[30]:


#! ipython id=ede317a83fcb4b96925572fbf8b61b64
pd.merge(left1, right1, left_on="key", right_index=True, how="outer")


# In[31]:


#! ipython id=ad3992c415a04214afbc6ee96919c800
lefth = pd.DataFrame({"key1": ["Ohio", "Ohio", "Ohio",
                               "Nevada", "Nevada"],
                      "key2": [2000, 2001, 2002, 2001, 2002],
                      "data": pd.Series(range(5), dtype="Int64")})
righth_index = pd.MultiIndex.from_arrays(
    [
        ["Nevada", "Nevada", "Ohio", "Ohio", "Ohio", "Ohio"],
        [2001, 2000, 2000, 2000, 2001, 2002]
    ]
)
righth = pd.DataFrame({"event1": pd.Series([0, 2, 4, 6, 8, 10], dtype="Int64",
                                           index=righth_index),
                       "event2": pd.Series([1, 3, 5, 7, 9, 11], dtype="Int64",
                                           index=righth_index)})
lefth
righth


# In[32]:


#! ipython id=b39ab44459844fccaf2b9d9efcfc4f39
pd.merge(lefth, righth, left_on=["key1", "key2"], right_index=True)
pd.merge(lefth, righth, left_on=["key1", "key2"],
         right_index=True, how="outer")


# In[33]:


#! ipython id=03abe17d800a4c2d87ed26e3734d3170
left2 = pd.DataFrame([[1., 2.], [3., 4.], [5., 6.]],
                     index=["a", "c", "e"],
                     columns=["Ohio", "Nevada"]).astype("Int64")
right2 = pd.DataFrame([[7., 8.], [9., 10.], [11., 12.], [13, 14]],
                      index=["b", "c", "d", "e"],
                      columns=["Missouri", "Alabama"]).astype("Int64")
left2
right2
pd.merge(left2, right2, how="outer", left_index=True, right_index=True)


# In[34]:


#! ipython id=852ad741bf6a4678833ce0a65ae27e38
left2.join(right2, how="outer")


# In[35]:


#! ipython id=ef7c8ddca1a14f0e9b2d08e30d2ced8f
left1.join(right1, on="key")


# In[36]:


#! ipython id=4de3092737b54b6db024df459ad1c77b
another = pd.DataFrame([[7., 8.], [9., 10.], [11., 12.], [16., 17.]],
                       index=["a", "c", "e", "f"],
                       columns=["New York", "Oregon"])
another
left2.join([right2, another])
left2.join([right2, another], how="outer")


# In[37]:


#! ipython id=83e9fddfd1654d50bf17752bb93fbbfa
arr = np.arange(12).reshape((3, 4))
arr
np.concatenate([arr, arr], axis=1)


# In[38]:


#! ipython id=e62e34f6a56e4d3f8d688ba9f226733b
s1 = pd.Series([0, 1], index=["a", "b"], dtype="Int64")
s2 = pd.Series([2, 3, 4], index=["c", "d", "e"], dtype="Int64")
s3 = pd.Series([5, 6], index=["f", "g"], dtype="Int64")


# In[39]:


#! ipython id=0e8d10f6da2746ada70388da4ce93c91
s1
s2
s3
pd.concat([s1, s2, s3])


# In[40]:


#! ipython id=8f872f81400043afb3c1bf48942de5a4
pd.concat([s1, s2, s3], axis="columns")


# In[41]:


#! ipython id=51ed91ff4a6d45bcb8a563b6b9752eab
s4 = pd.concat([s1, s3])
s4
pd.concat([s1, s4], axis="columns")
pd.concat([s1, s4], axis="columns", join="inner")


# In[42]:


#! ipython id=10b6e25547cf49868dfd9f769490d495
result = pd.concat([s1, s1, s3], keys=["one", "two", "three"])
result
result.unstack()


# In[43]:


#! ipython id=b7239f0a4b364eb4802a6a72d49d541b
pd.concat([s1, s2, s3], axis="columns", keys=["one", "two", "three"])


# In[44]:


#! ipython id=67e6126a29bf499b84243fe1121d8288
df1 = pd.DataFrame(np.arange(6).reshape(3, 2), index=["a", "b", "c"],
                   columns=["one", "two"])
df2 = pd.DataFrame(5 + np.arange(4).reshape(2, 2), index=["a", "c"],
                   columns=["three", "four"])
df1
df2
pd.concat([df1, df2], axis="columns", keys=["level1", "level2"])


# In[45]:


#! ipython id=6209f511f4bd408c823f3601438554b9
pd.concat({"level1": df1, "level2": df2}, axis="columns")


# In[46]:


#! ipython id=942c4ea1830145879af5d47d5c4d9248
pd.concat([df1, df2], axis="columns", keys=["level1", "level2"],
          names=["upper", "lower"])


# In[47]:


#! ipython id=81cb00cc63184f10aba558c214e4244c
df1 = pd.DataFrame(np.random.standard_normal((3, 4)),
                   columns=["a", "b", "c", "d"])
df2 = pd.DataFrame(np.random.standard_normal((2, 3)),
                   columns=["b", "d", "a"])
df1
df2


# In[48]:


#! ipython id=b663f7c1ded3455db4abd472a8154ada
pd.concat([df1, df2], ignore_index=True)


# In[49]:


#! ipython id=e4379b1ba6c74b9fbf6174fb4f2be35b
a = pd.Series([np.nan, 2.5, 0.0, 3.5, 4.5, np.nan],
              index=["f", "e", "d", "c", "b", "a"])
b = pd.Series([0., np.nan, 2., np.nan, np.nan, 5.],
              index=["a", "b", "c", "d", "e", "f"])
a
b
np.where(pd.isna(a), b, a)


# In[50]:


#! ipython id=f222a83337184ce4bf0efa05e07dc6d3
a.combine_first(b)


# In[51]:


#! ipython id=15c3d24d99974e409a9255e8c01222b5
df1 = pd.DataFrame({"a": [1., np.nan, 5., np.nan],
                    "b": [np.nan, 2., np.nan, 6.],
                    "c": range(2, 18, 4)})
df2 = pd.DataFrame({"a": [5., 4., np.nan, 3., 7.],
                    "b": [np.nan, 3., 4., 6., 8.]})
df1
df2
df1.combine_first(df2)


# In[52]:


#! ipython id=647c8bf32f9f4f68818781a117aec2a6
data = pd.DataFrame(np.arange(6).reshape((2, 3)),
                    index=pd.Index(["Ohio", "Colorado"], name="state"),
                    columns=pd.Index(["one", "two", "three"],
                    name="number"))
data


# In[53]:


#! ipython id=93b2cac6e12c4c4f84f63acea8ae9997
result = data.stack()
result


# In[54]:


#! ipython id=a4b865ed812a43c39ba96e004b7f58f0
result.unstack()


# In[55]:


#! ipython id=1381976b93554a30bca3f97ff21cef4a
result.unstack(level=0)
result.unstack(level="state")


# In[56]:


#! ipython id=c26a7396ad494449b09e718e2d9ed336
s1 = pd.Series([0, 1, 2, 3], index=["a", "b", "c", "d"], dtype="Int64")
s2 = pd.Series([4, 5, 6], index=["c", "d", "e"], dtype="Int64")
data2 = pd.concat([s1, s2], keys=["one", "two"])
data2


# In[57]:


#! ipython id=8d7eaaf573f3469a82a555f0dd73665c
data2.unstack()
data2.unstack().stack()
data2.unstack().stack(dropna=False)


# In[58]:


#! ipython id=f3e7f3eace4e497097fb20f56ec70436
df = pd.DataFrame({"left": result, "right": result + 5},
                  columns=pd.Index(["left", "right"], name="side"))
df
df.unstack(level="state")


# In[59]:


#! ipython id=62e63632842c40e7ab3913dffdb17734
df.unstack(level="state").stack(level="side")


# In[60]:


#! ipython id=7691f5f737a44b39a67861744c98d2ae
data = pd.read_csv("examples/macrodata.csv")
data = data.loc[:, ["year", "quarter", "realgdp", "infl", "unemp"]]
data.head()


# In[61]:


#! ipython id=cb853f8845dc4808b3bb2cd2697cbb14
periods = pd.PeriodIndex(year=data.pop("year"),
                         quarter=data.pop("quarter"),
                         name="date")
periods
data.index = periods.to_timestamp("D")
data.head()


# In[62]:


#! ipython id=a50d1ca51a9c4a8591f2ab4bb824d84b
data = data.reindex(columns=["realgdp", "infl", "unemp"])
data.columns.name = "item"
data.head()


# In[63]:


#! ipython id=1ccfe1d1bd2c4d3face047a0659bf635
long_data = (data.stack()
             .reset_index()
             .rename(columns={0: "value"}))


# In[64]:


#! ipython id=4005e426f21b47bead4aea340933266a
long_data[:10]


# In[65]:


#! ipython id=877e33a3f6114aa0966c3d68a139701c
pivoted = long_data.pivot(index="date", columns="item",
                          values="value")
pivoted.head()


# In[66]:


#! ipython id=b88155754cf540f7be5662ce714b0d12
long_data["value2"] = np.random.standard_normal(len(long_data))
long_data[:10]


# In[67]:


#! ipython id=112caaf289fa40ed9f88aba26faedfbe
pivoted = long_data.pivot(index="date", columns="item")
pivoted.head()
pivoted["value"].head()


# In[68]:


#! ipython id=2b53be975f41406f86b9a19cc1a04c9c
unstacked = long_data.set_index(["date", "item"]).unstack(level="item")
unstacked.head()


# In[69]:


#! ipython suppress id=f14d4b5d2fb44feb92f41233b5f019bc
get_ipython().run_line_magic('popd', '')


# In[70]:


#! ipython id=f95fff0c51f149dc88462804af6b5038
df = pd.DataFrame({"key": ["foo", "bar", "baz"],
                   "A": [1, 2, 3],
                   "B": [4, 5, 6],
                   "C": [7, 8, 9]})
df


# In[71]:


#! ipython id=30b180a4722e4db9af5953c471f6c603
melted = pd.melt(df, id_vars="key")
melted


# In[72]:


#! ipython id=7969c1c5a676429b82ccbe31c997fef4
reshaped = melted.pivot(index="key", columns="variable",
                        values="value")
reshaped


# In[73]:


#! ipython id=53f7023267ac43f399635dabee6b430f
reshaped.reset_index()


# In[74]:


#! ipython id=0d5675ca7ba94da3bc9168d3d9b7a47e
pd.melt(df, id_vars="key", value_vars=["A", "B"])


# In[75]:


#! ipython id=a39fe88030144c79a2ce8c7e7ec566d8
pd.melt(df, value_vars=["A", "B", "C"])
pd.melt(df, value_vars=["key", "A", "B"])

