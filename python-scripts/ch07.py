#!/usr/bin/env python
# coding: utf-8

# In[1]:


#! ipython suppress id=a299071e366b4bd590198b85b8625f2b
get_ipython().run_line_magic('pushd', 'book-materials')
import numpy as np
import pandas as pd
PREVIOUS_MAX_ROWS = pd.options.display.max_rows
pd.options.display.max_rows = 25
pd.options.display.max_columns = 20
pd.options.display.max_colwidth = 82
np.random.seed(12345)
import matplotlib.pyplot as plt
plt.rc("figure", figsize=(10, 6))
np.set_printoptions(precision=4, suppress=True)


# In[2]:


#! ipython suppress id=835e9c4e6d3942638244064db595a5bd
import numpy as np
import pandas as pd


# In[3]:


#! ipython id=290a88142afe4854bc724c7a6c735bae
float_data = pd.Series([1.2, -3.5, np.nan, 0])
float_data


# In[4]:


#! ipython id=4b91ad05aea045cdbf13b5900cc1f73e
float_data.isna()


# In[5]:


#! ipython id=8996ec87330f486b96f59f1f394ccf6a
string_data = pd.Series(["aardvark", np.nan, None, "avocado"])
string_data
string_data.isna()
float_data = pd.Series([1, 2, None], dtype='float64')
float_data
float_data.isna()


# In[6]:


#! ipython id=6dfb795c00c24da9a66cdd66870f9fa9
data = pd.Series([1, np.nan, 3.5, np.nan, 7])
data.dropna()


# In[7]:


#! ipython id=ce4c091e5f824312b06c6b1d6ff531fb
data[data.notna()]


# In[8]:


#! ipython id=be32168b85724034b62a96f6fb6e7f2c
data = pd.DataFrame([[1., 6.5, 3.], [1., np.nan, np.nan],
                     [np.nan, np.nan, np.nan], [np.nan, 6.5, 3.]])
data
data.dropna()


# In[9]:


#! ipython id=fc78d6354b4541f3a1c2416275e68c76
data.dropna(how="all")


# In[10]:


#! ipython id=87ba756b91eb4dd09d1b7e647a8ae7b9
data[4] = np.nan
data
data.dropna(axis="columns", how="all")


# In[11]:


#! ipython id=93bc1cf6fc10429a924cea596353e388
df = pd.DataFrame(np.random.standard_normal((7, 3)))
df.iloc[:4, 1] = np.nan
df.iloc[:2, 2] = np.nan
df
df.dropna()
df.dropna(thresh=2)


# In[12]:


#! ipython id=ceee3ece49cc4169a1f8ac1e6954accd
df.fillna(0)


# In[13]:


#! ipython id=a03bb4bc12e24a3e941daa79068471ff
df.fillna({1: 0.5, 2: 0})


# In[14]:


#! ipython id=bf828716b9e04ef5a37200b5d5086b47
df = pd.DataFrame(np.random.standard_normal((6, 3)))
df.iloc[2:, 1] = np.nan
df.iloc[4:, 2] = np.nan
df
df.fillna(method="ffill")
df.fillna(method="ffill", limit=2)


# In[15]:


#! ipython id=77b2b4714bd14dffb5e6b407fd0df952
data = pd.Series([1., np.nan, 3.5, np.nan, 7])
data.fillna(data.mean())


# In[16]:


#! ipython id=944377d16cdc466cb0a9e64fa1f79d5b
data = pd.DataFrame({"k1": ["one", "two"] * 3 + ["two"],
                     "k2": [1, 1, 2, 3, 3, 4, 4]})
data


# In[17]:


#! ipython id=bd2994de11fd4012a835bd0ee2bcf4cb
data.duplicated()


# In[18]:


#! ipython id=ac1c8d333c8d40068f33556cee5cb8b4
data.drop_duplicates()


# In[19]:


#! ipython id=ac73e11080b04551bdf1a23729e75793
data["v1"] = range(7)
data
data.drop_duplicates(subset=["k1"])


# In[20]:


#! ipython id=eb735c97c9b744aa923b8fb94ae22f64
data.drop_duplicates(["k1", "k2"], keep="last")


# In[21]:


#! ipython id=4fd09f7a425741d5b1f61a28f2ec73a2
data = pd.DataFrame({"food": ["bacon", "pulled pork", "bacon",
                              "pastrami", "corned beef", "bacon",
                              "pastrami", "honey ham", "nova lox"],
                     "ounces": [4, 3, 12, 6, 7.5, 8, 3, 5, 6]})
data


# In[22]:


#! ipython verbatim id=1459d863e4e2488884aa21b5841fa88d
meat_to_animal = {
  "bacon": "pig",
  "pulled pork": "pig",
  "pastrami": "cow",
  "corned beef": "cow",
  "honey ham": "pig",
  "nova lox": "salmon"
}


# In[23]:


#! ipython id=a779fa7330de487fb907e801bf054696
data["animal"] = data["food"].map(meat_to_animal)
data


# In[24]:


#! ipython id=4c24af18964744eda625a480f5cdd126
def get_animal(x):
    return meat_to_animal[x]
data["food"].map(get_animal)


# In[25]:


#! ipython id=ec9f0e43f8da42ed8b3119331d09c853
data = pd.Series([1., -999., 2., -999., -1000., 3.])
data


# In[26]:


#! ipython id=8fd47d92ab854f47bed89d8a72c9d8a0
data.replace(-999, np.nan)


# In[27]:


#! ipython id=6bd3f8909caf4eed928f7164f277f8ef
data.replace([-999, -1000], np.nan)


# In[28]:


#! ipython id=467da543042d473aa9ed8b43e5daefec
data.replace([-999, -1000], [np.nan, 0])


# In[29]:


#! ipython id=e3dfba217f5a460d9bdf04ba573481ad
data.replace({-999: np.nan, -1000: 0})


# In[30]:


#! ipython id=103a514cd451404fbc7b559bab5ea87c
data = pd.DataFrame(np.arange(12).reshape((3, 4)),
                    index=["Ohio", "Colorado", "New York"],
                    columns=["one", "two", "three", "four"])


# In[31]:


#! ipython id=aabdeea6a25f49a5a6949527b7d345bb
def transform(x):
    return x[:4].upper()

data.index.map(transform)


# In[32]:


#! ipython id=8923ced04de44246b6d710395cc93348
data.index = data.index.map(transform)
data


# In[33]:


#! ipython id=d125b3401a124c89a83007e36e3ee758
data.rename(index=str.title, columns=str.upper)


# In[34]:


#! ipython id=5f8e0d6ff0b4445d886c8ee0fb478191
data.rename(index={"OHIO": "INDIANA"},
            columns={"three": "peekaboo"})


# In[35]:


#! ipython id=5b066323cc4944c9938b36f4f8d52779
ages = [20, 22, 25, 27, 21, 23, 37, 31, 61, 45, 41, 32]


# In[36]:


#! ipython id=075935d1c34b41ba9ee1842e23605ad1
bins = [18, 25, 35, 60, 100]
age_categories = pd.cut(ages, bins)
age_categories


# In[37]:


#! ipython id=ce9e07485ac641e291688784a05e841c
age_categories.codes
age_categories.categories
age_categories.categories[0]
pd.value_counts(age_categories)


# In[38]:


#! ipython id=c40f7922de224c5682b37159bfb13adf
pd.cut(ages, bins, right=False)


# In[39]:


#! ipython id=5d5d56eba8384d11bd5d38c378c9fbf1
group_names = ["Youth", "YoungAdult", "MiddleAged", "Senior"]
pd.cut(ages, bins, labels=group_names)


# In[40]:


#! ipython id=33190ad9f1834874ba7cbd4066d8350b
data = np.random.uniform(size=20)
pd.cut(data, 4, precision=2)


# In[41]:


#! ipython id=5aa787199f2745d1a2358a097e9b23d9
data = np.random.standard_normal(1000)
quartiles = pd.qcut(data, 4, precision=2)
quartiles
pd.value_counts(quartiles)


# In[42]:


#! ipython id=6df8a2f012ad4735bbc92d478d946e1f
pd.qcut(data, [0, 0.1, 0.5, 0.9, 1.]).value_counts()


# In[43]:


#! ipython id=3979599fd2d247b7a32bf203a5c9a5c7
data = pd.DataFrame(np.random.standard_normal((1000, 4)))
data.describe()


# In[44]:


#! ipython id=14662c0a588c488d83d792107f20e528
col = data[2]
col[col.abs() > 3]


# In[45]:


#! ipython id=2a1bb1c468cf40a58f3c5223a992b59b
data[(data.abs() > 3).any(axis="columns")]


# In[46]:


#! ipython id=694246321bbd42adbb221b99486f388b
data[data.abs() > 3] = np.sign(data) * 3
data.describe()


# In[47]:


#! ipython id=d70c02173fe945abad1450a5709b520e
np.sign(data).head()


# In[48]:


#! ipython id=70d08ca5f48d48018126fd9ab1006314
df = pd.DataFrame(np.arange(5 * 7).reshape((5, 7)))
df
sampler = np.random.permutation(5)
sampler


# In[49]:


#! ipython id=f7c2008d514b4a549645c2c8a28861d9
df.take(sampler)
df.iloc[sampler]


# In[50]:


#! ipython id=ffbccf111b60408395f9eb116f6fc39a
column_sampler = np.random.permutation(7)
column_sampler
df.take(column_sampler, axis="columns")


# In[51]:


#! ipython id=c2429f2bce484acca0f590eed87cb896
df.sample(n=3)


# In[52]:


#! ipython id=f98970c0ba4942d880cf0d4076960bc3
choices = pd.Series([5, 7, -1, 6, 4])
choices.sample(n=10, replace=True)


# In[53]:


#! ipython id=590105f5b54b4b798b9c925339f1c000
df = pd.DataFrame({"key": ["b", "b", "a", "c", "a", "b"],
                   "data1": range(6)})
df
pd.get_dummies(df["key"])


# In[54]:


#! ipython id=c9bf047cafa2489cb537d963eb18d16a
dummies = pd.get_dummies(df["key"], prefix="key")
df_with_dummy = df[["data1"]].join(dummies)
df_with_dummy


# In[55]:


#! ipython allow_exceptions id=e311a84ee00c40479411ed79fc5a6c69
mnames = ["movie_id", "title", "genres"]
movies = pd.read_table("datasets/movielens/movies.dat", sep="::",
                       header=None, names=mnames, engine="python")
movies[:10]


# In[56]:


#! ipython id=8e085399c22341ee9632bf5efcb4a6dc
dummies = movies["genres"].str.get_dummies("|")
dummies.iloc[:10, :6]


# In[57]:


#! ipython id=9b2a0b211d424f6798485b72bd572e08
movies_windic = movies.join(dummies.add_prefix("Genre_"))
movies_windic.iloc[0]


# In[58]:


#! ipython id=f3df965a3e024e918c1e67e647a6589a
np.random.seed(12345) # to make the example repeatable
values = np.random.uniform(size=10)
values
bins = [0, 0.2, 0.4, 0.6, 0.8, 1]
pd.get_dummies(pd.cut(values, bins))


# In[59]:


#! ipython id=10771102e8084f5fa4a16200a93d058d
s = pd.Series([1, 2, 3, None])
s
s.dtype


# In[60]:


#! ipython id=66923eb117b34b21b548507e349d387d
s = pd.Series([1, 2, 3, None], dtype=pd.Int64Dtype())
s
s.isna()
s.dtype


# In[61]:


#! ipython id=8a80094d8808433599836ea7197e55b9
s[3]
s[3] is pd.NA


# In[62]:


#! ipython id=43b53a40545c451c94728b65b40c574e
s = pd.Series([1, 2, 3, None], dtype="Int64")


# In[63]:


#! ipython id=85af440e08ae45d9b570c8b65f496567
s = pd.Series(['one', 'two', None, 'three'], dtype=pd.StringDtype())
s


# In[64]:


#! ipython id=81a4d68317534367bdbd458b98a55329
df = pd.DataFrame({"A": [1, 2, None, 4],
                   "B": ["one", "two", "three", None],
                   "C": [False, None, False, True]})
df
df["A"] = df["A"].astype("Int64")
df["B"] = df["B"].astype("string")
df["C"] = df["C"].astype("boolean")
df


# In[65]:


#! ipython id=fe9efc6724224b7b9eccd32750524def
val = "a,b,  guido"
val.split(",")


# In[66]:


#! ipython id=9724ab6a052943e7803c2f855b47b29f
pieces = [x.strip() for x in val.split(",")]
pieces


# In[67]:


#! ipython id=e80b9a9828014622b86c088759d5e5a0
first, second, third = pieces
first + "::" + second + "::" + third


# In[68]:


#! ipython id=06baf1c7f132476f90f73524278f5972
"::".join(pieces)


# In[69]:


#! ipython id=d5571ea9402c4182af832d4b0b659ad4
"guido" in val
val.index(",")
val.find(":")


# In[70]:


#! ipython allow_exceptions id=ee7ecc7978534b3883cf050a76245ba3
val.index(":")


# In[71]:


#! ipython id=6c3ac8963e0348d5900d3b21f3d61421
val.count(",")


# In[72]:


#! ipython id=2385159093a248abbb0b590843359737
val.replace(",", "::")
val.replace(",", "")


# In[73]:


#! ipython id=b430b6ee3c904fa38a207e995ad4e152
import re
text = "foo    bar\t baz  \tqux"
re.split(r"\s+", text)


# In[74]:


#! ipython id=2af33f0f709d42a2853daaabafede024
regex = re.compile(r"\s+")
regex.split(text)


# In[75]:


#! ipython id=746b726a56dd4375a8a459b74d65789c
regex.findall(text)


# In[76]:


#! ipython verbatim id=8f2d09408b41491aa269044fc4375d93
text = """Dave dave@google.com
Steve steve@gmail.com
Rob rob@gmail.com
Ryan ryan@yahoo.com"""
pattern = r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,4}"

# re.IGNORECASE makes the regex case insensitive
regex = re.compile(pattern, flags=re.IGNORECASE)


# In[77]:


#! ipython id=4efa726d3118445db3dfa63e69147a3e
regex.findall(text)


# In[78]:


#! ipython id=5d3ea127f5064a0eabafa8f07982c0fe
m = regex.search(text)
m
text[m.start():m.end()]


# In[79]:


#! ipython id=56352eb1b66041658dc4a2df7cdd3659
print(regex.match(text))


# In[80]:


#! ipython id=16b2b19e1790421f8cbfd96272f49964
print(regex.sub("REDACTED", text))


# In[81]:


#! ipython id=48a7da4e764149fc9fbbc73d8836e703
pattern = r"([A-Z0-9._%+-]+)@([A-Z0-9.-]+)\.([A-Z]{2,4})"
regex = re.compile(pattern, flags=re.IGNORECASE)


# In[82]:


#! ipython id=0e0269c4dfb748a787340d99bd51af16
m = regex.match("wesm@bright.net")
m.groups()


# In[83]:


#! ipython id=ba6371f22b31476d99904236e025caf2
regex.findall(text)


# In[84]:


#! ipython id=adf56cb7344246a4981c9ccbc4520e45
print(regex.sub(r"Username: \1, Domain: \2, Suffix: \3", text))


# In[85]:


#! ipython id=d2dae337d0724a7cabb69f0d2e45d21a
data = {"Dave": "dave@google.com", "Steve": "steve@gmail.com",
        "Rob": "rob@gmail.com", "Wes": np.nan}
data = pd.Series(data)
data
data.isna()


# In[86]:


#! ipython id=b1a32c08159845ad97915762d2e9104c
data.str.contains("gmail")


# In[87]:


#! ipython id=e83b7f3cb11b42ea8344da39bc8313ed
data_as_string_ext = data.astype('string')
data_as_string_ext
data_as_string_ext.str.contains("gmail")


# In[88]:


#! ipython id=abd3d1bd7239406d8d764f812cb04018
pattern = r"([A-Z0-9._%+-]+)@([A-Z0-9.-]+)\.([A-Z]{2,4})"
data.str.findall(pattern, flags=re.IGNORECASE)


# In[89]:


#! ipython id=07c02acaf03844578f9b18af036e96b4
matches = data.str.findall(pattern, flags=re.IGNORECASE).str[0]
matches
matches.str.get(1)


# In[90]:


#! ipython id=140f72f7ebdf44fc832ac76596f1f500
data.str[:5]


# In[91]:


#! ipython id=086c74fc48364b859ab30d4c9278e96d
data.str.extract(pattern, flags=re.IGNORECASE)


# In[92]:


#! ipython id=5e0d1d0145d64d80870482f06001910e
values = pd.Series(['apple', 'orange', 'apple',
                    'apple'] * 2)
values
pd.unique(values)
pd.value_counts(values)


# In[93]:


#! ipython id=6b72957b5a1c4882828b5b6fc55bc645
values = pd.Series([0, 1, 0, 0] * 2)
dim = pd.Series(['apple', 'orange'])
values
dim


# In[94]:


#! ipython id=c4c6450bbb7f4c74b579d4cc989d0547
dim.take(values)


# In[95]:


#! ipython id=922c7a3dfaf5478c8ece876bf6e8c8a4
fruits = ['apple', 'orange', 'apple', 'apple'] * 2
N = len(fruits)
rng = np.random.default_rng(seed=12345)
df = pd.DataFrame({'fruit': fruits,
                   'basket_id': np.arange(N),
                   'count': rng.integers(3, 15, size=N),
                   'weight': rng.uniform(0, 4, size=N)},
                  columns=['basket_id', 'fruit', 'count', 'weight'])
df


# In[96]:


#! ipython id=d3efa50a15b64f78b8ee5d1bffd9e3db
fruit_cat = df['fruit'].astype('category')
fruit_cat


# In[97]:


#! ipython id=5b3590609c7f4a2984add7aefc62727c
c = fruit_cat.array
type(c)


# In[98]:


#! ipython id=4d5fd7e39e7b4dc4bf59a7597a66cbe1
c.categories
c.codes


# In[99]:


#! ipython id=a5d2afb03a3d46fe99b79633b5480fb7
dict(enumerate(c.categories))


# In[100]:


#! ipython id=b4e849ae87304ac1a2b717ecafbf58c4
df['fruit'] = df['fruit'].astype('category')
df["fruit"]


# In[101]:


#! ipython id=adba8c41d0094e49943f4c0f03bfb560
my_categories = pd.Categorical(['foo', 'bar', 'baz', 'foo', 'bar'])
my_categories


# In[102]:


#! ipython id=dbb5b2424ba146f0b001398542f64ae3
categories = ['foo', 'bar', 'baz']
codes = [0, 1, 2, 0, 0, 1]
my_cats_2 = pd.Categorical.from_codes(codes, categories)
my_cats_2


# In[103]:


#! ipython id=2ee990370ede41ddbcf285a6c1330ee0
ordered_cat = pd.Categorical.from_codes(codes, categories,
                                        ordered=True)
ordered_cat


# In[104]:


#! ipython id=27e720c7999143b8bc6fa95797bb78fd
my_cats_2.as_ordered()


# In[105]:


#! ipython id=3268082ea833473888cb6df910208de8
rng = np.random.default_rng(seed=12345)
draws = rng.standard_normal(1000)
draws[:5]


# In[106]:


#! ipython id=6d7daaf08a8240dea51848a3a6cc211d
bins = pd.qcut(draws, 4)
bins


# In[107]:


#! ipython id=6faabc704b9c436fa0e32ac6755e6359
bins = pd.qcut(draws, 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
bins
bins.codes[:10]


# In[108]:


#! ipython id=0f2080e060dd4d848ff1a9d846c8a71e
bins = pd.Series(bins, name='quartile')
results = (pd.Series(draws)
           .groupby(bins)
           .agg(['count', 'min', 'max'])
           .reset_index())
results


# In[109]:


#! ipython id=4d1f92dfe8e34a6d8173c0aca18d8b55
results['quartile']


# In[110]:


#! ipython id=cb512af2f6bf4eb5b312ae330bd4c91e
N = 10_000_000
labels = pd.Series(['foo', 'bar', 'baz', 'qux'] * (N // 4))


# In[111]:


#! ipython id=2978248572a743be8529a1668d424e8a
categories = labels.astype('category')


# In[112]:


#! ipython id=a16f1610329347ffa2f28661941350b7
labels.memory_usage(deep=True)
categories.memory_usage(deep=True)


# In[113]:


#! ipython id=dd7310d96d4d4c24bb7e3e9ad26b066f
get_ipython().run_line_magic('time', "_ = labels.astype('category')")


# In[114]:


#! ipython id=aca60c356d264f7c872287f1bfdf75c9
get_ipython().run_line_magic('timeit', 'labels.value_counts()')
get_ipython().run_line_magic('timeit', 'categories.value_counts()')


# In[115]:


#! ipython id=9dd1c0395c0743db8a283da215ae7f50
s = pd.Series(['a', 'b', 'c', 'd'] * 2)
cat_s = s.astype('category')
cat_s


# In[116]:


#! ipython id=84308427acbc491381531a16cd3bcc15
cat_s.cat.codes
cat_s.cat.categories


# In[117]:


#! ipython id=7b07ac0b91ef42809b1183ccc41555c1
actual_categories = ['a', 'b', 'c', 'd', 'e']
cat_s2 = cat_s.cat.set_categories(actual_categories)
cat_s2


# In[118]:


#! ipython id=919709709a4c4c8d93e135d137ea0757
cat_s.value_counts()
cat_s2.value_counts()


# In[119]:


#! ipython id=0350789cd1fe41c6908b5e604ea34eb0
cat_s3 = cat_s[cat_s.isin(['a', 'b'])]
cat_s3
cat_s3.cat.remove_unused_categories()


# In[120]:


#! ipython id=c6b7347f928a4580b5dac67f798d9e28
cat_s = pd.Series(['a', 'b', 'c', 'd'] * 2, dtype='category')


# In[121]:


#! ipython id=95950502e1bc4cfba88e7891c8062ee6
pd.get_dummies(cat_s)


# In[122]:


#! ipython suppress id=b8746f91160c4c9da000aeb4891514bf
get_ipython().run_line_magic('popd', '')


# In[123]:


#! ipython suppress id=621a2c4436a44e5db40091b1e4f8a167
pd.options.display.max_rows = PREVIOUS_MAX_ROWS

