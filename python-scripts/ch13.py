#!/usr/bin/env python
# coding: utf-8

# In[1]:


#! ipython suppress id=a26af182eb1f4597a9dbf174cdc570b9
get_ipython().run_line_magic('pushd', 'book-materials')

from numpy.random import randn
import numpy as np
np.random.seed(123)
import os
import matplotlib.pyplot as plt
import pandas as pd
plt.rc("figure", figsize=(10, 6))
np.set_printoptions(precision=4)
pd.options.display.max_columns = 20
pd.options.display.max_rows = 20
pd.options.display.max_colwidth = 80


# In[2]:


#! ipython suppress id=5f3e64ebebd744a4a972fce4303d263a
path = "datasets/bitly_usagov/example.txt"


# In[3]:


#! ipython verbatim id=74b843e3047b47d09a832031cd7a3753
import json
with open(path) as f:
    records = [json.loads(line) for line in f]


# In[4]:


#! ipython allow_exceptions id=b0d4c36e296a48e4b5bfcc7b5a0b5926
time_zones = [rec["tz"] for rec in records]


# In[5]:


#! ipython id=9bcb262c06f14a3aa36dbd73c2f33444
time_zones = [rec["tz"] for rec in records if "tz" in rec]
time_zones[:10]


# In[6]:


#! ipython verbatim id=64a59d263c4c4e8ba0617426c1b21573
def get_counts(sequence):
    counts = {}
    for x in sequence:
        if x in counts:
            counts[x] += 1
        else:
            counts[x] = 1
    return counts


# In[7]:


#! ipython verbatim id=b55b0800f7274320b5a7f97d93719a2d
from collections import defaultdict

def get_counts2(sequence):
    counts = defaultdict(int) # values will initialize to 0
    for x in sequence:
        counts[x] += 1
    return counts


# In[8]:


#! ipython id=ba1751a9d2244f46841fd9afbae7aa04
counts = get_counts(time_zones)
counts["America/New_York"]
len(time_zones)


# In[9]:


#! ipython verbatim id=562d875fd9704dc6a3614303c9bc7a8e
def top_counts(count_dict, n=10):
    value_key_pairs = [(count, tz) for tz, count in count_dict.items()]
    value_key_pairs.sort()
    return value_key_pairs[-n:]


# In[10]:


#! ipython id=d061b4cb313c469e94543974462530ba
top_counts(counts)


# In[11]:


#! ipython id=ab2efb020a1846699974cea106f89ba2
from collections import Counter
counts = Counter(time_zones)
counts.most_common(10)


# In[12]:


#! ipython id=8abcc73b62f5486a9a448c1e742c5f92
frame = pd.DataFrame(records)


# In[13]:


#! ipython id=58b54b37efa04ef9ad317bc29a1e8e2e
frame.info()
frame["tz"].head()


# In[14]:


#! ipython id=9694e9d51c944b9882129e348f79ae0c
tz_counts = frame["tz"].value_counts()
tz_counts.head()


# In[15]:


#! ipython id=b445fbf49098428fbc93855c9aa93571
clean_tz = frame["tz"].fillna("Missing")
clean_tz[clean_tz == ""] = "Unknown"
tz_counts = clean_tz.value_counts()
tz_counts.head()


# In[16]:


#! ipython suppress id=f156ad96962842e6ae79658999ea7134
plt.figure(figsize=(10, 4))


# In[17]:


#! ipython id=ea12bafce448419c90a8a118e018458f
import seaborn as sns
subset = tz_counts.head()
#! figure,id=usa_gov_counts,title="Top time zones in the 1.usa.gov sample data"
sns.barplot(y=subset.index, x=subset.to_numpy())


# In[18]:


#! ipython id=32e36ee621664722babaffe83900dff2
frame["a"][1]
frame["a"][50]
frame["a"][51][:50]  # long line


# In[19]:


#! ipython id=973a67b5d25d41fb8c7d53ddcfbd747f
results = pd.Series([x.split()[0] for x in frame["a"].dropna()])
results.head(5)
results.value_counts().head(8)


# In[20]:


#! ipython id=7ef631a4289942378d19f5bc510b93e1
cframe = frame[frame["a"].notna()].copy()


# In[21]:


#! ipython id=334e7d52678f48a28af3f14e7672b962
cframe["os"] = np.where(cframe["a"].str.contains("Windows"),
                        "Windows", "Not Windows")
cframe["os"].head(5)


# In[22]:


#! ipython id=c8beb8e211f741bdb3680861c6bed49a
by_tz_os = cframe.groupby(["tz", "os"])


# In[23]:


#! ipython id=e62bf85f81fd46f1a5d1f00399ae31f7
agg_counts = by_tz_os.size().unstack().fillna(0)
agg_counts.head()


# In[24]:


#! ipython id=684389ffe6c0421d9ced56415c1baeb9
indexer = agg_counts.sum("columns").argsort()
indexer.values[:10]


# In[25]:


#! ipython id=e861e46cf4b34f7690bbcbfbc490450c
count_subset = agg_counts.take(indexer[-10:])
count_subset


# In[26]:


#! ipython id=836cfdd2d671421f9e58b031cf142e3f
agg_counts.sum(axis="columns").nlargest(10)


# In[27]:


#! ipython suppress id=3edfe74f82f247cc99e3e4d7d4907a0c
plt.figure()


# In[28]:


#! ipython id=a194820b3f03419e844991ecdfaa3236
count_subset = count_subset.stack()
count_subset.name = "total"
count_subset = count_subset.reset_index()
count_subset.head(10)
#! figure,id=usa_gov_tz_os,title="Top time zones by Windows and non-Windows users"
sns.barplot(x="total", y="tz", hue="os",  data=count_subset)


# In[29]:


#! ipython verbatim id=1cf7faf398ca47e2b0c97b72beff6256
def norm_total(group):
    group["normed_total"] = group["total"] / group["total"].sum()
    return group

results = count_subset.groupby("tz").apply(norm_total)


# In[30]:


#! ipython suppress id=e7e063a913d244ff939d923e84f2f905
plt.figure()


# In[31]:


#! ipython id=512403c2efd149b1be89649e70dde50a
#! figure,id=usa_gov_tz_os_normed,title="Percentage Windows and non-Windows users in top occurring time zones"
sns.barplot(x="normed_total", y="tz", hue="os",  data=results)


# In[32]:


#! ipython id=8e6df52dcfd144489941522b36b5a32a
g = count_subset.groupby("tz")
results2 = count_subset["total"] / g["total"].transform("sum")


# In[33]:


#! ipython verbatim id=f5f892f460124f2499e6e040ad28239a
unames = ["user_id", "gender", "age", "occupation", "zip"]
users = pd.read_table("datasets/movielens/users.dat", sep="::",
                      header=None, names=unames, engine="python")

rnames = ["user_id", "movie_id", "rating", "timestamp"]
ratings = pd.read_table("datasets/movielens/ratings.dat", sep="::",
                        header=None, names=rnames, engine="python")

mnames = ["movie_id", "title", "genres"]
movies = pd.read_table("datasets/movielens/movies.dat", sep="::",
                       header=None, names=mnames, engine="python")


# In[34]:


#! ipython id=c53d153be32f4a31af7adcc70ba414a6
users.head(5)
ratings.head(5)
movies.head(5)
ratings


# In[35]:


#! ipython id=326328c171c349389adb61b8e687355e
data = pd.merge(pd.merge(ratings, users), movies)
data
data.iloc[0]


# In[36]:


#! ipython id=e5129c0aac3a400f91f247cb27ae9c3f
mean_ratings = data.pivot_table("rating", index="title",
                                columns="gender", aggfunc="mean")
mean_ratings.head(5)


# In[37]:


#! ipython id=3e174af9b5e342b38519d1b0ce2802a4
ratings_by_title = data.groupby("title").size()
ratings_by_title.head()
active_titles = ratings_by_title.index[ratings_by_title >= 250]
active_titles


# In[38]:


#! ipython id=083e410e3d7445baaddbb46d6760c956
mean_ratings = mean_ratings.loc[active_titles]
mean_ratings


# In[39]:


#! ipython suppress id=3920a865f3684ea6ab421a78dc0bcba4
mean_ratings = mean_ratings.rename(index={"Seven Samurai (The Magnificent Seven) (Shichinin no samurai) (1954)":
                           "Seven Samurai (Shichinin no samurai) (1954)"})


# In[40]:


#! ipython id=a7405fad58a444e283a9c70691b2cd1a
top_female_ratings = mean_ratings.sort_values("F", ascending=False)
top_female_ratings.head()


# In[41]:


#! ipython id=30f629f297c14048b06165aabeeed519
mean_ratings["diff"] = mean_ratings["M"] - mean_ratings["F"]


# In[42]:


#! ipython id=efe335603fda4b159496e6fc78c8546b
sorted_by_diff = mean_ratings.sort_values("diff")
sorted_by_diff.head()


# In[43]:


#! ipython id=d07b8e30ed1f4252a346e493dc775b4d
sorted_by_diff[::-1].head()


# In[44]:


#! ipython id=ce7c8a289d4942079cc643a7a30fe51d
rating_std_by_title = data.groupby("title")["rating"].std()
rating_std_by_title = rating_std_by_title.loc[active_titles]
rating_std_by_title.head()


# In[45]:


#! ipython id=1b53af1f9cdc44c7ae77c8a6a10e5780
rating_std_by_title.sort_values(ascending=False)[:10]


# In[46]:


#! ipython id=79b147a65ea34ffcbe2d799331fcba6d
movies["genres"].head()
movies["genres"].head().str.split("|")
movies["genre"] = movies.pop("genres").str.split("|")
movies.head()


# In[47]:


#! ipython id=3a1765e4004f467abde25424f4f68a19
movies_exploded = movies.explode("genre")
movies_exploded[:10]


# In[48]:


#! ipython id=12e06c516e144975bb4f6b33a19d3d17
ratings_with_genre = pd.merge(pd.merge(movies_exploded, ratings), users)
ratings_with_genre.iloc[0]
genre_ratings = (ratings_with_genre.groupby(["genre", "age"])
                 ["rating"].mean()
                 .unstack("age"))
genre_ratings[:10]


# In[49]:


#! ipython id=0b02ebafbf174f5b8b5b8487f314857a
get_ipython().system('head -n 10 datasets/babynames/yob1880.txt')


# In[50]:


#! ipython id=96d7c450b52749f6adc841c2ae245abf
names1880 = pd.read_csv("datasets/babynames/yob1880.txt",
                        names=["name", "sex", "births"])
names1880


# In[51]:


#! ipython id=93704598475f4ace96b41290892a8e22
names1880.groupby("sex")["births"].sum()


# In[52]:


#! ipython verbatim id=c1e1153045b246e6b46b81ed6d740365
pieces = []
for year in range(1880, 2011):
    path = f"datasets/babynames/yob{year}.txt"
    frame = pd.read_csv(path, names=["name", "sex", "births"])

    # Add a column for the year
    frame["year"] = year
    pieces.append(frame)

# Concatenate everything into a single DataFrame
names = pd.concat(pieces, ignore_index=True)


# In[53]:


#! ipython id=2d1371e5fc5b4989bfdcb50c224f6ecf
names


# In[54]:


#! ipython id=49e2f22b602741ff86b0e01def6297b5
total_births = names.pivot_table("births", index="year",
                                 columns="sex", aggfunc=sum)
total_births.tail()
#! figure,id=baby_names_total_births,title="Total births by sex and year"
total_births.plot(title="Total births by sex and year")


# In[55]:


#! ipython verbatim id=abb6e87b71ef43b0906c515cbf44ba33
def add_prop(group):
    group["prop"] = group["births"] / group["births"].sum()
    return group
names = names.groupby(["year", "sex"]).apply(add_prop)


# In[56]:


#! ipython id=af43bd66df8a464a9acf179ea1a2a724
names


# In[57]:


#! ipython id=799eddd440ee4047ad4dc571bd534914
names.groupby(["year", "sex"])["prop"].sum()


# In[58]:


#! ipython id=d7cdfcaeec094222850cbd0a754e062d
def get_top1000(group):
    return group.sort_values("births", ascending=False)[:1000]
grouped = names.groupby(["year", "sex"])
top1000 = grouped.apply(get_top1000)
top1000.head()


# In[59]:


#! ipython id=b817c8be4dcf441d826d41f0a679a8f7
top1000 = top1000.reset_index(drop=True)


# In[60]:


#! ipython id=4018f9a519914774bfd558cd9ebfc13b
top1000.head()


# In[61]:


#! ipython id=df3701a69ebd4b3cbc438de86d549d93
boys = top1000[top1000["sex"] == "M"]
girls = top1000[top1000["sex"] == "F"]


# In[62]:


#! ipython id=e8e5983e77f741d7b1ad8fe607c9b869
total_births = top1000.pivot_table("births", index="year",
                                   columns="name",
                                   aggfunc=sum)


# In[63]:


#! ipython id=ec63edf8240a4b739fbf6fd57e6b3c15
total_births.info()
subset = total_births[["John", "Harry", "Mary", "Marilyn"]]
#! figure,id=baby_names_some_names,title="A few boy and girl names over time"
subset.plot(subplots=True, figsize=(12, 10),
            title="Number of births per year")


# In[64]:


#! ipython suppress id=4353ea3630a8454b94c5093acfb6e8d9
plt.figure()


# In[65]:


#! ipython id=686e195247ca4394b0b8047c4019297c
table = top1000.pivot_table("prop", index="year",
                            columns="sex", aggfunc=sum)
#! figure,id=baby_names_tot_prop,title="Proportion of births represented in top one thousand names by sex"
table.plot(title="Sum of table1000.prop by year and sex",
           yticks=np.linspace(0, 1.2, 13))


# In[66]:


#! ipython id=900fa540badb4cb9be8210d13b2ebee5
df = boys[boys["year"] == 2010]
df


# In[67]:


#! ipython id=4410fdd6262e462e982a01a98d55b34e
prop_cumsum = df["prop"].sort_values(ascending=False).cumsum()
prop_cumsum[:10]
prop_cumsum.searchsorted(0.5)


# In[68]:


#! ipython id=e7b5a7c97cb34f8487bdd77ead95eec7
df = boys[boys.year == 1900]
in1900 = df.sort_values("prop", ascending=False).prop.cumsum()
in1900.searchsorted(0.5) + 1


# In[69]:


#! ipython verbatim id=3275861aaba4432a8159d7004efee61a
def get_quantile_count(group, q=0.5):
    group = group.sort_values("prop", ascending=False)
    return group.prop.cumsum().searchsorted(q) + 1

diversity = top1000.groupby(["year", "sex"]).apply(get_quantile_count)
diversity = diversity.unstack()


# In[70]:


#! ipython suppress id=a1adda1ceb0e4783ba0f3709016bede2
fig = plt.figure()


# In[71]:


#! ipython id=d194e4ae2a2b417aaedec5dcbc12d3a5
diversity.head()
#! figure,id=baby_names_diversity_fig,title="Plot of diversity metric by year"
diversity.plot(title="Number of popular names in top 50%")


# In[72]:


#! ipython verbatim id=086dc90d927b41e99f9af7ecd87aadee
def get_last_letter(x):
    return x[-1]

last_letters = names["name"].map(get_last_letter)
last_letters.name = "last_letter"

table = names.pivot_table("births", index=last_letters,
                          columns=["sex", "year"], aggfunc=sum)


# In[73]:


#! ipython id=81379828617f46a8b289d83319498423
subtable = table.reindex(columns=[1910, 1960, 2010], level="year")
subtable.head()


# In[74]:


#! ipython id=cd458b92c4e14381a5380abc577f750a
subtable.sum()
letter_prop = subtable / subtable.sum()
letter_prop


# In[75]:


#! ipython verbatim id=cddad4c5c57d4c2b9a445f3eda5e625e
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(10, 8))
letter_prop["M"].plot(kind="bar", rot=0, ax=axes[0], title="Male")
letter_prop["F"].plot(kind="bar", rot=0, ax=axes[1], title="Female",
                      legend=False)


# In[76]:


#! ipython suppress id=dae34a4294ba4b7d8399d480222a8298
#! figure,id=baby_names_last_letter,title="Proportion of boy and girl names ending in each letter"
plt.subplots_adjust(hspace=0.25)


# In[77]:


#! ipython id=bf323785196547a0bed71781cfca7f5b
letter_prop = table / table.sum()

dny_ts = letter_prop.loc[["d", "n", "y"], "M"].T
dny_ts.head()


# In[78]:


#! ipython suppress id=44c3ae58c11e4723869a9a36aafa582f
plt.close("all")


# In[79]:


#! ipython suppress id=ecaa55eb4a5a4c6b98547da72ce97cc7
fig = plt.figure()


# In[80]:


#! ipython id=18558439cbec454696a8d60c22092fee
#! figure,id=baby_names_letter_over_time,title="Proportion of boys born with names ending in d/n/y over time"
dny_ts.plot()


# In[81]:


#! ipython id=1e66b95ed2d74763b32127da7c80ec51
all_names = pd.Series(top1000["name"].unique())
lesley_like = all_names[all_names.str.contains("Lesl")]
lesley_like


# In[82]:


#! ipython id=09ef1cec057c4361a6a765175b169d98
filtered = top1000[top1000["name"].isin(lesley_like)]
filtered.groupby("name")["births"].sum()


# In[83]:


#! ipython id=d4f30978887e471f9d5364c31a608faa
table = filtered.pivot_table("births", index="year",
                             columns="sex", aggfunc="sum")
table = table.div(table.sum(axis="columns"), axis="index")
table.tail()


# In[84]:


#! ipython suppress id=553d5251c8d84a7b93a66b0908bd7709
fig = plt.figure()


# In[85]:


#! ipython id=e706bc2402d645b58fdf4bdf78db9844
#! figure,id="baby_names_lesley",title="Proportion of male/female Lesley-like names over time"
table.plot(style={"M": "k-", "F": "k--"})


# In[86]:


#! ipython id=b35c37358f2744baa362accb22c7b050
import json
db = json.load(open("datasets/usda_food/database.json"))
len(db)


# In[87]:


#! ipython id=f04bd1e02f604ea4bd5d295855fd1289
db[0].keys()
db[0]["nutrients"][0]
nutrients = pd.DataFrame(db[0]["nutrients"])
nutrients.head(7)


# In[88]:


#! ipython id=292500dd7d93484c98341fb4c74fa514
info_keys = ["description", "group", "id", "manufacturer"]
info = pd.DataFrame(db, columns=info_keys)
info.head()
info.info()


# In[89]:


#! ipython id=7ed01e31319d4bb6af5ef058fb0c7afc
pd.value_counts(info["group"])[:10]


# In[90]:


#! ipython verbatim id=24c112fe1ee44d54bf747bae742861fc
nutrients = []

for rec in db:
    fnuts = pd.DataFrame(rec["nutrients"])
    fnuts["id"] = rec["id"]
    nutrients.append(fnuts)

nutrients = pd.concat(nutrients, ignore_index=True)


# In[91]:


#! ipython id=185aa7bd01684e93a298345bbca145e5
nutrients


# In[92]:


#! ipython id=1c9de0a1f9734472a7ba009a638fa50e
nutrients.duplicated().sum()  # number of duplicates
nutrients = nutrients.drop_duplicates()


# In[93]:


#! ipython id=fad42e28e4ef433db2b3ac9a1310758a
col_mapping = {"description" : "food",
               "group"       : "fgroup"}
info = info.rename(columns=col_mapping, copy=False)
info.info()
col_mapping = {"description" : "nutrient",
               "group" : "nutgroup"}
nutrients = nutrients.rename(columns=col_mapping, copy=False)
nutrients


# In[94]:


#! ipython id=3d1bb6fec99b4df09611f192fbd12462
ndata = pd.merge(nutrients, info, on="id")
ndata.info()
ndata.iloc[30000]


# In[95]:


#! ipython suppress id=02da7933628d4350b5c495755caed9a4
fig = plt.figure()


# In[96]:


#! ipython id=276ba38a83974b82bbc4849eb024bcb7
result = ndata.groupby(["nutrient", "fgroup"])["value"].quantile(0.5)
#! figure,id=fig_wrangle_zinc,width=4in,title="Median zinc values by food group"
result["Zinc, Zn"].sort_values().plot(kind="barh")


# In[97]:


#! ipython verbatim id=539c0a81efd34f95835032021f1e5ab6
by_nutrient = ndata.groupby(["nutgroup", "nutrient"])

def get_maximum(x):
    return x.loc[x.value.idxmax()]

max_foods = by_nutrient.apply(get_maximum)[["value", "food"]]

# make the food a little smaller
max_foods["food"] = max_foods["food"].str[:50]


# In[98]:


#! ipython id=bdb361d50f794410823645a33be3c365
max_foods.loc["Amino Acids"]["food"]


# In[99]:


#! ipython id=1d39de88d04247a692ef7c90ac52edbc
fec = pd.read_csv("datasets/fec/P00000001-ALL.csv", low_memory=False)
fec.info()


# In[100]:


#! ipython id=00bb57847d384a1c868ac10bdf94db5e
fec.iloc[123456]


# In[101]:


#! ipython id=747ed603f23f442abe038ca6108295b6
unique_cands = fec["cand_nm"].unique()
unique_cands
unique_cands[2]


# In[102]:


#! ipython verbatim id=f3fee9f744a642bb93414dcdcc4b3eff
parties = {"Bachmann, Michelle": "Republican",
           "Cain, Herman": "Republican",
           "Gingrich, Newt": "Republican",
           "Huntsman, Jon": "Republican",
           "Johnson, Gary Earl": "Republican",
           "McCotter, Thaddeus G": "Republican",
           "Obama, Barack": "Democrat",
           "Paul, Ron": "Republican",
           "Pawlenty, Timothy": "Republican",
           "Perry, Rick": "Republican",
           "Roemer, Charles E. 'Buddy' III": "Republican",
           "Romney, Mitt": "Republican",
           "Santorum, Rick": "Republican"}


# In[103]:


#! ipython id=b659a8f33d0d48adad4fca5927da7a48
fec["cand_nm"][123456:123461]
fec["cand_nm"][123456:123461].map(parties)
# Add it as a column
fec["party"] = fec["cand_nm"].map(parties)
fec["party"].value_counts()


# In[104]:


#! ipython id=8d2df181b8d0474d9b3509df8c27c743
(fec["contb_receipt_amt"] > 0).value_counts()


# In[105]:


#! ipython id=6fcec3476c974121b94a2147252276ca
fec = fec[fec["contb_receipt_amt"] > 0]


# In[106]:


#! ipython id=2f86e0fe1e334c6ab6597b4531d650a5
fec_mrbo = fec[fec["cand_nm"].isin(["Obama, Barack", "Romney, Mitt"])]


# In[107]:


#! ipython id=5f3ce51804ef4ef29e4b443d69fc6e6d
fec["contbr_occupation"].value_counts()[:10]


# In[108]:


#! ipython verbatim id=c6c82bbba04f445282d55d072300889e
occ_mapping = {
   "INFORMATION REQUESTED PER BEST EFFORTS" : "NOT PROVIDED",
   "INFORMATION REQUESTED" : "NOT PROVIDED",
   "INFORMATION REQUESTED (BEST EFFORTS)" : "NOT PROVIDED",
   "C.E.O.": "CEO"
}

def get_occ(x):
    # If no mapping provided, return x
    return occ_mapping.get(x, x)

fec["contbr_occupation"] = fec["contbr_occupation"].map(get_occ)


# In[109]:


#! ipython verbatim id=6e79332d5ea4476f89503731765aedc7
emp_mapping = {
   "INFORMATION REQUESTED PER BEST EFFORTS" : "NOT PROVIDED",
   "INFORMATION REQUESTED" : "NOT PROVIDED",
   "SELF" : "SELF-EMPLOYED",
   "SELF EMPLOYED" : "SELF-EMPLOYED",
}

def get_emp(x):
    # If no mapping provided, return x
    return emp_mapping.get(x, x)

fec["contbr_employer"] = fec["contbr_employer"].map(f)


# In[110]:


#! ipython id=52c5ed9cbb6042a2a31d92b66349d3df
by_occupation = fec.pivot_table("contb_receipt_amt",
                                index="contbr_occupation",
                                columns="party", aggfunc="sum")
over_2mm = by_occupation[by_occupation.sum(axis="columns") > 2000000]
over_2mm


# In[111]:


#! ipython suppress id=ce914115742b401dbe73c3a8574063d3
plt.figure()


# In[112]:


#! ipython id=33becfe73ad14bd584bf661685329273
#! figure,id=groupby_fec_occ_party,width=4.5in,title="Total donations by party for top occupations"
over_2mm.plot(kind="barh")


# In[113]:


#! ipython verbatim id=3dcbd1e24a774aa0a87f1459733b8ecf
def get_top_amounts(group, key, n=5):
    totals = group.groupby(key)["contb_receipt_amt"].sum()
    return totals.nlargest(n)


# In[114]:


#! ipython id=ed90d977b2e8476ab7a241886ac725ca
grouped = fec_mrbo.groupby("cand_nm")
grouped.apply(get_top_amounts, "contbr_occupation", n=7)
grouped.apply(get_top_amounts, "contbr_employer", n=10)


# In[115]:


#! ipython id=9430750304c744018ebfe44fa7825f15
bins = np.array([0, 1, 10, 100, 1000, 10000,
                 100_000, 1_000_000, 10_000_000])
labels = pd.cut(fec_mrbo["contb_receipt_amt"], bins)
labels


# In[116]:


#! ipython id=53f9ac1f658442ab824c5684308e083d
grouped = fec_mrbo.groupby(["cand_nm", labels])
grouped.size().unstack(level=0)


# In[117]:


#! ipython suppress id=6e9668803e4f4b6ea96904c4ae9d7d1c
plt.figure()


# In[118]:


#! ipython id=94fab00fdb7847b88c42b8faba89c0b2
bucket_sums = grouped["contb_receipt_amt"].sum().unstack(level=0)
normed_sums = bucket_sums.div(bucket_sums.sum(axis="columns"),
                              axis="index")
normed_sums
#! figure,id=fig_groupby_fec_bucket,width=4.5in,title="Percentage of total donations received by candidates for each donation size"
normed_sums[:-2].plot(kind="barh")


# In[119]:


#! ipython id=bcffb8dcc3b741fe87058d44f39324f1
grouped = fec_mrbo.groupby(["cand_nm", "contbr_st"])
totals = grouped["contb_receipt_amt"].sum().unstack(level=0).fillna(0)
totals = totals[totals.sum(axis="columns") > 100000]
totals.head(10)


# In[120]:


#! ipython id=eda82c69272d44b3a8eb23c0adb6ede4
percent = totals.div(totals.sum(axis="columns"), axis="index")
percent.head(10)


# In[121]:


#! ipython suppress id=c7894724cf8b4ce087cfee42b10068b3
get_ipython().run_line_magic('popd', '')

