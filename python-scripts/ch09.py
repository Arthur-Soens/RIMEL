#!/usr/bin/env python
# coding: utf-8

# In[1]:


#! ipython suppress id=61c3e93e7da045348bdcf837eeeb1ed6
get_ipython().run_line_magic('pushd', 'book-materials')
import numpy as np
import pandas as pd
PREVIOUS_MAX_ROWS = pd.options.display.max_rows
pd.options.display.max_rows = 20
pd.options.display.max_colwidth = 80
pd.options.display.max_columns = 20
np.random.seed(12345)
import matplotlib.pyplot as plt
import matplotlib
plt.rc("figure", figsize=(10, 6))
np.set_printoptions(precision=4, suppress=True)


# In[2]:


#! ipython id=3559a49dc675465ba81dcbb46a0f6834
import matplotlib.pyplot as plt


# In[3]:


#! ipython id=930563ea71d04149a31044f6d16f5a98
data = np.arange(10)
data
#! figure,id=mpl_first_plot,width=3in,title="Simple line plot"
plt.plot(data)


# In[4]:


#! ipython id=94a3d2f7a4974091ab17639b5e889437
fig = plt.figure()


# In[5]:


#! ipython id=5e4d7eeddf0b4385a608bb67c347443a
ax1 = fig.add_subplot(2, 2, 1)


# In[6]:


#! ipython id=33657c29acd849d398ff9b6930fcd466
ax2 = fig.add_subplot(2, 2, 2)
#! figure,id=mpl_empty_subplots,width=4in,title="An empty matplotlib figure with three subplots"
ax3 = fig.add_subplot(2, 2, 3)


# In[7]:


#! ipython id=247be769f2414efda8edd7fa4a49ff93
#! figure,id=mpl_subplots_one,width=4in,title="Data visualization after a single plot"
ax3.plot(np.random.standard_normal(50).cumsum(), color="black",
         linestyle="dashed")


# In[8]:


#! ipython id=08c76c0be47f439eabc8022db0d67c03
ax1.hist(np.random.standard_normal(100), bins=20, color="black", alpha=0.3);
#! figure,id=mpl_subplots_two,width=4in,title="Data visualization after additional plots"
ax2.scatter(np.arange(30), np.arange(30) + 3 * np.random.standard_normal(30));


# In[9]:


#! ipython suppress id=9ca014254f154d0b8b9f9fdae36a04e4
plt.close("all")


# In[10]:


#! ipython id=a38b9f8424f348ebb559a7945b609121
fig, axes = plt.subplots(2, 3)
axes


# In[11]:


#! ipython suppress id=56bf61e88f414933be21cfca7bb7a89e
fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
for i in range(2):
    for j in range(2):
        axes[i, j].hist(np.random.standard_normal(500), bins=50,
                        color="black", alpha=0.5)
#! figure,id=mpl_subplots_adjust,width=4in,title="Data visualization with no inter-subplot spacing"
fig.subplots_adjust(wspace=0, hspace=0)


# In[12]:


#! ipython suppress id=5dc52ff0a09b4593ba638cba1be94aad
fig = plt.figure()


# In[13]:


#! ipython id=7c2ae213e9214fe0a8780d91a64ed273
ax = fig.add_subplot()
#! figure,id=mpl_marker_ex,width=4in,title="Line plot with markers"
ax.plot(np.random.standard_normal(30).cumsum(), color="black",
        linestyle="dashed", marker="o");


# In[14]:


#! ipython suppress id=5e0afabc7ebd43d383a0c42d2188d9af
plt.close("all")


# In[15]:


#! ipython id=600cec64d3e842749a84b16b8a549145
fig = plt.figure()
ax = fig.add_subplot()
data = np.random.standard_normal(30).cumsum()
ax.plot(data, color="black", linestyle="dashed", label="Default");
ax.plot(data, color="black", linestyle="dashed",
        drawstyle="steps-post", label="steps-post");
#! figure,id=mpl_drawstyle_ex,width=4in,title="Line plot with different drawstyle options"
ax.legend()


# In[16]:


#! ipython id=bac0ed2c52a04e62aad3cab21801a44c
fig, ax = plt.subplots()
#! figure,id=vis_ticks_one,width=3.5in,title="Simple plot for illustrating xticks (with default labels)"
ax.plot(np.random.standard_normal(1000).cumsum());


# In[17]:


#! ipython id=08aa54cdf68c46fa8137f4c30189fb7d
ticks = ax.set_xticks([0, 250, 500, 750, 1000])
labels = ax.set_xticklabels(["one", "two", "three", "four", "five"],
                            rotation=30, fontsize=8)


# In[18]:


#! ipython id=933a74fa372e40349697f0c32e50f259
ax.set_xlabel("Stages")
#! figure,id=vis_ticks_two,width=3.5in,title="Simple plot for illustrating custom xticks"
ax.set_title("My first matplotlib plot")


# In[19]:


#! ipython id=b89655b6b64e49adb709a8dcaa4b1fec
fig, ax = plt.subplots()
ax.plot(np.random.randn(1000).cumsum(), color="black", label="one");
ax.plot(np.random.randn(1000).cumsum(), color="black", linestyle="dashed",
        label="two");
ax.plot(np.random.randn(1000).cumsum(), color="black", linestyle="dotted",
        label="three");


# In[20]:


#! ipython id=930f945557234991a575666e39e2690a
#! figure,id=vis_legend_ex,width=4in,title="Simple plot with three lines and legend"
ax.legend()


# In[21]:


#! ipython verbatim id=00b59126f3364fdf900df06d486ed23d
from datetime import datetime

fig, ax = plt.subplots()

data = pd.read_csv("examples/spx.csv", index_col=0, parse_dates=True)
spx = data["SPX"]

spx.plot(ax=ax, color="black")

crisis_data = [
    (datetime(2007, 10, 11), "Peak of bull market"),
    (datetime(2008, 3, 12), "Bear Stearns Fails"),
    (datetime(2008, 9, 15), "Lehman Bankruptcy")
]

for date, label in crisis_data:
    ax.annotate(label, xy=(date, spx.asof(date) + 75),
                xytext=(date, spx.asof(date) + 225),
                arrowprops=dict(facecolor="black", headwidth=4, width=2,
                                headlength=4),
                horizontalalignment="left", verticalalignment="top")

# Zoom in on 2007-2010
ax.set_xlim(["1/1/2007", "1/1/2011"])
ax.set_ylim([600, 1800])

ax.set_title("Important dates in the 2008-2009 financial crisis")


# In[22]:


#! ipython suppress id=a30e29e9107c43e2a03b2330810368e4
#! figure,id=vis_crisis_dates,width=4.5in,title="Important dates in the 2008–2009 financial crisis"
ax.set_title("Important dates in the 2008–2009 financial crisis")


# In[23]:


#! ipython suppress id=038b73befd9a4dbab9a08fee140e53a3
fig, ax = plt.subplots(figsize=(12, 6))
rect = plt.Rectangle((0.2, 0.75), 0.4, 0.15, color="black", alpha=0.3)
circ = plt.Circle((0.7, 0.2), 0.15, color="blue", alpha=0.3)
pgon = plt.Polygon([[0.15, 0.15], [0.35, 0.4], [0.2, 0.6]],
                   color="green", alpha=0.5)
ax.add_patch(rect)
ax.add_patch(circ)
#! figure,id=vis_patch_ex,width=4in,title="Data visualization composed from three different patches"
ax.add_patch(pgon)


# In[24]:


#! ipython suppress id=d6b678466682413088fa5903f9197696
plt.close("all")


# In[25]:


#! ipython id=21edcf8da2154b7db12f7083ab4e9777
s = pd.Series(np.random.standard_normal(10).cumsum(), index=np.arange(0, 100, 10))
#! figure,id=vis_series_plot_1,width=4in,title="Simple Series plot"
s.plot()


# In[26]:


#! ipython id=1ef68f01e6f248259804975945e44951
df = pd.DataFrame(np.random.standard_normal((10, 4)).cumsum(0),
                  columns=["A", "B", "C", "D"],
                  index=np.arange(0, 100, 10))
plt.style.use('grayscale')
#! figure,id=vis_frame_plot_1,width=4in,title="Simple DataFrame plot"
df.plot()


# In[27]:


#! ipython id=55da9f184455476f8ce79e4eaf79742c
fig, axes = plt.subplots(2, 1)
data = pd.Series(np.random.uniform(size=16), index=list("abcdefghijklmnop"))
data.plot.bar(ax=axes[0], color="black", alpha=0.7)
#! figure,id=vis_bar_plot_ex,width=4.5in,title="Horizonal and vertical bar plot"
data.plot.barh(ax=axes[1], color="black", alpha=0.7)


# In[28]:


#! ipython suppress id=bbbe2554feb1460caf331d0a9c1a158a
np.random.seed(12348)


# In[29]:


#! ipython id=da2de3b4af64451ca5b47b6687e40c44
df = pd.DataFrame(np.random.uniform(size=(6, 4)),
                  index=["one", "two", "three", "four", "five", "six"],
                  columns=pd.Index(["A", "B", "C", "D"], name="Genus"))
df
#! figure,id=vis_frame_barplot,width=4in,title="DataFrame bar plot"
df.plot.bar()


# In[30]:


#! ipython suppress id=583fcb73d46a44ee9c25f4b652d8767a
plt.figure()


# In[31]:


#! ipython id=ead442e8fee747aea00117cb77de970c
#! figure,id=vis_frame_barplot_stacked,width=4in,title="DataFrame stacked bar plot"
df.plot.barh(stacked=True, alpha=0.5)


# In[32]:


#! ipython suppress id=70abd941e8d843678a1c1cf5010eaa27
plt.close("all")


# In[33]:


#! ipython id=1bb875213a92468788f61c4e8012932c
tips = pd.read_csv("examples/tips.csv")
tips.head()
party_counts = pd.crosstab(tips["day"], tips["size"])
party_counts = party_counts.reindex(index=["Thur", "Fri", "Sat", "Sun"])
party_counts


# In[34]:


#! ipython id=4966d9560d0740db829531eb2a8f6c7a
party_counts = party_counts.loc[:, 2:5]


# In[35]:


#! ipython id=b88726f319554e9cb32498dce98e66ca
# Normalize to sum to 1
party_pcts = party_counts.div(party_counts.sum(axis="columns"),
                              axis="index")
party_pcts
#! figure,id=vis_tips_barplot,width=4in,title="Fraction of parties by size within each day"
party_pcts.plot.bar(stacked=True)


# In[36]:


#! ipython suppress id=2e6c41575fc34b8ca7963b2c24167be2
plt.close("all")


# In[37]:


#! ipython id=97f4026c26cd42faa7d03d3e0dcc75fa
import seaborn as sns

tips["tip_pct"] = tips["tip"] / (tips["total_bill"] - tips["tip"])
tips.head()
#! figure,id=vis_tip_pct_seaborn,width=4in,title="Tipping percentage by day with error bars"
sns.barplot(x="tip_pct", y="day", data=tips, orient="h")


# In[38]:


#! ipython suppress id=05ae3cd4fb7048c5ba0d9ed6390c1f03
plt.close("all")


# In[39]:


#! ipython id=fd8ada60aab448848ded49ef8a81081c
#! figure,id=vis_tip_pct_sns_grouped,width=4in,title="Tipping percentage by day and time"
sns.barplot(x="tip_pct", y="day", hue="time", data=tips, orient="h")


# In[40]:


#! ipython suppress id=d849ee158e424d2ca9e9741fdef08a7a
plt.close("all")


# In[41]:


#! ipython id=07a3594954ff4d2fac3caa7927f64273
sns.set_style("whitegrid")


# In[42]:


#! ipython suppress id=ce4d75faf525411bbb4e8edbe016ff27
plt.figure()


# In[43]:


#! ipython id=1589674e6dda4960800de8803aa5c3ab
#! figure,id=vis_hist_ex,width=4in,title="Histogram of tip percentages"
tips["tip_pct"].plot.hist(bins=50)


# In[44]:


#! ipython suppress id=5db1481f3ef441688c9bd5512da6fb9f
plt.figure()


# In[45]:


#! ipython id=0173a110796b44ec85d62f05b9d703be
#! figure,id=vis_kde_ex,width=4in,title="Density plot of tip percentages"
tips["tip_pct"].plot.density()


# In[46]:


#! ipython suppress id=0bee68ae5d834d4f8ad68ff964bd39f6
plt.figure()


# In[47]:


#! ipython id=95802c949eee44d7a9f7e5760e85ddde
comp1 = np.random.standard_normal(200)
comp2 = 10 + 2 * np.random.standard_normal(200)
values = pd.Series(np.concatenate([comp1, comp2]))

#! figure,id=vis_series_kde,width=4in,title="Normalized histogram of normal mixture"
sns.histplot(values, bins=100, color="black")


# In[48]:


#! ipython id=b99bf04e3b824743a391b9590e4b201b
macro = pd.read_csv("examples/macrodata.csv")
data = macro[["cpi", "m1", "tbilrate", "unemp"]]
trans_data = np.log(data).diff().dropna()
trans_data.tail()


# In[49]:


#! ipython suppress id=0b0300f35b1a49d7935c15315a863735
plt.figure()


# In[50]:


#! ipython id=715be198942a441688a1bd8765f88cb3
ax = sns.regplot(x="m1", y="unemp", data=trans_data)
#! figure,id=scatter_plot_ex,width=3in,title="A seaborn regression/scatter plot"
ax.title("Changes in log(m1) versus log(unemp)")


# In[51]:


#! ipython id=c847f9d3a28f4c0f98518f94eee846ce
#! figure,id=scatter_matrix_ex,width=4in,title="Pair plot matrix of statsmodels macro data"
sns.pairplot(trans_data, diag_kind="kde", plot_kws={"alpha": 0.2})


# In[52]:


#! ipython id=e7ac5cdddb9a40afbc2d3d1fcc07f0e9
#! figure,id=vis_tip_pct_sns_factorplot,width=4in,title="Tipping percentage by day/time/smoker"
sns.catplot(x="day", y="tip_pct", hue="time", col="smoker",
            kind="bar", data=tips[tips.tip_pct < 1])


# In[53]:


#! ipython id=1039f6945d4242ccb653e1fa645e2c81
#! figure,id=vis_tip_pct_sns_factorplot2,width=4in,title="Tipping percentage by day split by time/smoker"
sns.catplot(x="day", y="tip_pct", row="time",
            col="smoker",
            kind="bar", data=tips[tips.tip_pct < 1])


# In[54]:


#! ipython id=511751cc57d844a1b39052cca90803f8
#! figure,id=vis_tip_pct_sns_factor_box,width=4in,title="Box plot of tipping percentage by day"
sns.catplot(x="tip_pct", y="day", kind="box",
            data=tips[tips.tip_pct < 0.5])


# In[55]:


#! ipython suppress id=b7f5b31c0dff42ec8441ca4b21003edb
get_ipython().run_line_magic('popd', '')


# In[56]:


#! ipython suppress id=19b6468a77c24957a335c58b413a9112
pd.options.display.max_rows = PREVIOUS_MAX_ROWS

