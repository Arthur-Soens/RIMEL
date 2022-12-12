#!/usr/bin/env python
# coding: utf-8

# In[1]:


#! ipython suppress id=d9f0464c2a1244b080f520999badd4bd
get_ipython().run_line_magic('pushd', 'book-materials')
import numpy as np
import pandas as pd
np.random.seed(12345)
import matplotlib.pyplot as plt
plt.rc("figure", figsize=(10, 6))
pd.options.display.max_colwidth = 75
pd.options.display.max_columns = 20
np.set_printoptions(precision=4, suppress=True)


# In[2]:


#! ipython id=46a8a771c67942e2bd64bc6ebe2a5fba
get_ipython().system('cat examples/ex1.csv')


# In[3]:


#! ipython id=8f03eaac937f43b3a880a650d81342fb
df = pd.read_csv("examples/ex1.csv")
df


# In[4]:


#! ipython id=04eaafad5e034dd888ca309bfa6fd75c
get_ipython().system('cat examples/ex2.csv')


# In[5]:


#! ipython id=f14a82cd794d41c19a587da1a8697b4e
pd.read_csv("examples/ex2.csv", header=None)
pd.read_csv("examples/ex2.csv", names=["a", "b", "c", "d", "message"])


# In[6]:


#! ipython id=c94028be0e0141988bcf4471971f84d4
names = ["a", "b", "c", "d", "message"]
pd.read_csv("examples/ex2.csv", names=names, index_col="message")


# In[7]:


#! ipython id=3bed923466d542a9b069c9aa7f84ce98
get_ipython().system('cat examples/csv_mindex.csv')
parsed = pd.read_csv("examples/csv_mindex.csv",
                     index_col=["key1", "key2"])
parsed


# In[8]:


#! ipython id=364c583bcfd44701a8bbabd67e6f2d92
get_ipython().system('cat examples/ex3.txt')


# In[9]:


#! ipython id=dbcfb3db7cb046139c6fe3965dfbc337
result = pd.read_csv("examples/ex3.txt", sep="\s+")
result


# In[10]:


#! ipython id=1a93afa48ac84550b6c5cc40b6c035c1
get_ipython().system('cat examples/ex4.csv')
pd.read_csv("examples/ex4.csv", skiprows=[0, 2, 3])


# In[11]:


#! ipython id=f88ef9456a224c569566b57060aee8b5
get_ipython().system('cat examples/ex5.csv')
result = pd.read_csv("examples/ex5.csv")
result


# In[12]:


#! ipython id=53a72d55cdb043db96baea401c67df1a
pd.isna(result)


# In[13]:


#! ipython id=1cf88e6ecb114f85be97158d778d240c
result = pd.read_csv("examples/ex5.csv", na_values=["NULL"])
result


# In[14]:


#! ipython id=0796a9351d054dec84bcafca9365b6f7
result2 = pd.read_csv("examples/ex5.csv", keep_default_na=False)
result2
result2.isna()
result3 = pd.read_csv("examples/ex5.csv", keep_default_na=False,
                      na_values=["NA"])
result3
result3.isna()


# In[15]:


#! ipython id=4b0175de660345f0a4ab265d7c417716
sentinels = {"message": ["foo", "NA"], "something": ["two"]}
pd.read_csv("examples/ex5.csv", na_values=sentinels,
            keep_default_na=False)


# In[16]:


#! ipython id=22dffe254b5f43f481cd0f8d13c62216
pd.options.display.max_rows = 10


# In[17]:


#! ipython id=055bff4f7aed4f418805382c1dda1646
result = pd.read_csv("examples/ex6.csv")
result


# In[18]:


#! ipython id=41032a5b1c6b4b34bb984ec9b5a64431
pd.read_csv("examples/ex6.csv", nrows=5)


# In[19]:


#! ipython id=84f794c881cb4d4c88785be1472217d7
chunker = pd.read_csv("examples/ex6.csv", chunksize=1000)
type(chunker)


# In[20]:


#! ipython verbatim id=1b95293666ba4c09bbb82b813e3fdc66
chunker = pd.read_csv("examples/ex6.csv", chunksize=1000)

tot = pd.Series([], dtype='int64')
for piece in chunker:
    tot = tot.add(piece["key"].value_counts(), fill_value=0)

tot = tot.sort_values(ascending=False)


# In[21]:


#! ipython id=1938b6094d7449989e3dd9ed18068208
tot[:10]


# In[22]:


#! ipython id=19731c9bcc2c4dc5aa3a08ac4e701b94
data = pd.read_csv("examples/ex5.csv")
data


# In[23]:


#! ipython id=46552b7351b343dbaf3cfa06fef2cc42
data.to_csv("examples/out.csv")
get_ipython().system('cat examples/out.csv')


# In[24]:


#! ipython id=60850d24fb5343f9a00e37f7d71bd70a
import sys
data.to_csv(sys.stdout, sep="|")


# In[25]:


#! ipython id=a3ea57dfedf44b7eaf59ba88302ec310
data.to_csv(sys.stdout, na_rep="NULL")


# In[26]:


#! ipython id=03d6831eeaae4d03962d93bb0c3a39aa
data.to_csv(sys.stdout, index=False, header=False)


# In[27]:


#! ipython id=bc662cdc7cee43eca790a6257e99e4a6
data.to_csv(sys.stdout, index=False, columns=["a", "b", "c"])


# In[28]:


#! ipython id=11f6d5afa5a240a8b5676cc98ea2efa6
get_ipython().system('cat examples/ex7.csv')


# In[29]:


#! ipython id=d7e0bd861ab44c83a85d999bf4112661
import csv
f = open("examples/ex7.csv")
reader = csv.reader(f)


# In[30]:


#! ipython id=f903ffcb7fdf4bb5b429d1ac45e4c8ce
for line in reader:
    print(line)
f.close()


# In[31]:


#! ipython id=c8a91bd55f77454a9508ca7e7fecb152
with open("examples/ex7.csv") as f:
    lines = list(csv.reader(f))


# In[32]:


#! ipython id=4e1906eda527462eb82b91617936c9ec
header, values = lines[0], lines[1:]


# In[33]:


#! ipython id=a4b8e87dc8c04c29b4ab31c98f95b586
data_dict = {h: v for h, v in zip(header, zip(*values))}
data_dict


# In[34]:


#! ipython verbatim id=b31977579a1a4e46a5b0227cf8144ad7
obj = """
{"name": "Wes",
 "cities_lived": ["Akron", "Nashville", "New York", "San Francisco"],
 "pet": null,
 "siblings": [{"name": "Scott", "age": 34, "hobbies": ["guitars", "soccer"]},
              {"name": "Katie", "age": 42, "hobbies": ["diving", "art"]}]
}
"""


# In[35]:


#! ipython id=08101dece6d14f2089396736bd970f4f
import json
result = json.loads(obj)
result


# In[36]:


#! ipython id=2c27b3b11547444f94476c8b10b8236c
asjson = json.dumps(result)
asjson


# In[37]:


#! ipython id=7e072ae2cab94f68b2d7c6785537c95d
siblings = pd.DataFrame(result["siblings"], columns=["name", "age"])
siblings


# In[38]:


#! ipython id=33f67ed06da0416f85f6c68c2ce0e160
get_ipython().system('cat examples/example.json')


# In[39]:


#! ipython id=e30efe35105546679bd86495938792e6
data = pd.read_json("examples/example.json")
data


# In[40]:


#! ipython id=f53e717419144837bb8cb98e0ec81105
data.to_json(sys.stdout)
data.to_json(sys.stdout, orient="records")


# In[41]:


#! ipython id=4fbb17d66d5e4ae4b0c145b59b8c1871
tables = pd.read_html("examples/fdic_failed_bank_list.html")
len(tables)
failures = tables[0]
failures.head()


# In[42]:


#! ipython id=009f410066cc488cb888a606b3b7318c
close_timestamps = pd.to_datetime(failures["Closing Date"])
close_timestamps.dt.year.value_counts()


# In[43]:


#! ipython id=08f87224a0e9499091e5860426305ec2
from lxml import objectify

path = "datasets/mta_perf/Performance_MNR.xml"
with open(path) as f:
    parsed = objectify.parse(f)
root = parsed.getroot()


# In[44]:


#! ipython verbatim id=96872f4505bb496c898ee0561a1e7bc0
data = []

skip_fields = ["PARENT_SEQ", "INDICATOR_SEQ",
               "DESIRED_CHANGE", "DECIMAL_PLACES"]

for elt in root.INDICATOR:
    el_data = {}
    for child in elt.getchildren():
        if child.tag in skip_fields:
            continue
        el_data[child.tag] = child.pyval
    data.append(el_data)


# In[45]:


#! ipython id=ef60a86e732442368c22f0c89edfd858
perf = pd.DataFrame(data)
perf.head()


# In[46]:


#! ipython id=c6055246ec4747979a1cd62185887324
perf2 = pd.read_xml(path)
perf2.head()


# In[47]:


#! ipython id=6163973a4ffe49e284a3300ae795a1ba
frame = pd.read_csv("examples/ex1.csv")
frame
frame.to_pickle("examples/frame_pickle")


# In[48]:


#! ipython id=ab75f25904ad4cc79d3f2cdc2395a190
pd.read_pickle("examples/frame_pickle")


# In[49]:


#! ipython suppress id=f5d02c3f7e9d4c1f8c38fd81fd1f27e6
get_ipython().system('rm examples/frame_pickle')


# In[50]:


#! ipython id=11f4d6a438044669885fb816317165a0
fec = pd.read_parquet('datasets/fec/fec.parquet')


# In[51]:


#! ipython id=e82fff111381496685a03a0c4047dfb9
xlsx = pd.ExcelFile("examples/ex1.xlsx")


# In[52]:


#! ipython id=3e8f42e30f3244edb4389f49233bdc21
xlsx.sheet_names


# In[53]:


#! ipython id=de5cdf10d38f4eeda477817b70639cad
xlsx.parse(sheet_name="Sheet1")


# In[54]:


#! ipython id=dec5fb5ce91a45119a981ab35a34bd98
xlsx.parse(sheet_name="Sheet1", index_col=0)


# In[55]:


#! ipython id=fd02195f471844f7ba4446f1f7c29cc8
frame = pd.read_excel("examples/ex1.xlsx", sheet_name="Sheet1")
frame


# In[56]:


#! ipython id=bc24648985c74f5eb035f54284ed2b47
writer = pd.ExcelWriter("examples/ex2.xlsx")
frame.to_excel(writer, "Sheet1")
writer.save()


# In[57]:


#! ipython id=52e2638f13584a1187b9d3e21d68911c
frame.to_excel("examples/ex2.xlsx")


# In[58]:


#! ipython suppress id=b1ace24a923042aab0697fdf963d524f
get_ipython().system('rm examples/ex2.xlsx')


# In[59]:


#! ipython suppress id=f8d4df208f1f46b89c6d918e3309daca
get_ipython().system('rm -f examples/mydata.h5')


# In[60]:


#! ipython id=87dcbcfb24f141af8f33a32e079b8ce6
frame = pd.DataFrame({"a": np.random.standard_normal(100)})
store = pd.HDFStore("examples/mydata.h5")
store["obj1"] = frame
store["obj1_col"] = frame["a"]
store


# In[61]:


#! ipython id=614a81745b2f4d2d84834dfddd932111
store["obj1"]


# In[62]:


#! ipython id=d18d0acfeddb4733a4577289403d09b7
store.put("obj2", frame, format="table")
store.select("obj2", where=["index >= 10 and index <= 15"])
store.close()


# In[63]:


#! ipython id=a9dd67161ad7429e9d3b395bc2270fde
frame.to_hdf("examples/mydata.h5", "obj3", format="table")
pd.read_hdf("examples/mydata.h5", "obj3", where=["index < 5"])


# In[64]:


#! ipython id=f8d4df208f1f46b89c6d918e3309daca
import os
os.remove("examples/mydata.h5")


# In[65]:


#! ipython id=82d63f060df84920825dce3f2dcb1855
import requests
url = "https://api.github.com/repos/pandas-dev/pandas/issues"
resp = requests.get(url)
resp.raise_for_status()
resp


# In[66]:


#! ipython id=b12f3d5f473b4758abae8e22071e5b77
data = resp.json()
data[0]["title"]


# In[67]:


#! ipython id=33a7e76fa12f492d8af3004826fb90aa
issues = pd.DataFrame(data, columns=["number", "title",
                                     "labels", "state"])
issues


# In[68]:


#! ipython id=efdf76a63fe546b4a6db89426856a104
import sqlite3

query = """
CREATE TABLE test
(a VARCHAR(20), b VARCHAR(20),
 c REAL,        d INTEGER
);"""

con = sqlite3.connect("mydata.sqlite")
con.execute(query)
con.commit()


# In[69]:


#! ipython id=aac2e8e703d7481a95bb95308ddd0914
data = [("Atlanta", "Georgia", 1.25, 6),
        ("Tallahassee", "Florida", 2.6, 3),
        ("Sacramento", "California", 1.7, 5)]
stmt = "INSERT INTO test VALUES(?, ?, ?, ?)"

con.executemany(stmt, data)
con.commit()


# In[70]:


#! ipython id=15dec60cced441718d2a1af3274c94e7
cursor = con.execute("SELECT * FROM test")
rows = cursor.fetchall()
rows


# In[71]:


#! ipython id=af798082a50344d1998d995300f377d6
cursor.description
pd.DataFrame(rows, columns=[x[0] for x in cursor.description])


# In[72]:


#! ipython id=c35bc91fbffb4145b634cd3093af8452
import sqlalchemy as sqla
db = sqla.create_engine("sqlite:///mydata.sqlite")
pd.read_sql("SELECT * FROM test", db)


# In[73]:


#! ipython suppress id=95e53f5f4b104e3fae84f2507f4c39db
get_ipython().system('rm mydata.sqlite')


# In[74]:


#! ipython suppress id=fd013d6c63b44310b5712f05570e3255
get_ipython().run_line_magic('popd', '')

