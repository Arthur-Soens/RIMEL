#!/usr/bin/env python
# coding: utf-8

# In[1]:


#! ipython suppress id=e926030638df4e2f922f33c9c27afc51
get_ipython().run_line_magic('pushd', 'book-materials')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rc('figure', figsize=(10, 6))
PREVIOUS_MAX_ROWS = pd.options.display.max_rows
pd.options.display.max_columns = 20
pd.options.display.max_rows = 20
pd.options.display.max_colwidth = 80
np.set_printoptions(precision=4, suppress=True)


# In[2]:


#! ipython id=d620c53910a240e69ece77ec9bd156f6
rng = np.random.default_rng(seed=12345)


# In[3]:


#! ipython id=b4eb8d5574a54518973f910d623bf5e8
np.ones((10, 5)).shape


# In[4]:


#! ipython id=7bf22ceac5c94f9bba173f578a933d8a
np.ones((3, 4, 5), dtype=np.float64).strides


# In[5]:


#! ipython id=7d56951cbc81495790806d99f88db611
ints = np.ones(10, dtype=np.uint16)
floats = np.ones(10, dtype=np.float32)
np.issubdtype(ints.dtype, np.integer)
np.issubdtype(floats.dtype, np.floating)


# In[6]:


#! ipython id=7c790b51cfef43789583e81d44f667eb
np.float64.mro()


# In[7]:


#! ipython id=802b4959858f42f68fdd8223778c6461
np.issubdtype(ints.dtype, np.number)


# In[8]:


#! ipython id=ce8ef0b9be0541258afa90e9a306b2d5
arr = np.arange(8)
arr
arr.reshape((4, 2))


# In[9]:


#! ipython id=7f3a7b2404af44ecb287104171672800
arr.reshape((4, 2)).reshape((2, 4))


# In[10]:


#! ipython id=a22eb31894eb44e9af770cafd2dc82f8
arr = np.arange(15)
arr.reshape((5, -1))


# In[11]:


#! ipython id=3c3a3b489e7c4e77b4405c114b61cebc
other_arr = np.ones((3, 5))
other_arr.shape
arr.reshape(other_arr.shape)


# In[12]:


#! ipython id=e9ffffb7cf6d4703bba796bee68d0a55
arr = np.arange(15).reshape((5, 3))
arr
arr.ravel()


# In[13]:


#! ipython id=07195f02b51e408cb21a126d3001d130
arr.flatten()


# In[14]:


#! ipython id=bc83d3159fff4c5da487f3981f727c07
arr = np.arange(12).reshape((3, 4))
arr
arr.ravel()
arr.ravel('F')


# In[15]:


#! ipython id=c0bea878f4d74ef9b5bce94b88b90b3b
arr1 = np.array([[1, 2, 3], [4, 5, 6]])
arr2 = np.array([[7, 8, 9], [10, 11, 12]])
np.concatenate([arr1, arr2], axis=0)
np.concatenate([arr1, arr2], axis=1)


# In[16]:


#! ipython id=33efa139bb234d10817538b49b41fc02
np.vstack((arr1, arr2))
np.hstack((arr1, arr2))


# In[17]:


#! ipython id=6e069f660a754dab80caf4ea98d8161f
arr = rng.standard_normal((5, 2))
arr
first, second, third = np.split(arr, [1, 3])
first
second
third


# In[18]:


#! ipython id=988edd872ce94799b21ff68a2aedc0cb
arr = np.arange(6)
arr1 = arr.reshape((3, 2))
arr2 = rng.standard_normal((3, 2))
np.r_[arr1, arr2]
np.c_[np.r_[arr1, arr2], arr]


# In[19]:


#! ipython id=9756c6c66e71442cbf99da0bc5723058
np.c_[1:6, -10:-5]


# In[20]:


#! ipython id=cebbe60be04246d6b5c099fb47cfe223
arr = np.arange(3)
arr
arr.repeat(3)


# In[21]:


#! ipython id=5ed5884236d343a4b896eaefe3282631
arr.repeat([2, 3, 4])


# In[22]:


#! ipython id=1ec3cee8c380412692d75b15c8a95f04
arr = rng.standard_normal((2, 2))
arr
arr.repeat(2, axis=0)


# In[23]:


#! ipython id=edf606b3fc364243bfb27eae78a59308
arr.repeat([2, 3], axis=0)
arr.repeat([2, 3], axis=1)


# In[24]:


#! ipython id=e8829e7ccec642d7980149682a573d36
arr
np.tile(arr, 2)


# In[25]:


#! ipython id=7edcdb77a9834138a2e1b2f21ec85248
arr
np.tile(arr, (2, 1))
np.tile(arr, (3, 2))


# In[26]:


#! ipython id=b69d0b1b35f94e5f93d50b9e91a51d82
arr = np.arange(10) * 100
inds = [7, 1, 2, 6]
arr[inds]


# In[27]:


#! ipython id=3078f81f1f1245438754131cfda83a71
arr.take(inds)
arr.put(inds, 42)
arr
arr.put(inds, [40, 41, 42, 43])
arr


# In[28]:


#! ipython id=ec4ebd6e59a842789ade7673186355fa
inds = [2, 0, 2, 1]
arr = rng.standard_normal((2, 4))
arr
arr.take(inds, axis=1)


# In[29]:


#! ipython id=c490df60ace6471fa7a275199f6648f4
arr = np.arange(5)
arr
arr * 4


# In[30]:


#! ipython id=b415c720eee543a183697f250dfc43eb
arr = rng.standard_normal((4, 3))
arr.mean(0)
demeaned = arr - arr.mean(0)
demeaned
demeaned.mean(0)


# In[31]:


#! ipython id=24389415f5f24b43ab770324a9c9ff5f
arr
row_means = arr.mean(1)
row_means.shape
row_means.reshape((4, 1))
demeaned = arr - row_means.reshape((4, 1))
demeaned.mean(1)


# In[32]:


#! ipython allow_exceptions id=2b090f7325404c93acd6bf0357012ef2
arr - arr.mean(1)


# In[33]:


#! ipython id=3274f631f8944b27849e55022d15e856
arr - arr.mean(1).reshape((4, 1))


# In[34]:


#! ipython id=22c9e0de0f1a42d1acbfcc8a63d6a4d6
arr = np.zeros((4, 4))
arr_3d = arr[:, np.newaxis, :]
arr_3d.shape
arr_1d = rng.standard_normal(3)
arr_1d[:, np.newaxis]
arr_1d[np.newaxis, :]


# In[35]:


#! ipython id=bc26b50944e54b2cba705986b6a9079f
arr = rng.standard_normal((3, 4, 5))
depth_means = arr.mean(2)
depth_means
depth_means.shape
demeaned = arr - depth_means[:, :, np.newaxis]
demeaned.mean(2)


# In[36]:


#! ipython id=29bae95d1b0842c0b457dda156c6bfc0
arr = np.zeros((4, 3))
arr[:] = 5
arr


# In[37]:


#! ipython id=ac69aca59c94463d9b6df157d9b521ac
col = np.array([1.28, -0.42, 0.44, 1.6])
arr[:] = col[:, np.newaxis]
arr
arr[:2] = [[-1.37], [0.509]]
arr


# In[38]:


#! ipython id=14ab862de0b54de6a88bcee07cc87d2b
arr = np.arange(10)
np.add.reduce(arr)
arr.sum()


# In[39]:


#! ipython id=6a6774bb315f4e68920057ba3e4313b5
my_rng = np.random.default_rng(12346)  # for reproducibility
arr = my_rng.standard_normal((5, 5))
arr
arr[::2].sort(1) # sort a few rows
arr[:, :-1] < arr[:, 1:]
np.logical_and.reduce(arr[:, :-1] < arr[:, 1:], axis=1)


# In[40]:


#! ipython id=492174bca93b477b813a9387627386bf
arr = np.arange(15).reshape((3, 5))
np.add.accumulate(arr, axis=1)


# In[41]:


#! ipython id=a611264ed18b446aa35e1f93fd9c3852
arr = np.arange(3).repeat([1, 2, 2])
arr
np.multiply.outer(arr, np.arange(5))


# In[42]:


#! ipython id=83b253bbc5b8420b9caadd08be216a15
x, y = rng.standard_normal((3, 4)), rng.standard_normal(5)
result = np.subtract.outer(x, y)
result.shape


# In[43]:


#! ipython id=9d967075734a4f12a98072f21ccc2f50
arr = np.arange(10)
np.add.reduceat(arr, [0, 5, 8])


# In[44]:


#! ipython id=2f1bc012f36d4b79b680ae14feb2f5f6
arr = np.multiply.outer(np.arange(4), np.arange(5))
arr
np.add.reduceat(arr, [0, 2, 4], axis=1)


# In[45]:


#! ipython id=4be00eeb2796461996ebf7fb17d7d9ef
def add_elements(x, y):
    return x + y
add_them = np.frompyfunc(add_elements, 2, 1)
add_them(np.arange(8), np.arange(8))


# In[46]:


#! ipython id=1141b1d6ac6c46509bb1ae3ced6bab2d
add_them = np.vectorize(add_elements, otypes=[np.float64])
add_them(np.arange(8), np.arange(8))


# In[47]:


#! ipython id=837c3af6451b473db934f3feda79f443
arr = rng.standard_normal(10000)
get_ipython().run_line_magic('timeit', 'add_them(arr, arr)')
get_ipython().run_line_magic('timeit', 'np.add(arr, arr)')


# In[48]:


#! ipython id=c0a9418af68f422db17fc103bba4be52
dtype = [('x', np.float64), ('y', np.int32)]
sarr = np.array([(1.5, 6), (np.pi, -2)], dtype=dtype)
sarr


# In[49]:


#! ipython id=38a27fd8a31f42eb81fd1f2f7c75690c
sarr[0]
sarr[0]['y']


# In[50]:


#! ipython id=4cca2ce62d114eb68b1b45f7f254c533
sarr['x']


# In[51]:


#! ipython id=e9302f10530049b6a23c7af55f170a39
dtype = [('x', np.int64, 3), ('y', np.int32)]
arr = np.zeros(4, dtype=dtype)
arr


# In[52]:


#! ipython id=514ec71c8e9349dead3b18e382625f21
arr[0]['x']


# In[53]:


#! ipython id=4daaa32c58614b8d98c4c8cedba3cea5
arr['x']


# In[54]:


#! ipython id=7bd40164ecdd459c8adc144cb17b9467
dtype = [('x', [('a', 'f8'), ('b', 'f4')]), ('y', np.int32)]
data = np.array([((1, 2), 5), ((3, 4), 6)], dtype=dtype)
data['x']
data['y']
data['x']['a']


# In[55]:


#! ipython id=73efe3c6188e410da27ab48c9a7f53c3
arr = rng.standard_normal(6)
arr.sort()
arr


# In[56]:


#! ipython id=69b9121e120542c980b2881fd318539a
arr = rng.standard_normal((3, 5))
arr
arr[:, 0].sort()  # Sort first column values in place
arr


# In[57]:


#! ipython id=bbf95e7e69c84b5d9217178b033279ff
arr = rng.standard_normal(5)
arr
np.sort(arr)
arr


# In[58]:


#! ipython id=60280db86cd24fdeb3b58b93d5382366
arr = rng.standard_normal((3, 5))
arr
arr.sort(axis=1)
arr


# In[59]:


#! ipython id=4b269d24749c43d09afe58fad81e2242
arr[:, ::-1]


# In[60]:


#! ipython id=06130b05850e48d68bc1ede1f371fea1
values = np.array([5, 0, 1, 3, 2])
indexer = values.argsort()
indexer
values[indexer]


# In[61]:


#! ipython id=e12e6b522b5b4519ad20abcdd428099d
arr = rng.standard_normal((3, 5))
arr[0] = values
arr
arr[:, arr[0].argsort()]


# In[62]:


#! ipython id=74b0473e78574cd39056cd680c9dd72d
first_name = np.array(['Bob', 'Jane', 'Steve', 'Bill', 'Barbara'])
last_name = np.array(['Jones', 'Arnold', 'Arnold', 'Jones', 'Walters'])
sorter = np.lexsort((first_name, last_name))
sorter
list(zip(last_name[sorter], first_name[sorter]))


# In[63]:


#! ipython id=d84e962e207145bba4e631af61a21c58
values = np.array(['2:first', '2:second', '1:first', '1:second',
                   '1:third'])
key = np.array([2, 2, 1, 1, 1])
indexer = key.argsort(kind='mergesort')
indexer
values.take(indexer)


# In[64]:


#! ipython id=2bedabe6402f4a6fbb99a44b2814267c
rng = np.random.default_rng(12345)
arr = rng.standard_normal(20)
arr
np.partition(arr, 3)


# In[65]:


#! ipython id=c5cc9ec52b18419d910da3cbb6a65e37
indices = np.argpartition(arr, 3)
indices
arr.take(indices)


# In[66]:


#! ipython id=6792cdc9a8b941598bfeae97a3c8fe53
arr = np.array([0, 1, 7, 12, 15])
arr.searchsorted(9)


# In[67]:


#! ipython id=0c0cc89abad747b38fd20a51eb8ecbc9
arr.searchsorted([0, 8, 11, 16])


# In[68]:


#! ipython id=1d38f21b28f84a539dab05393eef74d6
arr = np.array([0, 0, 0, 1, 1, 1, 1])
arr.searchsorted([0, 1])
arr.searchsorted([0, 1], side='right')


# In[69]:


#! ipython id=0916802173d441698ea70fa6267d1679
data = np.floor(rng.uniform(0, 10000, size=50))
bins = np.array([0, 100, 1000, 5000, 10000])
data


# In[70]:


#! ipython id=eff843c29e2b4f25bc876cde836ea118
labels = bins.searchsorted(data)
labels


# In[71]:


#! ipython id=fce3b76453ea4db391c4cd582a341b05
pd.Series(data).groupby(labels).mean()


# In[72]:


#! ipython verbatim id=103dc5eb76d24212aa0b6132d0340ab3
import numpy as np

def mean_distance(x, y):
    nx = len(x)
    result = 0.0
    count = 0
    for i in range(nx):
        result += x[i] - y[i]
        count += 1
    return result / count


# In[73]:


#! ipython id=93268b6cec1a4b6683f36f96b26794cc
mmap = np.memmap('mymmap', dtype='float64', mode='w+',
                 shape=(10000, 10000))
mmap


# In[74]:


#! ipython id=c5a4b74fb3fd4d0d8171d9fbeb646e8f
section = mmap[:5]


# In[75]:


#! ipython id=9cdaeb21f7e2482e85247400525f7530
section[:] = rng.standard_normal((5, 10000))
mmap.flush()
mmap
del mmap


# In[76]:


#! ipython id=004c3e0df5d54c7eb5e9666df4735017
mmap = np.memmap('mymmap', dtype='float64', shape=(10000, 10000))
mmap


# In[77]:


#! ipython id=031548779f4a45b8b9ab9b0d7f47485d
get_ipython().run_line_magic('xdel', 'mmap')
get_ipython().system('rm mymmap')


# In[78]:


#! ipython id=05f2436e37a144048f7a6b2f923e712e
arr_c = np.ones((100, 10000), order='C')
arr_f = np.ones((100, 10000), order='F')
arr_c.flags
arr_f.flags
arr_f.flags.f_contiguous


# In[79]:


#! ipython id=b886f529c4c74829ad119f83e5ccdeb9
get_ipython().run_line_magic('timeit', 'arr_c.sum(1)')
get_ipython().run_line_magic('timeit', 'arr_f.sum(1)')


# In[80]:


#! ipython id=6712242905fd4c97938998ffa4cc7b9b
arr_f.copy('C').flags


# In[81]:


#! ipython id=06dd1d79515047beb5a8a4895ce17aed
arr_c[:50].flags.contiguous
arr_c[:, :50].flags


# In[82]:


#! ipython suppress id=2ccd538ad8c64c5198411a19b44fabf4
get_ipython().run_line_magic('xdel', 'arr_c')
get_ipython().run_line_magic('xdel', 'arr_f')


# In[83]:


#! ipython suppress id=0affe014fbd34cde9a3769dd8c94fdc6
get_ipython().run_line_magic('popd', '')


# In[84]:


#! ipython suppress id=eb88d21e8e59441e884dda4884f3d0f8
pd.options.display.max_rows = PREVIOUS_MAX_ROWS

