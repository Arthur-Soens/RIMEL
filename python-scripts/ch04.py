#!/usr/bin/env python
# coding: utf-8

# In[1]:


#! ipython suppress id=9bd5de516dc841ddbd0dc1fbe6976f7f
get_ipython().run_line_magic('pushd', 'book-materials')
import numpy as np
np.random.seed(12345)
import matplotlib.pyplot as plt
plt.rc("figure", figsize=(10, 6))
np.set_printoptions(precision=4, suppress=True)


# In[2]:


#! ipython id=117c31586eff40c4aa555a73f2ae76fd
import numpy as np

my_arr = np.arange(1_000_000)
my_list = list(range(1_000_000))


# In[3]:


#! ipython id=ee0ca0270fa84c3b8ed5addd00c8d501
get_ipython().run_line_magic('timeit', 'my_arr2 = my_arr * 2')
get_ipython().run_line_magic('timeit', 'my_list2 = [x * 2 for x in my_list]')


# In[4]:


#! ipython id=16ba9a58a2ff46f2906b1921533388e1
import numpy as np
data = np.array([[1.5, -0.1, 3], [0, -3, 6.5]])
data


# In[5]:


#! ipython id=11bdd03aeaa440d1bd4a4f3f020379ec
data * 10
data + data


# In[6]:


#! ipython id=390330db0c71459fae6dcc2305e20784
data.shape
data.dtype


# In[7]:


#! ipython id=87f4fd1af05d40d3842cc370754b5bbe
data1 = [6, 7.5, 8, 0, 1]
arr1 = np.array(data1)
arr1


# In[8]:


#! ipython id=01cba83a62334079a59f0f4315ef8c1a
data2 = [[1, 2, 3, 4], [5, 6, 7, 8]]
arr2 = np.array(data2)
arr2


# In[9]:


#! ipython id=b6e274af36874ec38a3e1c6ada7db9fd
arr2.ndim
arr2.shape


# In[10]:


#! ipython id=42bc19d28e944b8e895be4a5f51087d2
arr1.dtype
arr2.dtype


# In[11]:


#! ipython id=931071fc61f9465d822c0b7abebbed05
np.zeros(10)
np.zeros((3, 6))
np.empty((2, 3, 2))


# In[12]:


#! ipython id=91356647856e47cfb1da6bd6c0f748c9
np.arange(15)


# In[13]:


#! ipython id=1b695d6668ff4d4ab25124ea46b94468
arr1 = np.array([1, 2, 3], dtype=np.float64)
arr2 = np.array([1, 2, 3], dtype=np.int32)
arr1.dtype
arr2.dtype


# In[14]:


#! ipython id=b622f969a13b40738619b11428843191
arr = np.array([1, 2, 3, 4, 5])
arr.dtype
float_arr = arr.astype(np.float64)
float_arr
float_arr.dtype


# In[15]:


#! ipython id=82b22a75106b46989228b7f336e4c68c
arr = np.array([3.7, -1.2, -2.6, 0.5, 12.9, 10.1])
arr
arr.astype(np.int32)


# In[16]:


#! ipython id=2f5917c56c18454282912d10eeb27c86
numeric_strings = np.array(["1.25", "-9.6", "42"], dtype=np.string_)
numeric_strings.astype(float)


# In[17]:


#! ipython id=4fff3f93a97b47a79a9ea8a382a7689c
int_array = np.arange(10)
calibers = np.array([.22, .270, .357, .380, .44, .50], dtype=np.float64)
int_array.astype(calibers.dtype)


# In[18]:


#! ipython id=1068940f540c4bccadaa9aae596861da
zeros_uint32 = np.zeros(8, dtype="u4")
zeros_uint32


# In[19]:


#! ipython id=eb7dcad5fb7749f09a16ef1f506d70be
arr = np.array([[1., 2., 3.], [4., 5., 6.]])
arr
arr * arr
arr - arr


# In[20]:


#! ipython id=1a6d83d7fb54451b916a8d878742a9dd
1 / arr
arr ** 2


# In[21]:


#! ipython id=6805726b60054c508f13763f799314a9
arr2 = np.array([[0., 4., 1.], [7., 2., 12.]])
arr2
arr2 > arr


# In[22]:


#! ipython id=202d7c3fac3442f8ad50555b1b4ee757
arr = np.arange(10)
arr
arr[5]
arr[5:8]
arr[5:8] = 12
arr


# In[23]:


#! ipython id=3fb5aac558f74791a8d0e3497f71be87
arr_slice = arr[5:8]
arr_slice


# In[24]:


#! ipython id=b94ea5960a7546c092676b8f6031a2de
arr_slice[1] = 12345
arr


# In[25]:


#! ipython id=0a9d3c1be0c24fb6a35a0fc10d99f8e0
arr_slice[:] = 64
arr


# In[26]:


#! ipython id=aa71cda85fc643efa5e4006e7ee7cab6
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
arr2d[2]


# In[27]:


#! ipython id=9eaf9635ac7b46f69113cbbb62d3daa0
arr2d[0][2]
arr2d[0, 2]


# In[28]:


#! ipython id=0acb0f0bb57646a0948d4456d2b0b7c2
arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
arr3d


# In[29]:


#! ipython id=97d41836d6b54dfba8a4fb6597966d3a
arr3d[0]


# In[30]:


#! ipython id=8e8691e0383e46df88a0052b67feda29
old_values = arr3d[0].copy()
arr3d[0] = 42
arr3d
arr3d[0] = old_values
arr3d


# In[31]:


#! ipython id=da598a108c024fbe911e91df67f58f94
arr3d[1, 0]


# In[32]:


#! ipython id=d779c7e844aa42a29fd3a72ac5584270
x = arr3d[1]
x
x[0]


# In[33]:


#! ipython id=cc6ad0926b53492aba7ed7afb2b15418
arr
arr[1:6]


# In[34]:


#! ipython id=c5474da10fd14eb3aea3575f86e1d7ed
arr2d
arr2d[:2]


# In[35]:


#! ipython id=e70d77b768cd4341b346b726933e44d3
arr2d[:2, 1:]


# In[36]:


#! ipython id=d5c54634ebee4441a379aed41221a768
lower_dim_slice = arr2d[1, :2]


# In[37]:


#! ipython id=dc5163856fd84027b6dd2039342fe262
lower_dim_slice.shape


# In[38]:


#! ipython id=6240d76c64eb474dbd9a2e25809a9344
arr2d[:2, 2]


# In[39]:


#! ipython id=a8d9d0df78b7429091825169e748eaa2
arr2d[:, :1]


# In[40]:


#! ipython id=953cee364dbc4eb6b21569a24927798b
arr2d[:2, 1:] = 0
arr2d


# In[41]:


#! ipython id=b74784ed32e643fd852c39a37ffd56d0
names = np.array(["Bob", "Joe", "Will", "Bob", "Will", "Joe", "Joe"])
data = np.array([[4, 7], [0, 2], [-5, 6], [0, 0], [1, 2],
                 [-12, -4], [3, 4]])
names
data


# In[42]:


#! ipython id=3b14eea790cc4b4583c2336a3862205d
names == "Bob"


# In[43]:


#! ipython id=1c5109d52691483d9636434cff981bdf
data[names == "Bob"]


# In[44]:


#! ipython id=aa825b8cb74e4d2299f2dbc2fdddeef4
data[names == "Bob", 1:]
data[names == "Bob", 1]


# In[45]:


#! ipython id=134639f975f14caabfcc947a21e47ff4
names != "Bob"
~(names == "Bob")
data[~(names == "Bob")]


# In[46]:


#! ipython id=5e22b0c2e98e4c0db8d91a3c533ae265
cond = names == "Bob"
data[~cond]


# In[47]:


#! ipython id=472d33a0a1fd4e2e9867dd2a480a7057
mask = (names == "Bob") | (names == "Will")
mask
data[mask]


# In[48]:


#! ipython id=82f7506ff90c4d7ab4566a5772d31d05
data[data < 0] = 0
data


# In[49]:


#! ipython id=25b0df346bec45bcb99bfd2cfcf837d1
data[names != "Joe"] = 7
data


# In[50]:


#! ipython id=ab1d36f594744a4d80d4f588bf2a8730
arr = np.zeros((8, 4))
for i in range(8):
    arr[i] = i
arr


# In[51]:


#! ipython id=3446842767e74bf08df8ece4191e228a
arr[[4, 3, 0, 6]]


# In[52]:


#! ipython id=ef21cf603d014a8890d5f84870a2c87c
arr[[-3, -5, -7]]


# In[53]:


#! ipython id=068a901c2cae41a79fad9ced9f8ddef7
arr = np.arange(32).reshape((8, 4))
arr
arr[[1, 5, 7, 2], [0, 3, 1, 2]]


# In[54]:


#! ipython id=451615d4351f459fb8bf29552a0a4ec2
arr[[1, 5, 7, 2]][:, [0, 3, 1, 2]]


# In[55]:


#! ipython id=6e158e85f5124d059a6e4d778332f650
arr[[1, 5, 7, 2], [0, 3, 1, 2]]
arr[[1, 5, 7, 2], [0, 3, 1, 2]] = 0
arr


# In[56]:


#! ipython id=ecf5f6b6a0e04778830c298839b80030
arr = np.arange(15).reshape((3, 5))
arr
arr.T


# In[57]:


#! ipython id=feeb9a3a39814003849455556ac06226
arr = np.array([[0, 1, 0], [1, 2, -2], [6, 3, 2], [-1, 0, -1], [1, 0, 1]])
arr
np.dot(arr.T, arr)


# In[58]:


#! ipython id=FD4EF8FA96A844A9843F1AD3757FE1A9
arr.T @ arr


# In[59]:


#! ipython id=6deb5a9b3df94092a22d96307ab7d954
arr
arr.swapaxes(0, 1)


# In[60]:


#! ipython id=46098a5c730449d98a74ac297f64abbc
samples = np.random.standard_normal(size=(4, 4))
samples


# In[61]:


#! ipython id=13e708f90e034b5ead6b0350698c3a18
from random import normalvariate
N = 1_000_000
get_ipython().run_line_magic('timeit', 'samples = [normalvariate(0, 1) for _ in range(N)]')
get_ipython().run_line_magic('timeit', 'np.random.standard_normal(N)')


# In[62]:


#! ipython id=bf957faf233f444796e4334e130b5717
rng = np.random.default_rng(seed=12345)
data = rng.standard_normal((2, 3))


# In[63]:


#! ipython id=c78d186866fe4cd5a533b5286e29a676
type(rng)


# In[64]:


#! ipython id=a0283c014e304d979bdd777a383f0246
arr = np.arange(10)
arr
np.sqrt(arr)
np.exp(arr)


# In[65]:


#! ipython id=92e3208042994937bab1a86b422f35d1
x = rng.standard_normal(8)
y = rng.standard_normal(8)
x
y
np.maximum(x, y)


# In[66]:


#! ipython id=7fd2522a5f084f76a7460a0c52aacab3
arr = rng.standard_normal(7) * 5
arr
remainder, whole_part = np.modf(arr)
remainder
whole_part


# In[67]:


#! ipython id=f048d6c61245412cb96ab1b57c13059a
arr
out = np.zeros_like(arr)
np.add(arr, 1)
np.add(arr, 1, out=out)
out


# In[68]:


#! ipython id=02493b7ab002498bb1cb83b74f7d7ac3
points = np.arange(-5, 5, 0.01) # 100 equally spaced points
xs, ys = np.meshgrid(points, points)
ys


# In[69]:


#! ipython id=b4da7227c86d45af9d2e317194d8f139
z = np.sqrt(xs ** 2 + ys ** 2)
z


# In[70]:


#! ipython id=3147d7f7f73f4007999abb7fa0a86e4c
import matplotlib.pyplot as plt
plt.imshow(z, cmap=plt.cm.gray, extent=[-5, 5, -5, 5])
plt.colorbar()
plt.title("Image plot of $\sqrt{x^2 + y^2}$ for a grid of values")


# In[71]:


#! ipython suppress id=74a786384bdb4ce6b12d36550d2692b5
#! figure,id=numpy_vectorize_circle,title="Plot of function evaluated on a grid",width=3in
plt.draw()


# In[72]:


#! ipython id=7600fde54ed649749f998eafcca31012
plt.close("all")


# In[73]:


#! ipython id=e7dd323ea42b4065830c1a6884ddd39b
xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
cond = np.array([True, False, True, True, False])


# In[74]:


#! ipython id=650e5042836c4cc08137e097a40c157d
result = [(x if c else y)
          for x, y, c in zip(xarr, yarr, cond)]
result


# In[75]:


#! ipython id=3ca8ca53322f413e8f9314c9856457ee
result = np.where(cond, xarr, yarr)
result


# In[76]:


#! ipython id=742abaae66da4816a73b6e7219cb05b7
arr = rng.standard_normal((4, 4))
arr
arr > 0
np.where(arr > 0, 2, -2)


# In[77]:


#! ipython id=24cc9834de534d7fa61d6c2646431e7a
np.where(arr > 0, 2, arr) # set only positive values to 2


# In[78]:


#! ipython id=3fdd850d25954e35bef515ab9c317c6e
arr = rng.standard_normal((5, 4))
arr
arr.mean()
np.mean(arr)
arr.sum()


# In[79]:


#! ipython id=f57ab0bb15004b49adf00d9ac3eaafcc
arr.mean(axis=1)
arr.sum(axis=0)


# In[80]:


#! ipython id=d9d2af75bc02480f8ec96ab405e0bfbc
arr = np.array([0, 1, 2, 3, 4, 5, 6, 7])
arr.cumsum()


# In[81]:


#! ipython id=6325620b7b21486eb341f9bdd26d0592
arr = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
arr


# In[82]:


#! ipython id=fee512a8c31e4821b880952d5dcb9871
arr.cumsum(axis=0)
arr.cumsum(axis=1)


# In[83]:


#! ipython id=c9b9d9305fa74033b0e4f8cf592b71ab
arr = rng.standard_normal(100)
(arr > 0).sum() # Number of positive values
(arr <= 0).sum() # Number of non-positive values


# In[84]:


#! ipython id=2509d496877b4d11b4fdabb5203d828a
bools = np.array([False, False, True, False])
bools.any()
bools.all()


# In[85]:


#! ipython id=d2abca3990204292845fa1773445ed1e
arr = rng.standard_normal(6)
arr
arr.sort()
arr


# In[86]:


#! ipython id=9706c0664763497abcc763ea917ac005
arr = rng.standard_normal((5, 3))
arr


# In[87]:


#! ipython id=e5efdd03e40b46ef90fbff8c2bc7cd58
arr.sort(axis=0)
arr
arr.sort(axis=1)
arr


# In[88]:


#! ipython id=C0B5AB6E22444377A78BE7A4ADAEC149
arr2 = np.array([5, -10, 7, 1, 0, -3])
sorted_arr2 = np.sort(arr2)
sorted_arr2


# In[89]:


#! ipython id=d0661787b25b463cab49e5707f440dc1
names = np.array(["Bob", "Will", "Joe", "Bob", "Will", "Joe", "Joe"])
np.unique(names)
ints = np.array([3, 3, 3, 2, 2, 1, 1, 4, 4])
np.unique(ints)


# In[90]:


#! ipython id=defea8216e0c43aa8cac6e8044375f7a
sorted(set(names))


# In[91]:


#! ipython id=0228289e5be043c189e011251dcffdc6
values = np.array([6, 0, 0, 3, 2, 5, 6])
np.in1d(values, [2, 3, 6])


# In[92]:


#! ipython id=b3597533081e4753b6074823c3b5674f
arr = np.arange(10)
np.save("some_array", arr)


# In[93]:


#! ipython id=fb53288f306e4de7a7c24f3535bf7114
np.load("some_array.npy")


# In[94]:


#! ipython id=d587f08b74b9448eb1e24b7812b8a559
np.savez("array_archive.npz", a=arr, b=arr)


# In[95]:


#! ipython id=8ba454d47fde4bd082c3eb647ec642f9
arch = np.load("array_archive.npz")
arch["b"]


# In[96]:


#! ipython id=2a2682b4078c49c981d34e9966f55632
np.savez_compressed("arrays_compressed.npz", a=arr, b=arr)


# In[97]:


#! ipython suppress id=1d4c714fb55a49fe81a7c3dca0318267
get_ipython().system('rm some_array.npy')
get_ipython().system('rm array_archive.npz')
get_ipython().system('rm arrays_compressed.npz')


# In[98]:


#! ipython id=5ff70f667285496e9853f9ce5c69cac7
x = np.array([[1., 2., 3.], [4., 5., 6.]])
y = np.array([[6., 23.], [-1, 7], [8, 9]])
x
y
x.dot(y)


# In[99]:


#! ipython id=4a27f60b6ad34c179277e3567556f6ad
np.dot(x, y)


# In[100]:


#! ipython id=c604db1abc904450aecf44582381e04e
x @ np.ones(3)


# In[101]:


#! ipython id=f6b72d7a1ad443768226bdf5a3aed368
from numpy.linalg import inv, qr
X = rng.standard_normal((5, 5))
mat = X.T @ X
inv(mat)
mat @ inv(mat)


# In[102]:


#! ipython verbatim id=9060763bcc2b4f5e91a19bb8d8d5d887
#! blockstart
import random
position = 0
walk = [position]
nsteps = 1000
for _ in range(nsteps):
    step = 1 if random.randint(0, 1) else -1
    position += step
    walk.append(position)
#! blockend


# In[103]:


#! ipython suppress id=99ca0670c2f946c6af46c9292c6180fa
plt.figure()


# In[104]:


#! ipython id=373f266d65ca4e5c87d443fa4d1e706c
#! figure,id=figure_random_walk1,title="A simple random walk",width=4in
plt.plot(walk[:100])


# In[105]:


#! ipython id=90295621e09c479590357865bd506c9a
nsteps = 1000
rng = np.random.default_rng(seed=12345)  # fresh random generator
draws = rng.integers(0, 2, size=nsteps)
steps = np.where(draws == 0, 1, -1)
walk = steps.cumsum()


# In[106]:


#! ipython id=0e9f7bf0c3bd41f985b1b0e297b3d4c7
walk.min()
walk.max()


# In[107]:


#! ipython id=20c784b0675246eeb99329c5d7394101
(np.abs(walk) >= 10).argmax()


# In[108]:


#! ipython id=79f63c1dd0b34fe78ac34492f9bfb0a0
nwalks = 5000
nsteps = 1000
draws = rng.integers(0, 2, size=(nwalks, nsteps)) # 0 or 1
steps = np.where(draws > 0, 1, -1)
walks = steps.cumsum(axis=1)
walks


# In[109]:


#! ipython id=351f44c55f494a9ab09e273bbfc258f5
walks.max()
walks.min()


# In[110]:


#! ipython id=319266ed7c2c4e749b488168eef3f1c6
hits30 = (np.abs(walks) >= 30).any(axis=1)
hits30
hits30.sum() # Number that hit 30 or -30


# In[111]:


#! ipython id=50fb5813d360469bb339b7e6de32eae7
crossing_times = (np.abs(walks[hits30]) >= 30).argmax(axis=1)
crossing_times


# In[112]:


#! ipython id=c529b3e238a641c8a5beae0755683656
crossing_times.mean()


# In[113]:


#! ipython id=9cbb0b66c9e0461ab0149703bfbe8834
draws = 0.25 * rng.standard_normal((nwalks, nsteps))


# In[114]:


#! ipython suppress id=9b38f187f59e46c3bc91061ee6b7f998
get_ipython().run_line_magic('popd', '')

