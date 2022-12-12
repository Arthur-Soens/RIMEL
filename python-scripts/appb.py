#!/usr/bin/env python
# coding: utf-8

# In[1]:


#! ipython suppress id=009f36b80b8e4e0f8571b70fa54e4511
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


#! ipython id=93f746046cba429da26cf71ac13fc271
# a very large list of strings
strings = ['foo', 'foobar', 'baz', 'qux',
           'python', 'Guido Van Rossum'] * 100000

method1 = [x for x in strings if x.startswith('foo')]

method2 = [x for x in strings if x[:3] == 'foo']


# In[3]:


#! ipython id=1cddb2d3bd564e93bd66d8d625b93b30
get_ipython().run_line_magic('time', "method1 = [x for x in strings if x.startswith('foo')]")
get_ipython().run_line_magic('time', "method2 = [x for x in strings if x[:3] == 'foo']")


# In[4]:


#! ipython suppress id=362c743f25684f409c9562cffa1a76db
get_ipython().run_line_magic('popd', '')


# In[5]:


#! ipython suppress id=eb88d21e8e59441e884dda4884f3d0f8
pd.options.display.max_rows = PREVIOUS_MAX_ROWS

