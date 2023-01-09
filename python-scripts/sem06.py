#!/usr/bin/env python
# coding: utf-8

# # Машинное обучение, ФКН ВШЭ
# 
# # Семинар 6

# In[1]:


%pylab inline

# ## Работа с текстовыми данными
# 
# Разреженные матрицы имеют место в машинном обучении, в частности, в задачах обработки текстов. 
# 
# Как правило, модели машинного обучения действуют в предположении, что матрица "объект-признак" является вещественнозначной, поэтому при работе с текстами сперва для каждого из них необходимо составить его признаковое описание. Для этого широко используются техники векторизации, tf-idf и пр. Рассмотрим их на примере [датасета](https://www.dropbox.com/s/f9xsff8xluriy95/banki_responses.json.bz2?dl=0) отзывов о банках.
# 
# Сперва загрузим данные:

# In[3]:


from tqdm import tqdm
from sklearn.datasets import fetch_20newsgroups

# In[4]:


data = fetch_20newsgroups(subset='all', categories=['comp.graphics', 'sci.med'])

# Данные содержат тексты новостей, которые надо классифицировать на разделы.

# In[5]:


data['target_names']

# In[6]:


texts = data['data']
target = data['target']

# Например:

# In[7]:


texts[0]

# In[8]:


data['target_names'][target[0]]

# ### Векторизация
# 
# Самый очевидный способ формирования признакового описания текстов — векторизация. Пусть у нас имеется коллекция текстов $D = \{d_i\}_{i=1}^l$ и словарь всех слов, встречающихся в выборке $V = \{v_j\}_{j=1}^d.$ В этом случае некоторый текст $d_i$ описывается вектором $(x_{ij})_{j=1}^d,$ где
# $$x_{ij} = \sum_{v \in d_i} [v = v_j].$$
# 
# Таким образом, текст $d_i$ описывается вектором количества вхождений каждого слова из словаря в данный текст.

# In[9]:


from sklearn.feature_extraction.text import CountVectorizer

# In[10]:


vectorizer = CountVectorizer(encoding='utf8', min_df=5)
_ = vectorizer.fit(texts)

# Результатом является разреженная матрица.

# In[11]:


vectorizer.transform(texts[:1])

# In[12]:


print(vectorizer.transform(texts[:1]).indptr)
print(vectorizer.transform(texts[:1]).indices)
print(vectorizer.transform(texts[:1]).data)

# ### TF-IDF
# 
# Ещё один способ работы с текстовыми данными — [TF-IDF](https://en.wikipedia.org/wiki/Tf–idf) (**T**erm **F**requency–**I**nverse **D**ocument **F**requency). Рассмотрим коллекцию текстов $D$.  Для каждого уникального слова $t$ из документа $d \in D$ вычислим следующие величины:
# 
# 1. Term Frequency – количество вхождений слова в отношении к общему числу слов в тексте:
# $$\text{tf}(t, d) = \frac{n_{td}}{\sum_{t \in d} n_{td}},$$
# где $n_{td}$ — количество вхождений слова $t$ в текст $d$.
# 1. Inverse Document Frequency
# $$\text{idf}(t, D) = \log \frac{\left| D \right|}{\left| \{d\in D: t \in d\} \right|},$$
# где $\left| \{d\in D: t \in d\} \right|$ – количество текстов в коллекции, содержащих слово $t$.
# 
# Тогда для каждой пары (слово, текст) $(t, d)$ вычислим величину:
# $$\text{tf-idf}(t,d, D) = \text{tf}(t, d)\cdot \text{idf}(t, D).$$
# 
# Отметим, что значение $\text{tf}(t, d)$ корректируется для часто встречающихся общеупотребимых слов при помощи значения $\text{idf}(t, D).$
# 
# Признаковым описанием одного объекта $d \in D$ будет вектор $\bigg(\text{tf-idf}(t,d, D)\bigg)_{t\in V}$, где $V$ – словарь всех слов, встречающихся в коллекции $D$.

# In[13]:


from sklearn.feature_extraction.text import TfidfVectorizer

# In[14]:


vectorizer = TfidfVectorizer(encoding='utf8', min_df=5)
_ = vectorizer.fit(texts)

# На выходе получаем разреженную матрицу.

# In[15]:


vectorizer.transform(texts[:1])

# In[16]:


print(vectorizer.transform(texts[:1]).indptr)
print(vectorizer.transform(texts[:1]).indices)
print(vectorizer.transform(texts[:1]).data)

# Заметим, что оба метода возвращают вектор длины 35189 (размер нашего словаря).

# ## Классификация
# 
# Воспользуемся изученными методами обработки текстов для решения задачи классификации.

# In[17]:


vectorizer = CountVectorizer(encoding='utf8', min_df=5)
vectorizer.fit(texts)

# In[18]:


X = vectorizer.transform(texts)

# In[19]:


from sklearn.cross_validation import cross_val_score, ShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

# In[20]:


cv = ShuffleSplit(X.shape[0], n_iter=1, test_size=0.3)
for train_ids, test_ids in cv:
    lr = LogisticRegression()
    lr.fit(X[train_ids], target[train_ids])
    preds = lr.predict_proba(X[test_ids])[:,1]
    print('ROC-AUC: %.3f, ACC: %.3f' % (roc_auc_score(target[test_ids], preds), 
                                        accuracy_score(target[test_ids], (preds > 0.5).astype(int))))

# То же самое с tf-idf.

# In[21]:


vectorizer = TfidfVectorizer(encoding='utf8', min_df=5)
vectorizer.fit(texts)

# In[22]:


X = vectorizer.transform(texts)

# In[23]:


cv = ShuffleSplit(X.shape[0], n_iter=1, test_size=0.3)
for train_ids, test_ids in cv:
    lr = LogisticRegression()
    lr.fit(X[train_ids], target[train_ids])
    preds = lr.predict_proba(X[test_ids])[:,1]
    print('ROC-AUC: %.3f, ACC: %.3f' % (roc_auc_score(target[test_ids], preds), 
                                        accuracy_score(target[test_ids], (preds > 0.5).astype(int))))

# ## Важность признаков
# 
# Как уже упоминалось ранее, веса признаков в линейной модели в случае, если признаки отмасштабированы, характеризуют степень их влияния на значение целевой переменной. В задаче классификации текстов, кроме того, признаки являются хорошо интерпретируемыми, поскольку каждый из них соответствует конкретному слову. Изучим влияние конкретных слов на значение целевой переменной:

# In[24]:


f_weights = zip(vectorizer.get_feature_names(), lr.coef_[0])
f_weights = sorted(f_weights, key=lambda i: i[1])
for i in range(1,30):
    print('%s, %.2f' % f_weights[-i])
    
print('...')
for i in reversed(range(1,30)):
    print('%s, %.2f' % f_weights[i])

# ## Лемматизация и стемминг
# 
# Заметим, что одно и то же слово может встречаться в различных формах (например, "сотрудник" и "сотрудника"), но описанные выше методы интерпретируют их как различные слова, что делает признаковое описание избыточным. Устранить эту проблему можно при помощи **лемматизации** и **стемминга**.
# 
# ### Стемминг
# 
# [**Stemming**](https://en.wikipedia.org/wiki/Stemming) –  это процесс нахождения основы слова. В результате применения данной процедуры однокоренные слова, как правило, преобразуются к одинаковому виду.
# 
# **Примеры стемминга:**
# 
# | Word        | Stem           |
# | ----------- |:-------------:|
# | вагон | вагон |
# | вагона | вагон |
# | вагоне | вагон |
# | вагонов | вагон |
# | вагоном | вагон |
# | вагоны | вагон |
# | важная | важн |
# | важнее | важн |
# | важнейшие | важн |
# | важнейшими | важн |
# | важничал | важнича |
# | важно | важн |
# 
# [Snowball](http://snowball.tartarus.org/) – фрэймворк для написания алгоритмов стемминга. Алгоритмы стемминга отличаются для разных языков и используют знания о конкретном языке – списки окончаний для разных чистей речи, разных склонений и т.д. Пример алгоритма для русского языка – [Russian stemming](http://snowballstem.org/algorithms/russian/stemmer.html).

# In[25]:


import nltk

# In[26]:


stemmer = nltk.stem.snowball.RussianStemmer()
print(stemmer.stem(u'машинное'), stemmer.stem(u'обучение'))

# In[27]:


stemmer = nltk.stem.snowball.EnglishStemmer()

def stem_text(text, stemmer):
    tokens = text.split()
    return ' '.join(map(lambda w: stemmer.stem(w), tokens))

stemmed_texts = []
for t in tqdm(texts[:1000]):
    stemmed_texts.append(stem_text(t, stemmer))

# In[28]:


print(texts[0])

# In[29]:


print(stemmed_texts[0])

# Как видим, стеммер работает не очень быстро и запускать его для всей выборки достаточно накладно.

# ### Лемматизация
# 
# [Лемматизация](https://en.wikipedia.org/wiki/Lemmatisation) — процесс приведения слова к его нормальной форме (**лемме**):
# - для существительных — именительный падеж, единственное число;
# - для прилагательных — именительный падеж, единственное число, мужской род;
# - для глаголов, причастий, деепричастий — глагол в инфинитиве.

# Например, для русского языка есть библиотека pymorphy2.

# In[30]:


import pymorphy2
morph = pymorphy2.MorphAnalyzer()

# In[31]:


morph.parse('играющих')[0]

# Сравним работу стеммера и лемматизатора на примере:

# In[32]:


stemmer = nltk.stem.snowball.RussianStemmer()
print(stemmer.stem('играющих'))

# In[33]:


print(morph.parse('играющих')[0].normal_form)

# ## Разреженные матрицы
# 
# Разреженная матрица — это матрица, большинство элементов которой равны нулю. Такие матрицы возникают во многих областях науки, в том числе и в машинном обучении.
# 
# Для разреженных матриц можно определить следующие характеристики:
# - разреженность (sparsity) — доля нулевых элементов матрицы,
# - плотность (density) — доля ненулевых элементов матрицы, или $1 - \text(sparsity)$.
# 
# Для разреженных матриц существуют специальные способы их хранения в памяти компьютера, при которых хранятся только ненулевые значения, тем самым сокращается объём занимаемой памяти. Эти способы реализованы в библиотеке [scipy.sparse](http://docs.scipy.org/doc/scipy/reference/sparse.html). Кроме того, разреженные матрицы поддерживаются большинством реализаций методов машинного обучения.

# In[34]:


import numpy as np
import scipy.sparse as sp

# ### COOrdinate format
# 
# [Координатный формат](http://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html#scipy.sparse.coo_matrix) задаёт матрицу при помощи троек (индекс строки, индекс столбца, значение элемента), описывающих ненулевые элементы матрицы. Как правило, тройки сортируют по индексу строки, а затем индексу столбца для ускорения работы. 
# 
# Объём занимаемой памяти — $O(n),$ где $n$ — число ненулевых элементов в матрице.

# In[36]:


m = (np.arange(9) + 1).reshape(3,3)
print(m)
sparse_m = sp.coo_matrix(m)

# In[37]:


for i in range(len(sparse_m.data)):
    print('(%d, %d, %d)' % (sparse_m.row[i], sparse_m.col[i], sparse_m.data[i]))

# Для матрицы, содержащей нулевые элементы, имеем:

# In[38]:


m = np.eye(3)*np.arange(1,4)
print(m)
sparse_m = sp.coo_matrix(m)

# In[39]:


for i in range(len(sparse_m.data)):
    print '(%d, %d, %d)' % (sparse_m.row[i], sparse_m.col[i], sparse_m.data[i])

# ### Compressed Sparse Row matrix
# 
# [CSR формат](http://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html#scipy.sparse.csr_matrix) - разреженная по строчкам матрица. 
# 
# <img src="images/arrays.png">
# 
# Формат задаёт матрицу при помощи трёх массивов:
# 1. $i$-ый элемент первого массива соответствует $i$-ой строке и содержит индекс некоторого элемента во втором массиве,
# 2. во втором массиве по порядку для каждой строки записаны индексы столбцов ненулевых элементов,
# 3. третий массив имеет такую же длину, как и второй, и содержит значения соответствующих ненулевых элементов.
# 
# Обозначим описанные массивы $a,b,c$. Для получения элемента матрицы на позиции $(i, j)$ необходимо осуществить следующую последовательность действий:
# 1. Получить значения $a[i]=k_{left}, a[i+1]=k_{right}$.
# 2. Тогда индексы столбцов ненулевых элементов $i$-ой строки будут находиться в "подмассиве" $b[k_{left}:k_{right}]$.
# 3. В цикле перебираем элементы подмассива $b[k_{left}:k_{right}]$, пока не встретим элемент, равный $j$.
# 4. Если такой элемент обнаружен на позиции $m$ (в терминах массива $b$), то ответом является значение $c[m]$.
# 5. Иначе ответом является 0.если мы не встретили элемент, равный $j$, то возвращаем $0$.
# 
# Объём занимаемой памяти — $O(n)$, где $n$ - число ненулевых элементов.

# In[ ]:


m = (np.arange(9) + 1).reshape(3,3)
print(m)
sparse_m = sp.csr_matrix(m)

# In[ ]:


print('a', sparse_m.indptr)
print('b', sparse_m.indices)
print('c', sparse_m.data)

# Для матрицы, содержащей нулевые элементы:

# In[ ]:


m = np.tril(np.arange(1,4))
print(m)
sparse_m = sp.csr_matrix(m)

# In[ ]:


print('a', sparse_m.indptr)
print('b', sparse_m.indices)
print('c', sparse_m.data)

# ### Compressed Sparse Column matrix
# 
# [CSC формат](http://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.html#scipy.sparse.csc_matrix) - разреженная по столбцам матрица. 
# 
# Формат CSC задаёт матрицу аналогично формату CSR, но при этом элементы первого массива соответствуют столбцам, а не строкам.
# 
# Объём занимаемой памяти — $O(n)$, где $n$ - число ненулевых элементов.

# In[ ]:


m = (np.arange(9) + 1).reshape(3,3)
print(m)
sparse_m = sp.csc_matrix(m)

# In[ ]:


print('a', sparse_m.indptr)
print('b', sparse_m.indices)
print('c', sparse_m.data)

# In[ ]:


m = np.tril(np.arange(1,4))
print(m)
sparse_m = sp.csc_matrix(m)

# In[ ]:


print('a', sparse_m.indptr)
print('b', sparse_m.indices)
print('c', sparse_m.data)

# ## Умножение разреженных матриц
# 
# Как мы убедились, объём занимаемой памяти практически не отличается для всех вариантов хранения разреженных матриц. В таком случае использование какого из вариантов даёт больше преимуществ? Оказывается, что все три способа кардинально различаются по времени умножения матриц.
# 
# Для начала вспомним правило умножения матриц:
# $$C = A\cdot B$$
# $$C_{ij} = \sum_k A_{ik}B_{kj}$$
# 
# Для нахождения элемента $C_{ij}$ необходимо получить $i$-ую строчку матрицы $A$ и $j$-ый столбец матрицы $B$. Исследуем время выполнения этих операций для каждого из форматов:
# 
# - **COO.** Стоимость получения строки — $O(n)$. Стоимость получения столбца — $O(n)$. При условии, что тройки отсортированы, время поиска можно сократить, воспользовавшись бинарным поиском.
# - **CSR.** Стоимость получения строки — $O(1)$. Стоимость получения столбца — $O(n)$.
# - **CSC.** Стоимость получения строки — $O(n)$. Стоимость получения столбца — $O(1)$.
# 
# Таким образом, время перемножения матриц будет оптимальным, если матрица $A$ задаётся в формате CSR, а матрица $B$ — в формате CSC.

# ## Разреженные матрицы в линейных моделях
# 
# Рассмотрим задачу линейной регрессии с функционалом качества MSE:
# 
# $$Q = ||Xw - y||^2 \rightarrow \min_{w}.$$
# 
# Как уже говорилось на предыдущих семинарах, вместо нахождения оптимального значения вектора $w$ используют градиентные методы оптимизации функционала. Запишем формулу его градиента:
# 
# $$\frac{\partial Q}{\partial w} = 2X^T(Xw - y).$$
# 
# Заметим, что матрица $X$, заданная в формате CSR, может быть представлена как $X^T$ в формате CSC (действительно, используя те же массивы, мы можем придать им "симметричный" смысл).
# 
# Рассмотрим, как осуществляется умножение разреженной матрицы $A$ на вектор $z$:
# 
# 1) **CSR**
# $$(Az)_{i} = \sum_{k}A_{ik}z_k.$$
# 
# Для матрицы в формале CSR обращение к строчкам матрицы выполняется за $O(1)$, поэтому перемножение выполняется за $O(n)$, где $n$ - кол-во ненулевых элементов матрицы $X$.
#     
# 2) **CSC** 
# 
# Для матрицы в формате CSC обращение к строчкам матрицы выполняется за $O(n)$. В этом случае умножение будем производить следующим образом:
#     - Аллоцируем результирующий вектор, который предварительно заполним нулями. 
#     - Обращаемся к $i$-ому столбцу матрицы $A$ и $i$-ому элементу вектора $z$.
#     - Каждый ненулевой элемент в столбце домножаем на $z_i$ и добавляем результат к соответствующему значению результирующего вектора.
#     
# Итого, для умножения разреженной матрицы на вектор получаем следующую асимптотику:
#  - $O(l)$ по памяти;
#  - $O(n)$ по времени.
# 
# Таким образом, мы описали процедуру умножения разреженной матрицы на вектор, и теперь её можно применить для вычисления градиента в задачах с разреженными матрицами "объект-признак".