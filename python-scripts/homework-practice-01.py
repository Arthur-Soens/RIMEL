#!/usr/bin/env python
# coding: utf-8

# # Машинное обучение, ФКН ВШЭ
# 
# ## Практическое задание 1
# 
# ### Общая информация
# Дата выдачи: 06.09.2019  
# 
# Мягкий дедлайн: 23:59MSK 15.09.2019 (за каждый день просрочки снимается 1 балл)
# 
# Жесткий дедлайн: 23:59MSK 17.09.2019  

# ### О задании
# 
# Задание состоит из двух разделов, посвященных работе с табличными данными с помощью библиотеки pandas и визуализации с помощью matplotlib. В каждом разделе вам предлагается выполнить несколько заданий. Баллы даются за выполнение отдельных пунктов. Задачи в рамках одного раздела рекомендуется решать в том порядке, в котором они даны в задании.
# 
# Задание направлено на освоение jupyter notebook (будет использоваться в дальнейших заданиях), библиотекам pandas и matplotlib.
# 
# ### Оценивание и штрафы
# Каждая из задач имеет определенную «стоимость» (указана в скобках около задачи). Максимально допустимая оценка за работу — 10 баллов.
# 
# Сдавать задание после указанного срока сдачи нельзя. При выставлении неполного балла за задание в связи с наличием ошибок на усмотрение проверяющего предусмотрена возможность исправить работу на указанных в ответном письме условиях.
# 
# Задание выполняется самостоятельно. «Похожие» решения считаются плагиатом и все задействованные студенты (в том числе те, у кого списали) не могут получить за него больше 0 баллов (подробнее о плагиате см. на странице курса). Если вы нашли решение какого-то из заданий (или его часть) в открытом источнике, необходимо указать ссылку на этот источник в отдельном блоке в конце вашей работы (скорее всего вы будете не единственным, кто это нашел, поэтому чтобы исключить подозрение в плагиате, необходима ссылка на источник).
# 
# ### Формат сдачи
# Задания сдаются через систему Anytask. Инвайт можно найти на странице курса. Присылать необходимо ноутбук с выполненным заданием. 
# 
# Для удобства проверки самостоятельно посчитайте свою максимальную оценку (исходя из набора решенных задач) и укажите ниже.
# 
# Оценка: xx.

# ## 0. Введение

# Сейчас мы находимся в jupyter-ноутбуке (или ipython-ноутбуке). Это удобная среда для написания кода, проведения экспериментов, изучения данных, построения визуализаций и других нужд, не связанных с написаем production-кода. 
# 
# Ноутбук состоит из ячеек, каждая из которых может быть либо ячейкой с кодом, либо ячейкой с текстом размеченным и неразмеченным. Текст поддерживает markdown-разметку и формулы в Latex.
# 
# Для работы с содержимым ячейки используется *режим редактирования* (*Edit mode*, включается нажатием клавиши **Enter** после выбора ячейки), а для навигации между ячейками искользуется *командный режим* (*Command mode*, включается нажатием клавиши **Esc**). Тип ячейки можно задать в командном режиме либо с помощью горячих клавиш (**y** to code, **m** to markdown, **r** to edit raw text), либо в меню *Cell -> Cell type*. 
# 
# После заполнения ячейки нужно нажать *Shift + Enter*, эта команда обработает содержимое ячейки: проинтерпретирует код или сверстает размеченный текст.

# In[ ]:


# ячейка с кодом, при выполнении которой появится output
2 + 2

# А это ___ячейка с текстом___.
Ячейка с неразмеченыным текстом.
# Попробуйте создать свои ячейки, написать какой-нибудь код и текст какой-нибудь формулой.

# In[ ]:


# your code

# [Здесь](https://athena.brynmawr.edu/jupyter/hub/dblank/public/Jupyter%20Notebook%20Users%20Manual.ipynb) находится <s>не</s>большая заметка о используемом языке разметки Markdown. Он позволяет:
# 
# 0. Составлять упорядоченные списки
# 1. #Делать 
# ##заголовки 
# ###разного уровня
# 3. Выделять *текст* <s>при</s> **необходимости**
# 4. Добавлять [ссылки](http://imgs.xkcd.com/comics/the_universal_label.png)
# 
# 
# * Составлять неупорядоченные списки
# 
# Делать вставки с помощью LaTex:
#     
# $
# \left\{
# \begin{array}{ll}
# x = 16 \sin^3 (t) \\ 
# y = 13 \cos (t) - 5 \cos (2t) - 2 \cos (3t) - \cos (4t) \\
# t \in [0, 2 \pi]
# \end{array}
# \right.$

# А ещё можно вставлять картинки:
# <img src="https://st2.depositphotos.com/1177973/9266/i/950/depositphotos_92668716-stock-photo-red-cat-with-computer-keyboard.jpg" style="width: 400px">

# ## 1. Табличные данные и Pandas

# Pandas — удобная библиотека для работы с табличными данными в Python, если данных не слишком много и они помещаются в оперативную память вашего компьютера. Несмотря на неэффективность реализации и некоторые проблемы, библиотека стала стандартом в анализе данных. С этой библиотекой мы сейчас и познакомимся.
# 
# Основной объект в pandas это DataFrame, представляющий собой таблицу с именованными колонками различных типов, индексом (может быть многоуровневым). DataFrame можно создавать, считывая таблицу из файла или задавая вручную из других объектов.
# 
# В этой части потребуется выполнить несколько небольших заданий. Можно пойти двумя путями: сначала изучить материалы, а потом приступить к заданиям, или же разбираться "по ходу". Выбирайте сами.
# 
# Материалы:
# 1. [Pandas за 10 минут из официального руководства](http://pandas.pydata.org/pandas-docs/stable/10min.html)
# 2. [Документация](http://pandas.pydata.org/pandas-docs/stable/index.html) (стоит обращаться, если не понятно, как вызывать конкретный метод)
# 3. [Примеры использования функционала](http://nbviewer.jupyter.org/github/justmarkham/pandas-videos/blob/master/pandas.ipynb)
# 
# Многие из заданий можно выполнить несколькими способами. Не существуют единственно верного, но попробуйте максимально задействовать арсенал pandas и ориентируйтесь на простоту и понятность вашего кода. Мы не будем подсказывать, что нужно использовать для решения конкретной задачи, попробуйте находить необходимый функционал сами (название метода чаще всего очевидно). В помощь вам документация, поиск и stackoverflow.

# In[ ]:


%pylab inline  
# import almost all we need
import pandas as pd

# Данные можно скачать [отсюда](https://www.dropbox.com/s/5qq94wzmbw4e54r/data.csv?dl=0).

# #### 1. [0.5 баллов] Откройте файл с таблицей (не забудьте про её формат). Выведите последние 10 строк.
# 
# Посмотрите на данные и скажите, что они из себя представляют, сколько в таблице строк, какие столбцы?

# In[ ]:


# your code

# #### 2. [0.25 баллов] Ответьте на вопросы:
# 1. Сколько заказов попало в выборку?
# 2. Сколько уникальных категорий товара было куплено? (item_name)

# In[ ]:


# your code

# #### 3. [0.25 баллов] Есть ли в данных пропуски? В каких колонках? 

# In[ ]:


# your code

# Заполните пропуски пустой строкой для строковых колонок и нулём для числовых.

# In[ ]:


# your code

# #### 4. [0.5 баллов] Посмотрите внимательнее на колонку с ценой товара. Какого она типа? Создайте новую колонку так, чтобы в ней цена была числом.
# 
# Для этого попробуйте применить функцию-преобразование к каждой строке вашей таблицы (для этого есть соответствующая функция).

# In[ ]:


# your code

# Какая средняя/минимальная/максимальная цена у товара? 

# In[ ]:


# your code

# Удалите старую колонку с ценой.

# In[ ]:


# your code

# #### 5. [0.25 баллов] Какие 5 товаров были самыми дешёвыми и самыми дорогими? (по item_name)
# 
# Для этого будет удобно избавиться от дубликатов и отсортировать товары. Не забудьте про количество товара.

# In[ ]:


# your code

# #### 6. [0.5 баллов] Сколько раз клиенты покупали больше 1 Chicken Bowl (item_name)?

# In[ ]:


# your code

# #### 7. [0.5 баллов] Какой средний чек у заказа? Сколько в среднем товаров покупают?
# 
# Если необходимо провести вычисления в терминах заказов, то будет удобно сгруппировать строки по заказам и посчитать необходимые статистики.

# In[ ]:


# your code

# #### 8. [0.25 баллов] Сколько заказов содержали ровно 1 товар?

# In[ ]:


# your code

# #### 9. [0.25 баллов] Какая самая популярная категория товара? 

# In[ ]:


# your code

# #### 10. [0.5 баллов] Какие виды Burrito существуют? Какой из них чаще всего покупают? Какой из них самый дорогой? 

# In[ ]:


# your code

# #### 11. [0.75 баллов] В каком количестве заказов есть товар, который стоит более 40% от суммы всего чека?
# 
# Возможно, будет удобно посчитать отдельно среднюю стоимость заказа, добавить ее в исходные данные и сделать необходимые проверки.
# 
# *Данный комментарий стоит воспринимать как подсказку к одному из вариантов решений задания. Если в вашем варианте решения он не нужнен, это не страшно*

# In[ ]:


# your code

# #### 12. [0.75 баллов] Предположим, что в данных была ошибка и Diet Coke (choice_description), который стоил $1.25, должен был стоить 1.35. Скорректируйте данные в таблицы и посчитайте, на какой процент больше денег было заработано с этого товара. Не забывайте, что количество товара не всегда равно 1.

# In[ ]:


# your code

# #### 13. [0.75 баллов] Создайте новый DateFrame из матрицы, созданной ниже. Назовите колонки index, column1, column2 и сделайте первую колонку индексом.

# In[ ]:


data = np.random.rand(10, 3)

# your code

# Сохраните DataFrame на диск в формате csv без индексов и названий столбцов.

# In[ ]:


# your code

# ## 2. Визуализации и matplotlib

# При работе с данными часто неудобно делать какие-то выводы, если смотреть на таблицу и числа в частности, поэтому важно уметь визуализировать данные. В этом разделе мы этим и займёмся.
# 
# У matplotlib, конечно, же есть [документация](https://matplotlib.org/users/index.html) с большим количеством [примеров](https://matplotlib.org/examples/), но для начала достаточно знать про несколько основных типов графиков:
# - plot — обычный поточечный график, которым можно изображать кривые или отдельные точки;
# - hist — гистограмма, показывающая распределение некоторое величины;
# - scatter — график, показывающий взаимосвязь двух величин;
# - bar — столбцовый график, показывающий взаимосвязь количественной величины от категориальной.
# 
# В этом задании вы попробуете построить каждый из них. Не менее важно усвоить базовые принципы визуализаций:
# - на графиках должны быть подписаны оси;
# - у визуализации должно быть название;
# - если изображено несколько графиков, то необходима поясняющая легенда;
# - все линии на графиках должны быть чётко видны (нет похожих цветов или цветов, сливающихся с фоном);
# - если отображена величина, имеющая очевидный диапазон значений (например, проценты могут быть от 0 до 100), то желательно масштабировать ось на весь диапазон значений (исключением является случай, когда вам необходимо показать малое отличие, которое незаметно в таких масштабах).
# - сетка на графике помогает оценить значения в точках на глаз, это обычно полезно, поэтому лучше ее отрисовывать.

# In[ ]:


%matplotlib inline  # нужно для отображения графиков внутри ноутбука
import matplotlib.pyplot as plt

# На самом деле мы уже импортировали matplotlib внутри %pylab inline в начале задания.
# 
# Работать мы будем с той же выборкой покупкок. Добавим новую колонку с датой покупки.

# In[ ]:


import datetime

start = datetime.datetime(2018, 1, 1)
end = datetime.datetime(2018, 1, 31)
delta_seconds = int((end - start).total_seconds())

dates = pd.DataFrame(index=df.order_id.unique())
dates['date'] = [
    (start + datetime.timedelta(seconds=random.randint(0, delta_seconds))).strftime('%Y-%m-%d')
    for _ in range(df.order_id.nunique())]

# если DataFrame с покупками из прошлого заказа называется не df, замените на ваше название ниже
df['date'] = df.order_id.map(dates['date'])

# #### 1. [1 балл] Постройте гистограмму распределения сумм покупок и гистограмму средних цен отдельных видов продуктов item_name. 
# 
# Изображайте на двух соседних графиках. Для этого может быть полезен subplot.

# In[ ]:


# your code

# #### 2. [1 балл] Постройте график зависимости суммы покупок от дней.

# In[ ]:


# your code

# #### 3. [1 балл] Постройте средних сумм покупок по дням недели (bar plot).

# In[ ]:


# your code

# #### 4. [1 балл] Постройте график зависимости денег за товар от купленного количества (scatter plot).

# In[ ]:


# your code

# Сохраните график в формате pdf (так он останется векторизованным).

# In[ ]:


# your code

# Еще одна билиотека для визуализации: [seaborn](https://jakevdp.github.io/PythonDataScienceHandbook/04.14-visualization-with-seaborn.html). Это настройка над matplotlib, иногда удобнее и красивее делать визуализации через неё. 