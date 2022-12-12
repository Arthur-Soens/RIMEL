#!/usr/bin/env python
# coding: utf-8

# In[1]:


#! ipython suppress id=dbedddc4257c4aa8bff5f9b0d4d8c1bc
get_ipython().run_line_magic('pushd', 'book-materials')


# In[2]:


#! ipython id=87d4ac2ad13c491496f5706a050b89d0
tup = (4, 5, 6)
tup


# In[3]:


#! ipython id=6f7af42292e84654b7317956cc8f6e68
tup = 4, 5, 6
tup


# In[4]:


#! ipython id=9799bf9875e7452cb40082f8479a1ba6
tuple([4, 0, 2])
tup = tuple('string')
tup


# In[5]:


#! ipython id=5e49a072c38647488543066b9d082fcb
tup[0]


# In[6]:


#! ipython id=e499888dd95f4b4394ade24148f54180
nested_tup = (4, 5, 6), (7, 8)
nested_tup
nested_tup[0]
nested_tup[1]


# In[7]:


#! ipython allow_exceptions id=c17bbb6c35414b86b7bbacd140256406
tup = tuple(['foo', [1, 2], True])
tup[2] = False


# In[8]:


#! ipython id=48557abbe01245c682d53c11b2c6da39
tup[1].append(3)
tup


# In[9]:


#! ipython id=03226ed8d01b4c57a57259f9931a418c
(4, None, 'foo') + (6, 0) + ('bar',)


# In[10]:


#! ipython id=1e55cec4a8b343a691c0c58a039c59bc
('foo', 'bar') * 4


# In[11]:


#! ipython id=e5652393811c45f38146118b45dcbdcc
tup = (4, 5, 6)
a, b, c = tup
b


# In[12]:


#! ipython id=a6dfb4144dad4b2f9360dc94c618ed87
tup = 4, 5, (6, 7)
a, b, (c, d) = tup
d


# In[13]:


#! ipython id=5b9675723bc546dd8e5494a63479838e
a, b = 1, 2
a
b
b, a = a, b
a
b


# In[14]:


#! ipython id=5db070301a5e462588ea2f6e56491b16
seq = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
for a, b, c in seq:
    print(f'a={a}, b={b}, c={c}')


# In[15]:


#! ipython id=2150bcef09da49b79898e3131b9d55aa
values = 1, 2, 3, 4, 5
a, b, *rest = values
a
b
rest


# In[16]:


#! ipython id=88673596fc474ab7aad7f3c53f5d4c43
a, b, *_ = values


# In[17]:


#! ipython id=08adf09ff6f04bfaa24519224688103f
a = (1, 2, 2, 2, 3, 4, 2)
a.count(2)


# In[18]:


#! ipython id=1e4a94249d2b454e9eab4c5c5668ca53
a_list = [2, 3, 7, None]

tup = ("foo", "bar", "baz")
b_list = list(tup)
b_list
b_list[1] = "peekaboo"
b_list


# In[19]:


#! ipython id=318993071ad64015b2ff7e2cc2471b1e
gen = range(10)
gen
list(gen)


# In[20]:


#! ipython id=b8bf8fab6a7945cba118ded7c5832d7d
b_list.append("dwarf")
b_list


# In[21]:


#! ipython id=c37cb14bfdc347f7a875d7f2d792a5e8
b_list.insert(1, "red")
b_list


# In[22]:


#! ipython id=03999831e06b4c5680299112f55070d5
b_list.pop(2)
b_list


# In[23]:


#! ipython id=db5248fa20ab409ab2bd27b150a674a4
b_list.append("foo")
b_list
b_list.remove("foo")
b_list


# In[24]:


#! ipython id=36396876129145c7a39e128c5782b07d
"dwarf" in b_list


# In[25]:


#! ipython id=c5e00b324c1f43b084aa5b3005c6383f
"dwarf" not in b_list


# In[26]:


#! ipython id=a7057ca531ac416d8794605b19e63190
[4, None, "foo"] + [7, 8, (2, 3)]


# In[27]:


#! ipython id=e1f5254102104c92b988c358b6344319
x = [4, None, "foo"]
x.extend([7, 8, (2, 3)])
x


# In[28]:


#! ipython id=f7dc01342ead4647bd025c02160ace46
a = [7, 2, 5, 1, 3]
a.sort()
a


# In[29]:


#! ipython id=5805c218da894959836bca33cae741b7
b = ["saw", "small", "He", "foxes", "six"]
b.sort(key=len)
b


# In[30]:


#! ipython id=56f8cc98455b49838595dfade4ab90d1
seq = [7, 2, 3, 7, 5, 6, 0, 1]
seq[1:5]


# In[31]:


#! ipython id=df2909e780bc4cf0b48272e57a7fc69f
seq[3:5] = [6, 3]
seq


# In[32]:


#! ipython id=2785d920b1d047ba8ed411fcfd8a7bf1
seq[:5]
seq[3:]


# In[33]:


#! ipython id=0ddaf435400e4bee8241f3785e7cdd74
seq[-4:]
seq[-6:-2]


# In[34]:


#! ipython id=099a7a8977f747ccb839bf11eccc4f11
seq[::2]


# In[35]:


#! ipython id=74c7a1f7938f4e029799a354247bab44
seq[::-1]


# In[36]:


#! ipython id=102e43ef794d434e9273444cc7874157
empty_dict = {}
d1 = {"a": "some value", "b": [1, 2, 3, 4]}
d1


# In[37]:


#! ipython id=206c17b9ac344ecbbe709abb43a4d4c5
d1[7] = "an integer"
d1
d1["b"]


# In[38]:


#! ipython id=c3d1090a30354b9b8cc4ba5909661aa2
"b" in d1


# In[39]:


#! ipython id=525e7836241848f783e5b25b66ff0627
d1[5] = "some value"
d1
d1["dummy"] = "another value"
d1
del d1[5]
d1
ret = d1.pop("dummy")
ret
d1


# In[40]:


#! ipython id=17b5a6e9001d47a0b604e547ee929a5e
list(d1.keys())
list(d1.values())


# In[41]:


#! ipython id=d711dee217894970ab40d16b495a1418
list(d1.items())


# In[42]:


#! ipython id=2950abfb881b43d4b0d0b4ee9866e283
d1.update({"b": "foo", "c": 12})
d1


# In[43]:


#! ipython id=8adf5329762c41d1b80f3d7be30162ae
tuples = zip(range(5), reversed(range(5)))
tuples
mapping = dict(tuples)
mapping


# In[44]:


#! ipython id=8bdf9587c72d4e7f8d332558a707ad6b
words = ["apple", "bat", "bar", "atom", "book"]
by_letter = {}

#! blockstart
for word in words:
    letter = word[0]
    if letter not in by_letter:
        by_letter[letter] = [word]
    else:
        by_letter[letter].append(word)
#! blockend

by_letter


# In[45]:


#! ipython id=d541b1633e0a4a9aaa3a75f4487c293c
by_letter = {}
#! blockstart
for word in words:
    letter = word[0]
    by_letter.setdefault(letter, []).append(word)
#! blockend
by_letter


# In[46]:


#! ipython id=83920435a755492fa706ecec0b394d4a
from collections import defaultdict
by_letter = defaultdict(list)
for word in words:
    by_letter[word[0]].append(word)


# In[47]:


#! ipython allow_exceptions id=ef6987776ec0412ebea75e5dabf8f405
hash("string")
hash((1, 2, (2, 3)))
hash((1, 2, [2, 3])) # fails because lists are mutable


# In[48]:


#! ipython id=6537602c2b2749f092d5289a226a30fd
d = {}
d[tuple([1, 2, 3])] = 5
d


# In[49]:


#! ipython id=39fbedd72ba046688ab288378af6d8ab
set([2, 2, 2, 1, 3, 3])
{2, 2, 2, 1, 3, 3}


# In[50]:


#! ipython id=960a9862306d45b9bebf33933303f16b
a = {1, 2, 3, 4, 5}
b = {3, 4, 5, 6, 7, 8}


# In[51]:


#! ipython id=930360aa3e46418fbfc9a8398a0b64a3
a.union(b)
a | b


# In[52]:


#! ipython id=d2fee1cd0d2b4f33a88964578841a586
a.intersection(b)
a & b


# In[53]:


#! ipython id=bc93c7d7808448908a39e8c48516f8ea
c = a.copy()
c |= b
c
d = a.copy()
d &= b
d


# In[54]:


#! ipython id=d3ba3bbecde44f16931f20b7613400ed
my_data = [1, 2, 3, 4]
my_set = {tuple(my_data)}
my_set


# In[55]:


#! ipython id=69fdfc8010a64cb9b5bd227afae109a1
a_set = {1, 2, 3, 4, 5}
{1, 2, 3}.issubset(a_set)
a_set.issuperset({1, 2, 3})


# In[56]:


#! ipython id=4895c04ae7fa49cda3a760d5dd952a24
{1, 2, 3} == {3, 2, 1}


# In[57]:


#! ipython id=2adc3677205342eca43882c88ecc673e
sorted([7, 1, 2, 6, 0, 3, 2])
sorted("horse race")


# In[58]:


#! ipython id=9ebb445902b34f8a9e4f30c3b2492235
seq1 = ["foo", "bar", "baz"]
seq2 = ["one", "two", "three"]
zipped = zip(seq1, seq2)
list(zipped)


# In[59]:


#! ipython id=6cb65a648c764d428fab7b7f0e3088e2
seq3 = [False, True]
list(zip(seq1, seq2, seq3))


# In[60]:


#! ipython id=33b017981136408db89261408551d481
#! blockstart
for index, (a, b) in enumerate(zip(seq1, seq2)):
    print(f"{index}: {a}, {b}")
#! blockend


# In[61]:


#! ipython id=ba42b4701d824e03b5798c66a4e8ba52
list(reversed(range(10)))


# In[62]:


#! ipython id=4a46030633924bcf9ed372378866d5e5
strings = ["a", "as", "bat", "car", "dove", "python"]
[x.upper() for x in strings if len(x) > 2]


# In[63]:


#! ipython id=27c87458145e4e74b262c8d4eec99e5c
unique_lengths = {len(x) for x in strings}
unique_lengths


# In[64]:


#! ipython id=a612eb89648142969dd70a25f2329dc7
set(map(len, strings))


# In[65]:


#! ipython id=fcbefbee024e496eb46392a631375d9f
loc_mapping = {value: index for index, value in enumerate(strings)}
loc_mapping


# In[66]:


#! ipython id=018be34f69874e1e9871f87d3d02c69b
all_data = [["John", "Emily", "Michael", "Mary", "Steven"],
            ["Maria", "Juan", "Javier", "Natalia", "Pilar"]]


# In[67]:


#! ipython id=E829182B11D647519558FD46F48626A9
names_of_interest = []
#! blockstart
for names in all_data:
    enough_as = [name for name in names if name.count("a") >= 2]
    names_of_interest.extend(enough_as)
#! blockend
names_of_interest


# In[68]:


#! ipython id=d3c630997f40472db3bafccfb746f819
result = [name for names in all_data for name in names
          if name.count("a") >= 2]
result


# In[69]:


#! ipython id=bd50aef619e1481ab2451c8f86ce3d42
some_tuples = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
flattened = [x for tup in some_tuples for x in tup]
flattened


# In[70]:


#! ipython verbatim id=7E5891774F464818BCBCB3C2692682A6
flattened = []

for tup in some_tuples:
    for x in tup:
        flattened.append(x)


# In[71]:


#! ipython id=db1e87dd90fb4645820ff22fb0ca1379
[[x for x in tup] for tup in some_tuples]


# In[72]:


#! ipython id=4673A12D54CE49829790310977A675A5
def my_function(x, y):
    return x + y


# In[73]:


#! ipython id=E4F308DD00ED486B93AB620C2AC256B5
my_function(1, 2)
result = my_function(1, 2)
result


# In[74]:


#! ipython id=1C8501E576A0458F98DD8EB0F59E8933
def function_without_return(x):
    print(x)

result = function_without_return("hello!")
print(result)


# In[75]:


#! ipython verbatim id=4673A12D54CE49829790310977A675A5
def my_function2(x, y, z=1.5):
    if z > 1:
        return z * (x + y)
    else:
        return z / (x + y)


# In[76]:


#! ipython id=729E08FB177541928A87326230C7868A
my_function2(5, 6, z=0.7)
my_function2(3.14, 7, 3.5)
my_function2(10, 20)


# In[77]:


#! ipython id=988c7d961f684af09d7d9a9daa7c2bb9
a = []
def func():
    for i in range(5):
        a.append(i)


# In[78]:


#! ipython id=b1fef1a73b914b2eb8effdd4df727d25
func()
a
func()
a


# In[79]:


#! ipython id=b5d2338ec6b64bdf9b50d4674b32f14f
a = None
#! blockstart
def bind_a_variable():
    global a
    a = []
bind_a_variable()
#! blockend
print(a)


# In[80]:


#! ipython id=022108f1e242478da353a3d86708ed47
states = ["   Alabama ", "Georgia!", "Georgia", "georgia", "FlOrIda",
          "south   carolina##", "West virginia?"]


# In[81]:


#! ipython verbatim id=fb7c1a7bea534d85ba3b735097ad0235
import re

def clean_strings(strings):
    result = []
    for value in strings:
        value = value.strip()
        value = re.sub("[!#?]", "", value)
        value = value.title()
        result.append(value)
    return result


# In[82]:


#! ipython id=7ea42274bc2947f3b4ebe209f9c7ee82
clean_strings(states)


# In[83]:


#! ipython verbatim id=ce250f90f14e4c98ad8b8d108fa98bba
def remove_punctuation(value):
    return re.sub("[!#?]", "", value)

clean_ops = [str.strip, remove_punctuation, str.title]

def clean_strings(strings, ops):
    result = []
    for value in strings:
        for func in ops:
            value = func(value)
        result.append(value)
    return result


# In[84]:


#! ipython id=e7f5b8fba1784d33833db3ee759a7e89
clean_strings(states, clean_ops)


# In[85]:


#! ipython id=68ccca27ff884667a981d5843c3b370a
for x in map(remove_punctuation, states):
    print(x)


# In[86]:


#! ipython id=6995cf9cd38c4851b47bea0108ff374d
def short_function(x):
    return x * 2

equiv_anon = lambda x: x * 2


# In[87]:


#! ipython id=c5fb98a7a6d7427085a4fc16d1aeaf8a
def apply_to_list(some_list, f):
    return [f(x) for x in some_list]

ints = [4, 0, 1, 5, 6]
apply_to_list(ints, lambda x: x * 2)


# In[88]:


#! ipython id=3c96075c117b4d29a8b0de3741b66f03
strings = ["foo", "card", "bar", "aaaa", "abab"]


# In[89]:


#! ipython id=35dc027b565e41be8c24a69bd348830e
strings.sort(key=lambda x: len(set(x)))
strings


# In[90]:


#! ipython id=ef32fe84f029417092e95b48dff21ee3
some_dict = {"a": 1, "b": 2, "c": 3}
for key in some_dict:
    print(key)


# In[91]:


#! ipython id=e21664c09fd74763b5bd9fd1c9b58bb2
dict_iterator = iter(some_dict)
dict_iterator


# In[92]:


#! ipython id=295f6871ebff40a9a1113898043d6fbc
list(dict_iterator)


# In[93]:


#! ipython verbatim id=0cee7d7909eb4073b8c6424f1808a668
def squares(n=10):
    print(f"Generating squares from 1 to {n ** 2}")
    for i in range(1, n + 1):
        yield i ** 2


# In[94]:


#! ipython id=772dbe0e7ccf428c8dc574bb2909af24
gen = squares()
gen


# In[95]:


#! ipython id=6498f8f9727f48598248e2c51ae7b280
for x in gen:
    print(x, end=" ")


# In[96]:


#! ipython id=27d6215331ba4508a6660521a46fda25
gen = (x ** 2 for x in range(100))
gen


# In[97]:


#! ipython id=8eeac2424de84f5c8e7d21dbbfb02566
sum(x ** 2 for x in range(100))
dict((i, i ** 2) for i in range(5))


# In[98]:


#! ipython id=71edd1b5622e4acbbe73dfe2ba077f2d
import itertools
def first_letter(x):
    return x[0]

names = ["Alan", "Adam", "Wes", "Will", "Albert", "Steven"]

for letter, names in itertools.groupby(names, first_letter):
    print(letter, list(names)) # names is a generator


# In[99]:


#! ipython allow_exceptions id=9bd5bd1b77e04bb9a199a632e657a587
float("1.2345")
float("something")


# In[100]:


#! ipython verbatim id=7324afff252e43d88e9f87c0668273e2
def attempt_float(x):
    try:
        return float(x)
    except:
        return x


# In[101]:


#! ipython id=761d1de6e4c24a9981e15516315ed406
attempt_float("1.2345")
attempt_float("something")


# In[102]:


#! ipython allow_exceptions id=ba156be511f3418792b8abb5fa790d01
float((1, 2))


# In[103]:


#! ipython verbatim id=599fbddb78b2439883ea0cb3d2fe7400
def attempt_float(x):
    try:
        return float(x)
    except ValueError:
        return x


# In[104]:


#! ipython allow_exceptions id=7e3e9049b61d416f9e411378e7043e2c
attempt_float((1, 2))


# In[105]:


#! ipython verbatim id=9a2f8c5351814558b1019f4f42d07d2e
def attempt_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return x


# In[106]:


#! ipython id=b4981a31b1a340448473e18ed3db5ccf
path = "examples/segismundo.txt"
f = open(path, encoding="utf-8")


# In[107]:


#! ipython id=3c4cc539ca7f465a8c40c73ac17e4e5f
lines = [x.rstrip() for x in open(path, encoding="utf-8")]
lines


# In[108]:


#! ipython id=4b5aaa74913f4d41a264eca0d626c465
f.close()


# In[109]:


#! ipython id=174c699a268447c4ba28d47a99402980
with open(path, encoding="utf-8") as f:
    lines = [x.rstrip() for x in f]


# In[110]:


#! ipython id=d4529c243e7c49418ea310270a60f524
f1 = open(path)
f1.read(10)
f2 = open(path, mode="rb")  # Binary mode
f2.read(10)


# In[111]:


#! ipython id=a8f60ea5015849598ecd2bbd87a75174
f1.tell()
f2.tell()


# In[112]:


#! ipython id=1b25166af8444a1d939b4184bf8f909b
import sys
sys.getdefaultencoding()


# In[113]:


#! ipython id=91c150712de941cdb037bff70bf85fe4
f1.seek(3)
f1.read(1)
f1.tell()


# In[114]:


#! ipython id=0551307ffb834d36bbc23c1c26bb8e5f
f1.close()
f2.close()


# In[115]:


#! ipython id=e1abf2e446664da7ab1e1234a37603b7
path

with open("tmp.txt", mode="w") as handle:
    handle.writelines(x for x in open(path) if len(x) > 1)

with open("tmp.txt") as f:
    lines = f.readlines()

lines


# In[116]:


#! ipython suppress id=7d7eec59e35d41a481366e32072e964c
import os
os.remove("tmp.txt")


# In[117]:


#! ipython id=5b95866731ae4b87b79b9dddc95622d9
with open(path) as f:
    chars = f.read(10)

chars
len(chars)


# In[118]:


#! ipython id=e85aea11b3b5478f8f3e9b0b7d118c7a
with open(path, mode="rb") as f:
    data = f.read(10)

data


# In[119]:


#! ipython allow_exceptions id=c5e296d66bab43e5bf74f097c5650f8d
data.decode("utf-8")
data[:4].decode("utf-8")


# In[120]:


#! ipython id=cb101430288e455dad4024b59b17aee1
sink_path = "sink.txt"
with open(path) as source:
    with open(sink_path, "x", encoding="iso-8859-1") as sink:
        sink.write(source.read())

with open(sink_path, encoding="iso-8859-1") as f:
    print(f.read(10))


# In[121]:


#! ipython suppress id=691ae68a0bd840d4b8d757af71ca26ca
os.remove(sink_path)


# In[122]:


#! ipython allow_exceptions id=e915449c2d9e4f0b8f639a2cb1d20552
f = open(path, encoding='utf-8')
f.read(5)
f.seek(4)
f.read(1)
f.close()


# In[123]:


#! ipython suppress id=9ccad96340cf4d5ebf20ee72f884f64f
get_ipython().run_line_magic('popd', '')

