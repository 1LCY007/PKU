# Cheatsheet

##### Users: 骆春阳 工学院



## 字典

#### 一些语法

##### e.g.打怪兽

![image-20240227162909816](C:\Users\骆春阳\AppData\Roaming\Typora\typora-user-images\image-20240227162909816.png)

1. ```Python
   n = int(input())
   count1 = 0
   result = []
   while n > count1:
       sums,amount,hurt = map(int,input().split())
       dic = {}
       J = []
       for _ in range(sums):
           ti,xi = map(int,input().split())
           dic.setdefault(ti,[]).append(xi)  ##setdefault
           if ti not in J:
               J.append(ti)
       def judge(dic,amount,hurt,J):
           m = 0
           J.sort()
           for _ in sorted(dic.keys()): ##将字典按keys排序后
                   lst = dic.get(_)
                   lst.sort(reverse = True)
                   if len(lst) <= amount:
                       hurt = hurt - sum(lst)
                   else:
                       hurt -= sum(lst[:amount])  ##把第一个时刻能放的技能都放掉
                   if hurt <= 0:
                       return J[m]
                   m += 1
           return 'alive'
       result.append(judge(dic,amount,hurt,J))
       count1 += 1
   for i in result:
       print(i)
   ```

== setdefault的用法

== 用get获取相应key的值

== 按key排序后顺序进行输出

#### defaultdict包

```python
from collections import defaultdict
for _ in range(int(input())):
    a=defaultdict(list)
	n,m,b=map(int,input().split())
    for _ in range(n):
        t,x=map(int,input().split())
        a[t].append (x)
    for t in sorted(a):
        b-=sum(sorted(a[tl,reverse-True)[:m])
        if b<=0:
            print(t)
            break
	else:
		print( alive’)
```



## 类：

```python
class Dog:
    def _init_(self,name):
        self.name = name
    def sit(self):
        print('down')
```



## 筛法

### 线性筛（欧拉筛法）



```python
def euler(r):
    prime = [0 for i in range(r+1)]
    common = []
    for i in range(2, r+1):
        if prime[i] == 0:
            common.append(i)
        for j in common:
            if i*j > r:
                break
            prime[i*j] = 1
            if i % j == 0:
                break
    return common ##common就是质数表
```

#### 埃氏筛法

```python
def judge(number):
    nlist = list(range(1,number+1))
    nlist[0] = 0
    k = 2
    while k * k <= number:
        if nlist[k-1] != 0:
            for i in range(2*k,number+1,k):
                nlist[i-1] = 0
        k += 1
    result = []
    for num in nlist:
        if num != 0:
            result.append(num)
    return result
```



## 细节问题

### 1.关于输入输出

#### 1.

原代码

```python
lst = map(int,input().split())
result = []
for i in lst:
	##对lst中每个元素进行操作
	result.append(i) ##加入到答案表
for _ in result:
print(_)  ##遍历输出
```

改进后的代码

```python
for i in map(int,input().split()):
    ##对元素进行操作
    print(i)
```

#### 2.关于将表中元素连续输出

```python
lst = [1,2,3]
print(*[lst])
## 1 2 3
```





### 2.时间复杂度的优化

1. 用set（）比用list更好

   ==用list会很慢
   
   ==set中的元素无序排列

### 3.细节语法

#### 1.列表：

用extend可以将另一个表的所有元素添加到另一张列表的末尾。

primes.extend(extras)





#### 2.eval

eval() 是 python 中功能非常强大的一个函数
将字符串当成有效的表达式来求值，并返回计算结果
所谓表达式就是：eval 这个函数会把里面的字符串参数的引号去掉，把中间的内容当成Python的代码，eval 函数会执行这段代码并且返回执行结果
也可以这样来理解：eval() 函数就是实现 list、dict、tuple、与str 之间的转化
————————————————

```python
result = eval("1 + 1")
print(result)  # 2   

result = eval("'+' * 5")
print(result)  # +++++

# 3. 将字符串转换成列表
a = "[1, 2, 3, 4]"
result = type(eval(a))
print(result)  # <class 'list'>

input_number = input("请输入一个加减乘除运算公式：")
print(eval(input_number))
## 1*2 +3
## 5
```



原文链接：https://blog.csdn.net/qq_46450354/article/details/127183649

#### 3.字符串

字符串处理 

str.title()首字母大写（每个单词） str.lower()/upper(）每个字母小/大写 str.strip()去除空格，有相应 的rstrip/lstrip去掉尾部/头部的空格 在格式化字符串时可能有用（查到strip也可以带参数，去掉指定的 某些字符，但是没用过） 空格处理有时候很麻烦（OJ04030 统计单词数） 字符串是不可变对象！可以取索引、切片、连接（“+”），但是不能修改。如果需要修改用列表会更好 （不然就得每次弄一个新的字符串，浪费内存）

ord() chr() 可以完成字符与ASCII码的转化，有些时候（比如要把字母和其在字母表内的顺序对应时）有 用（但是我很少用）

str.find()查找指定字符，注意如果有的话会返回第一个找到的，如果没有会返回-1而不是报错！（这些 特性有时候很好用但是另一些时候可能导致错误）

str.zfill()自动在前面补0补到所需位数



==!!!注意字符串比较大小是按个字符串比较，所以12比3小==

==要比较大小的时候记得注意换成浮点数或者整数==



#### 4.val方法

将字符串中的数字提取出来，如果没有数字就会报错

```python
x = '123abc'
y = val(x)
print(y)
##输出 ‘123’
```



## Python包

### 1.math

有gcd包，求最大公因式

== float('inf)' 表示正无穷

==可使用atof(str)把字符串转换为一个double类型的浮点数。atof定义在math.h中

### 2.bisect

```python
from bisect import *
```

```python
import bisect ##导入bisect包

##bisect是一个排序模块，操作对象必须为排好序的列表。
##bisect操作并不改变列表中的元素，仅仅是确认插入元素的位置
##与之对应的insort
lst = [1,3,5,7,9]
s = int(input())
bisect.bisect_left(lst, x, [lo=0, hi=len(a)])   ##[]中表示插入位置的上界和下届
##改成right同理

## 测试序列 a2
>>> a2 = [1, 3, 3, 4, 7]  # 元素从小到大排列，有重复, 不等距
 
# 限定查找范围：[lo=1, hi=3] 
>>> bisect.bisect_left(a2, 0, 1, 3)  # 与 x=0 右侧最近的元素是 1, 其位置 index=0, 但下限 lo=1, 故只能返回位置 index=1
1
>>> bisect.bisect_left(a2, 1, 1, 3)  # x=1 的位置 index=0, 但下限 lo=1, 故只能返回位置 index=1
1
>>> bisect.bisect_left(a2, 2, 1, 3)  # 与 x=2 右侧最近的元素是 3, 其位置 index=1
1
>>> bisect.bisect_left(a2, 3, 1, 3)  # 第一个(最左侧) x=3 的位置 index=1 
1
>>> bisect.bisect_left(a2, 4, 1, 3)  # x=4 的位置 index=3
3
>>> bisect.bisect_left(a2, 5, 1, 3)  # 与 x=5 右侧最近的元素是 7, 其位置 index=4, 但上限 hi=3, 故只能返回位置 index=3 
3
>>> bisect.bisect_left(a2, 6, 1, 3)  # 与 x=6 右侧最近的元素是 7, 其位置 index=4, 但上限 hi=3, 故只能返回位置 index=3 
3
>>> bisect.bisect_left(a2, 7, 1, 3)  # x=7 的位置 index=4, 但上限 hi=3, 故只能返回位置 index=3 
3
>>> bisect.bisect_left(a2, 8, 1, 3)  # 上限 hi=3
3


```



```python
##如果说 bisect.bisect_left() 是为了在序列 a 中 查找 元素 x 的插入点 (左侧)，那么 bisect.insort_left() 就是在找到插入点的基础上，真正地将元素 x 插入序列 a，从而改变序列 a 同时保持元素顺序。
>>> a11 = [5, 6, 7, 8, 9]
>>> bisect.insort_left(a11, 4.5)
>>> a11
[4.5, 5, 6, 7, 8, 9]
 
>>> a12 = [5, 6, 7, 8, 9]
>>> bisect.insort_left(a12, 5.5)
>>> a12
[5, 5.5, 6, 7, 8, 9]
 
>>> a13 = [5, 6, 7, 8, 9]
>>> bisect.insort_left(a13, 6.5)
>>> a13
[5, 6, 6.5, 7, 8, 9]
 
>>> a14 = [5, 6, 7, 8, 9]
>>> bisect.insort_left(a14, 7.5)
>>> a14
[5, 6, 7, 7.5, 8, 9]
 
>>> a15 = [5, 6, 7, 8, 9]
>>> bisect.insort_left(a15, 8.5)
>>> a15
[5, 6, 7, 8, 8.5, 9]
 
>>> a16 = [5, 6, 7, 8, 9]
>>> bisect.insort_left(a16, 9.5)
>>> a16
[5, 6, 7, 8, 9, 9.5]

```







## 样题：

### 1.波兰表达式

```pyhton
s = input().split()
def cal():
    cur = s.pop(0)
    if cur in "+-*/":
        return str(eval(cal() + cur + cal()))
    else:
        return cur
print("%.6f" % float(cal()))
```



### 2.最大上升子序列

```python
input()
b = [int(x) for x in input().split()]

n = len(b)
dp = [0]*n

for i in range(n):
    dp[i] = b[i]
    for j in range(i):
        if b[j]<b[i]:
            dp[i] = max(dp[j]+b[i], dp[i])
    
print(max(dp))
```



