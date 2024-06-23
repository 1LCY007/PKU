# Assignment #2: 编程练习 

Updated 0953 GMT+8 Feb 24, 2024

 2024 spring, Complied by **骆春阳 工学院**



#### 编程环境

操作系统：Windows11

Python编程环境：PyCharm 2023.1.4 (Professional Edition)



## 1.题⽬ 

### 27653: Fraction类 

http://cs101.openjudge.cn/2024sp_routine/27653/



#### 思路：

用埃氏筛法加一个遍历除法

#### 源代码：

```Python
a1,a2,b1,b2 = map(int,input().split())
c1 = a1 * b2 + a2 * b1
c2 = a2 * b2
if c1 % c2 == 0:
    print(c1 / c2)
    exit()
else:
    def judge(number):
        if number < 2:
            return []
        nlist = list(range(1, number + 1))
        nlist[0] = 0
        k = 2
        while k * k <= number:
            if nlist[k - 1] != 0:
                for i in range(2 * k, number + 1, k):
                    nlist[i - 1] = 0
            k += 1
        result = []
        for x in nlist:
            if x != 0:
                result.append(x)
        return result
    for i in judge(min(c1,c2)):
        while c1 % i == 0 and c2 % i == 0:
            c1 = c1 / i
            c2 = c2 / i
    c1 = int(c1)
    c2 = int(c2)
    print(str(c1)+'/'+str(c2))
```

![image-20240227142254460](C:\Users\骆春阳\AppData\Roaming\Typora\typora-user-images\image-20240227142254460.png)

#### 时间：

七八分钟左右



## 04110: 圣诞⽼⼈的礼物-Santa Clau’s Gifts 

greedy/dp, http://cs101.openjudge.cn/practice/04110



#### 思路：

简单滴贪心算法~

#### 源代码：

```python
n,weight = map(int,input().split())
count1 = 0
judge = []
add = 0
while count1 < n:
    v,w = map(int,input().split())
    add += w
    judge.append([v,w,v/w])
    count1 += 1
judge.sort(key = lambda x:x[-1])
result = 0
for i in judge[::-1]:
    if weight >= i[1]:
        result += i[0]
        weight -= i[1]
        continue
    else:
        result += weight * i[0] / i[1]
        break
print(f'{result:.1f}')

```

![image-20240227143921389](C:\Users\骆春阳\AppData\Roaming\Typora\typora-user-images\image-20240227143921389.png)

#### 时间：

三十分钟左右（边听讲座边摸鱼写的）



### 18182: 打怪兽 

implementation/sortings/data structures, http://cs101.openjudge.cn/practice/18182/

#### 思路：

刚开始用t循环超时了，后来发现字典排序输出后就会好很多

#### 源代码：

```python
n = int(input())
count1 = 0
result = []
while n > count1:
    sums,amount,hurt = map(int,input().split())
    dic = {}
    J = []
    for _ in range(sums):
        ti,xi = map(int,input().split())
        dic.setdefault(ti,[]).append(xi)
        if ti not in J:
            J.append(ti)
    def judge(dic,amount,hurt,J):
        m = 0
        J.sort()
        for _ in sorted(dic.keys()):
                lst = dic.get(_)
                lst.sort(reverse = True)
                if len(lst) <= amount:
                    hurt = hurt - sum(lst)
                else:
                    hurt -= sum(lst[:amount])##把第一个时刻能放的技能都放掉
                if hurt <= 0:
                    return J[m]
                m += 1
        return 'alive'
    result.append(judge(dic,amount,hurt,J))
    count1 += 1
for i in result:
    print(i)
```

![image-20240227144049108](C:\Users\骆春阳\AppData\Roaming\Typora\typora-user-images\image-20240227144049108.png)

#### 时间：

40分钟

### 230B. T-primes 

binary search/implementation/math/number theory, 1300, http://codeforces.com/problemset/problem/230/B

#### 思路：

思路清晰，现实很骨感。

用埃氏筛法进行摆弄，却反复的超时/超空间，实在弄不出来了，看了答案反复学习了一下收获颇丰。

首先，如果用欧式筛法的话，时间复杂度就会从nlogn变成n。

本来想就此放弃，直接抄答案的

后来看到答案里也有用埃氏筛法通过的，于是决定看看自己的哪里能简化。

第一就是将埃氏筛法改变了一下，让它适用于这个题目，直接得出一个质数平方和的表

第二就是在输入输出方面的省略，原先我将输出放到一个list里，遍历，然后再把答案放到另一个list,再遍历，增加了很多复杂度，现在发现for循环可以直接套map

第三就是第一条中，发现质数平方和的表也会超时，后来在群里讨论发现，set和list的时间差真的是太大了。

最后成品就是这个咯，也是终于ac了

#### 源代码：

```python
n = int(input())
result = set()
number = 1000000
nlist = list(range(1, number + 1))
nlist[0] = 0
k = 2
while k <= number:
    if nlist[k - 1] != 0:
        result.add(nlist[k - 1] ** 2)
        for i in range(2 * k, number + 1, k):
            nlist[i - 1] = 0
    k += 1
for i in map(int,input().split()):
    if i in result:
        print('YES')
    else:
        print('NO')

```

![image-20240227160832600](C:\Users\骆春阳\AppData\Roaming\Typora\typora-user-images\image-20240227160832600.png)

#### 时间：

未知（研究了太久!!!）





### 1364A. XXXXX

brute force/data structures/number theory/two pointers, 1200, https://codeforces.com/problemset/problem/1364/A

#### 思路：

先sum一次表检查一下

两个指针向表的中间收缩，直到碰到一个不能整除x的数即可。

#### 源代码：

```python
def judge(lst,m,nums):
    k = 0
    j = m
    if sum(lst) % nums != 0:
        return m
    while j >= k:
        if lst[k] % nums != 0:
            return m - k-1
        if lst[j-1] % nums != 0:
            return j-1
        k += 1
        j -= 1
    return -1
for i in range(int(input())):
    m,nums = map(int,input().split())
    lst = list(map(int,input().split()))
    print(judge(lst,m,nums))
```

![image-20240229163626218](C:\Users\骆春阳\AppData\Roaming\Typora\typora-user-images\image-20240229163626218.png)

#### 时间：

30分钟

（刚开始一直在sum，后知后觉才反应过来不用这么麻烦）

### 18176: 2050年成绩计算 

http://cs101.openjudge.cn/practice/18176/

#### 思路：

思路，发现时间限制比较严格，于是现将质数表印了出来再进行查找

#### 源代码：

```python
n,m = map(int,input().split())
##先创建一个欧式筛法的函数
number = 10001
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
result = set(judge(number))
count1 = 0
L = []
while count1 < n:
    lst = list(map(int,input().split()))
    add = 0
    l = len(lst)
    for i in lst:
        if i ** (0.5) == int(i ** (0.5)) and i ** (0.5) in result:
            add += i
    if add == 0:
        L.append('0')
    else:
        L.append(f'{add / l:.2f}')
    count1 += 1
for i in L:
    print(i)
```

![image-20240227143944121](C:\Users\骆春阳\AppData\Roaming\Typora\typora-user-images\image-20240227143944121.png)

#### 时间：

四十分钟

## 2.学习总结和收获

太爽了，尤其是t-prime一题，让我进一步在输入输出和遍历上优化了自己的程序结构，使程序更加精简。（同时惊讶于原来set比list能省很多时间）看了一遍t-prime的答案后，还学到了一个欧式筛法。

哦还有发现了一下自己的问题，有时候自己想问题的算法比较简陋，都是笨方法，应该继续学习如何简化程序，当然主要还是在思考问题的方式上有待改善。



