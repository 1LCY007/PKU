# Assignment\#1: 拉⻬⼤家Python⽔平

Updated 0940 GMT+8 Feb 19, 2024 

2024 spring, Complied by **骆春阳 工学院**

#### 编程环境：

PyCharm 2023.1.4 (Professional Edition)

操作系统Windows11



## 1.题⽬ 

### 20742: 泰波拿契數 

http://cs101.openjudge.cn/practice/20742/

 

#### 思路:

采用递归,并用记忆性递归来优化时间结构

#### 源代码:

```python
n =int(input())
if n <= 2:
    print(1)
    exit()
else:
    lst = [-1 for i in range(n + 1)]
    lst[0] = 0
    lst[1] = 1
    lst[2] = 1
    def count(n,lst):
        if lst[n] == -1:
            lst[n] = count(n-1,lst) + count(n-2,lst) + count(n-3,lst)
        if lst[n] != -1:
            return lst[n]
        return count(n-1,lst) + count(n-2,lst) + count(n-3,lst)
    print(count(n,lst))

```

![](C:\Users\骆春阳\AppData\Roaming\Typora\typora-user-images\image-20240227133958747.png)

#### 时间：

大约八分钟

 

### 58A. Chat room 

greedy/strings, 1000, http://codeforces.com/problemset/problem/58/A

 

#### 思路：

用一个对照的表当做答案，之后将输入的字符串遍历。

 

#### 源代码:

 ```python
 s = input()
 answer = ['h','e','l','l','o']
 for i in s:
     if i == answer[0]:
         answer.pop(0)
     if answer == []:
         print('YES')
         exit()
 print('NO')
 
 ```

![image-20240227134244684](C:\Users\骆春阳\AppData\Roaming\Typora\typora-user-images\image-20240227134244684.png)

#### 时间：

大约三分钟



### 118A. String Task 

implementation/strings, 1000, http://codeforces.com/problemset/problem/118/A

#### 思路：

用一个给定的字符串当做对照，来遍历输入的字符串并进行相关操作。

#### 源代码：

```python
1.	s = input()
2.	s = s.lower()
3.	judge = 'aeiouy'
4.	answer = ''
5.	for i in s:
6.	    if i not in judge:
7.	        answer = answer + '.' + i
8.	    else:
9.	        pass
10.	print(answer)

```



#### ![image-20240227134826065](C:\Users\骆春阳\AppData\Roaming\Typora\typora-user-images\image-20240227134826065.png)

#### 时间：

三分钟



### 22359: Goldbach Conjecture 

http://cs101.openjudge.cn/practice/22359/ 

#### 思路：

先找到所有质数，然后遍历

#### 源代码：

```python
from math import sqrt
n = int(input())
def judge(number):
    if number < 2:
        return []
    nlist = list(range(1,number+1))
    nlist[0] = 0
    k = 2
    while k * k <= number:
        if nlist[k-1] != 0:
            for i in range(2 * k,number + 1,k):
                nlist[i-1] = 0
        k += 1
    result = []
    for x in nlist:
        if x != 0:
            result.append(x)
    return result
lst = judge(n)
for i in lst:
    if n - i in lst:
        print(i,n-i)
        exit()
```

![image-20240227134813865](C:\Users\骆春阳\AppData\Roaming\Typora\typora-user-images\image-20240227134813865.png)

#### 时间：

大约二十分钟（埃氏筛法忘了复习了一下）



### 23563: 多项式时间复杂度 

http://cs101.openjudge.cn/practice/23563/



#### 思路：

刚开始一直没想好怎么把中间的数字提出来，后来才决定用分割解决问题

#### 源代码：

```python
s = input()
s = s
result = []
lst = list(s.split('+'))
for i in lst:
    if i != '':
        if i[0] != '0':
            lst2 = list(i.split('^'))
            num = int(lst2[-1])
            result.append(num)
m = str(max(result))
print(f'n^{m}')

```

![image-20240227134941890](C:\Users\骆春阳\AppData\Roaming\Typora\typora-user-images\image-20240227134941890.png)

#### 时间：

 三十分钟（事实上，当我想到用分割解决问题之后，用了五分钟就OK了www)



### 24684: 直播计票 

http://cs101.openjudge.cn/practice/24684/



#### 思路：

刚开始想的比较乱，然后后来决定用五个表来解决。。。

#### 源代码：

```python
lst = list(map(int,input().split()))
lst2 = []
lst3 = []
result = []
for num in lst:
    if num not in lst2:
        lst2.append(num)
        a = lst.count(num)
        lst3.append(a)
        result.append([num,a])
M = max(lst3)
l = []
for i in result:
    if M == i[-1]:
        l.append(i[0])
for i in sorted(l)[:-1]:
    print(i,end = ' ')
print(sorted(l)[-1])
```

​               ![image-20240227135138003](C:\Users\骆春阳\AppData\Roaming\Typora\typora-user-images\image-20240227135138003.png)                

#### 时间：

20分钟

 

## 2.学习总结和收获

因为初来乍到，对许多东西还不太熟悉，虽然上学期我也学的python，但是相对老师的提高班的底子还是弱了。最近正在疯狂刷题（当然实力有限，做题比较慢），有时间就补补算法。群里大家讨论的各种“堆”“桶”“树”“图”我都不知道是什么，我想争取抓紧补齐这方面的短板。

这几天也试着刷了每日一题，有的时候两道题都能写上，有的时候弄不上。之前自学了一下“栈”，然后做一个合法出栈序列的题目还是崩了，甚至题刚开始都没读懂。跟着一堆大佬学习真的很累，不过却有很多收获，真的很喜欢群里大家无时无刻不在一起讨论的氛围，喜欢每天都能在oj上看到新的题目而跃跃欲试的感觉。期待能从闫宏飞老师这里学到更多知识。

ps：第一次试着用typora弄md，终于学会了好开心！
