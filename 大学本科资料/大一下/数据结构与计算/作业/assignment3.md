# Assignment #3: March月考

Updated 1537 GMT+8 March 6, 2024

2024 spring, Complied by **骆春阳 工学院**



**说明：**

1）The complete process to learn DSA from scratch can be broken into 4 parts:

- Learn about Time and Space complexities
- Learn the basics of individual Data Structures
- Learn the basics of Algorithms
- Practice Problems on DSA

2）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

3）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

4）如果不能在截止前提交作业，请写明原因。



**编程环境**

操作系统：Windows 11

Python编程环境：PyCharm 2023.1.4 (Professional Edition)



## 1. 题目

**02945: 拦截导弹**

http://cs101.openjudge.cn/practice/02945/



思路：dp动态规划



##### 代码

```python
# 
n = int(input())
lst = list(map(int,input().split()))
dp = [1 for _ in range(n)]
for i in range(n):
    for j in range(i):
        if lst[j] >= lst[i]:
            dp[i] = max(dp[j] + 1,dp[i])
print(max(dp))
```



代码运行截图 ==（至少包含有"Accepted"）==

![image-20240311164941141](C:\Users\骆春阳\AppData\Roaming\Typora\typora-user-images\image-20240311164941141.png)

时间：算上复习dp的话一个小时多了。



**04147:汉诺塔问题(Tower of Hanoi)**

http://cs101.openjudge.cn/practice/04147



思路：递归（但是没递归明白，就看了一下答案）



##### 代码

```python

# 将编号为numdisk的盘子从init杆移至desti杆 
def moveOne(numDisk : int, init : str, desti : str):
    print("{}:{}->{}".format(numDisk, init, desti))

#将numDisks个盘子从init杆借助temp杆移至desti杆
def move(numDisks : int, init : str, temp : str, desti : str):
    if numDisks == 1:
        moveOne(1, init, desti)
    else: 
        # 首先将上面的（numDisk-1）个盘子从init杆借助desti杆移至temp杆
        move(numDisks-1, init, desti, temp) 
        
        # 然后将编号为numDisks的盘子从init杆移至desti杆
        moveOne(numDisks, init, desti)
        
        # 最后将上面的（numDisks-1）个盘子从temp杆借助init杆移至desti杆 
        move(numDisks-1, temp, init, desti)

n, a, b, c = input().split()
move(int(n), a, b, c)
```



代码运行截图 ==（至少包含有"Accepted"）==



![image-20240311185022930](C:\Users\骆春阳\AppData\Roaming\Typora\typora-user-images\image-20240311185022930.png)

时间：一分钟（ac的时间），Ctrl+c+v。

之后学习递归的时间就要好久了，这一周接下来还要去力扣针对刷一些

**03253: 约瑟夫问题No.2**

http://cs101.openjudge.cn/practice/03253

思路：队列实现（队列的rotate方法太好用了）

##### 代码

```python
# 
from collections import deque
while True:
    number,pos,judge = map(int,input().split())
    if number == 0:
        break
    lst = deque(i for i in range(1,number + 1))
    lst.rotate(-pos+1)
    R = []
    while len(lst) != 0:
        lst.rotate(-judge+1)
        R.append(str(lst.popleft()))
    print(','.join(R))
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240311181143127](C:\Users\骆春阳\AppData\Roaming\Typora\typora-user-images\image-20240311181143127.png)

时间：3分钟



**21554:排队做实验 (greedy)v0.2**

http://cs101.openjudge.cn/practice/21554



思路：一点小小的贪心



##### 代码

```python
# 
n = int(input())
lst = list(map(int,input().split()))
sequence = []
dic = {}
for i in range(len(lst)):
    dic.setdefault(lst[i],[]).append(i+1) ##找到那个元素的位置
add = 0
p = len(lst)
for _ in sorted(dic.keys()):
    L = dic.get(_)
    for i in L:
        add += (p-1) * _
        p -= 1
    sequence.extend(L)
print(*sequence)
print(f'{(add)/n:.2f}')
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240311181304014](C:\Users\骆春阳\AppData\Roaming\Typora\typora-user-images\image-20240311181304014.png)

时间：30分钟（考试的时候写的）

主要是刚开始题没读懂，写错了

**19963:买学区房**

http://cs101.openjudge.cn/practice/19963



思路：排序



##### 代码

```python
# 
n = int(input())
pairs = [i[1:-1] for i in input().split()]
distances = [sum(map(int,i.split(','))) for i in pairs] ##距离求和
cost = list(map(int,input().split()))
lst = []
for i in range(n):
    lst.append([distances[i] / cost[i],cost[i]])
lst.sort(key = lambda x:x[0])
result = 0
distances.sort()
cost.sort()
if n % 2 == 0:
    mid_judge = (lst[n // 2][0]+lst[n // 2 - 1][0]) / 2
    mid_cost = (cost[n // 2] + cost[n // 2 - 1]) / 2
elif n % 2 == 1:
    mid_judge = lst[(n-1) // 2][0]
    mid_cost = cost[(n - 1) // 2]

for j in lst[n//2 - 2:]:
    if j[0] > mid_judge and j[1] < mid_cost:
        result += 1
print(result)
```

![image-20240311184637508](C:\Users\骆春阳\AppData\Roaming\Typora\typora-user-images\image-20240311184637508.png)

时间：20分钟（全是写代码的时间，思考的时间比较少）

代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==





**27300: 模型整理**

http://cs101.openjudge.cn/practice/27300



思路：

将输入按照M和B分隔开，然后分别排序，再合并

##### 代码

```python
# 
dic = {}
for _ in range(int(input())):
    model,sale = map(str,input().split('-'))
    dic.setdefault(model,[]).append(sale)
for k in sorted(dic.keys()):
    print(f'{k}:',end = ' ')
    L = dic.get(k)
    R1 = []
    R2 = []
    R = []
    for i in L:
        if 'B' in i:
            R1.append(i[:-1])
        else:
            R2.append(i[:-1])
    R1.sort(key = lambda x: float(x))
    R2.sort(key = lambda x: float(x))
    for i in R2:
        R.append(i+'M')
    for j in R1:
        R.append(j+'B')
    print(', '.join(R))
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240311181437971](C:\Users\骆春阳\AppData\Roaming\Typora\typora-user-images\image-20240311181437971.png)

时间：30分钟



## 2. 学习总结和收获

首先是考试的总结。这次考试前两道题对我来说难度都很高，递归和dp我在原来的基础班几乎是没有练过的，老师讲过但是缺乏练习。于是先做了后面的，最终ac3。

我的策略有一些问题，当时有些固执的，买学区房的问题太长暂时先没看，ac3了之后剩下50分钟都在弄拦截导弹，自己写了一个繁琐复杂的“假动态规划”，最终以wa告终。今天做了买学区房问题，发现自己完全有能力在规定时间内写出来，所以正常的话是能够ac4的。

然后是这周的学习，这周的每日选做弃了很多，一是要备战北大杯，占了很多时间，几乎天天都要训练（累死，本来以为训练后有精力学习，结果训练的时候很费专注力，回寝室后根本专注不起来，只想睡觉补状态）；

二是决定先抽大块的时间针对的练一些算法，每日选做的题毕竟是什么样的题都有。周二的时候发现力扣可以针对训练，决定先在上面练习一些。接下来的一个月左右都没有比赛，训练仍然有但是不用太心急了（因为很难出线了已经www），决定把本周的大块时间整理一下专门攻克数算问题，一是dp，二是递归。

树的话决定跟老师慢慢走。自己还有dfs，桶之类的算法没有学习，慢慢来总能学会的。

3.11今天下午比较清闲，写完数算的会的作业后。花了两个小时左右研究动态规划，练了一个最长上升子序列，重写了一遍拦截导弹。又看群里，学习了一个bisect方法，剩下的时间弄二维数组。

（发现01背包问题其实有很多解法，自己上学期自学的一点点动态规划的代码有些繁琐，还可以更精进，也问了同学，配合递归的方式会更加简洁）决定周三左右精进递归，顺带着重学01背包。



每日选坐的猪堆，顺便去学了如何写堆，还有懒删除的想法，收获很大。

要知道省时间的原理：获取堆顶的元素以及从set中删除元素的时间复杂度都是O(1)。

另外要时刻提醒自己，设计到数字比较大小，一定不能忘记int或者float一下再比较（哭）



