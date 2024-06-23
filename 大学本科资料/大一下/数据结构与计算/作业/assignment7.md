# Assignment #7: April 月考

Updated 1557 GMT+8 Apr 3, 2024

2024 spring, Complied by ==骆春阳 工学院==



**说明：**

1）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

2）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

3）如果不能在截止前提交作业，请写明原因。



**编程环境**

操作系统：Windows 11

Python编程环境：PyCharm 2023.1.4



## 1. 题目

### 27706: 逐词倒放

http://cs101.openjudge.cn/practice/27706/



思路：没什么好说的



代码

```python
lst = list(map(str,input().split()))
print(*lst[::-1])

```



![image-20240407171441565](C:\Users\骆春阳\AppData\Roaming\Typora\typora-user-images\image-20240407171441565.png)



时间：20s

### 27951: 机器翻译

http://cs101.openjudge.cn/practice/27951/



思路：简单的队列



代码

```python
from collections import deque

v,number = map(int,input().split())
lst = deque(maxlen = v)
count = 0
for i in map(int,input().split()):

    if i not in lst:
        count += 1
        lst.append(i)
print(count)

```



![image-20240407171547728](C:\Users\骆春阳\AppData\Roaming\Typora\typora-user-images\image-20240407171547728.png)



时间：10分钟左右吧

### 27932: Less or Equal

http://cs101.openjudge.cn/practice/27932/



思路：处理特殊情况浪费了时间

前面用归并排序

因为我以为归并排序最快（

事实上用sorted直接就行（哭）



代码

```python
def mergeSort(arr):
    if len(arr) > 1:
        mid = len(arr) // 2

        L = arr[:mid]
        R = arr[mid:]

        mergeSort(L)
        mergeSort(R)

        i = j = k = 0
        while i < len(L) and j < len(R):
            if L[i] <= R[j]:
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1

        while i < len(L):
            arr[k] = L[i]
            i += 1
            k += 1

        while j < len(R):
            arr[k] = R[j]
            j += 1
            k += 1


n, k = map(int, input().split())
lst = list(map(int, input().split()))
mergeSort(lst)
if k == 0 and lst[0] == 1:
    print(-1)
elif k == 0 and lst[0] != 1:
    print(1)
elif k >= 1:
    if k == n:
        print(lst[n-1])
    elif k != n:
        if lst[k - 1] == lst[k]:
            print('-1')
        else:
            print(lst[k - 1])

```



![image-20240407171646479](C:\Users\骆春阳\AppData\Roaming\Typora\typora-user-images\image-20240407171646479.png)

时间：15分钟左右

### 27948: FBI树

http://cs101.openjudge.cn/practice/27948/



思路：比较简单的二叉树



代码

```python
class tree_node:
    def __init__(self,value):
        self.value = value
        self.left = None
        self.right = None

n = int(input())
s = input() ##输入01字符串
def judge(s):
    if '0' in s and '1' in s:
        value = 'F'
    elif '0' in s and '1' not in s:
        value = 'B'
    elif '0' not in s and '1' in s:
        value = 'I'
    return value
def build_tree(s):
    value = judge(s)
    node = tree_node(value)
    if len(s) > 1:

        node.left = build_tree(s[: len(s) // 2])
        node.right = build_tree(s[len(s) // 2 :])
    return node
root = build_tree(s)

def behind_order(root):
    if root is None:
        return []
    result = []
    result.extend(behind_order(root.left))
    result.extend(behind_order(root.right))
    result.append(root.value)
    return result
print(''.join(behind_order(root)))

```



![image-20240407171817321](C:\Users\骆春阳\AppData\Roaming\Typora\typora-user-images\image-20240407171817321.png)



时间：25分钟

### 27925: 小组队列

http://cs101.openjudge.cn/practice/27925/



思路：处理输入非常简单，用字典将其存起来，每个小组有一个编号（0到n-1）而且这个时候可以用集合set存储，因为不需要考虑顺序

在接下来的时候我打算用一个临时表储记录已经进入队列的小组号，每次插入的时候先检索一下这个小组号是否被记录过，如果被记录过就index小组号的位置，并在D列表里插入。

刚开始RE了

后来发现会有些人不在任何一个小组（不符题意啊喂）

但是思路没有变，只需要看看这个人有没有小组号，没有的话直接插入到D的最后面，并且在临时表里用-1填补它的位置即可。

感觉思路清晰就会很快做出来

代码

```python
n = int(input())
dic = {}
lst_sequence = [] ##存储位置
D = []
for _ in range(n):
    dic[_] = set(input().split())
def insert(s):
    root = None
    for key in dic.keys():
        if s in dic[key]:
            root = key ##找到对应的位置
            break
    if root is None:
        lst_sequence.append(-1) ##加-1占位
        D.append([s])
    elif root not in lst_sequence: ##如果没被记录过
        lst_sequence.append(root)
        D.append([s])
    else: ##如果被记录过
        root_index = lst_sequence.index(root)
        D[root_index].append(s)
def delete(s):
    if D[0]:
        print(D[0].pop(0))
    if not D[0]:##如果删没了
        lst_sequence.pop(0)
        D.pop(0)

while True:
    x = input()
    if x[0] == 'D':
        delete(x)
    elif x[0] == 'E':
        s = list(x.split())[-1]
        insert(s)
    elif x[0] == 'S':
        break

```



![image-20240408112948343](C:\Users\骆春阳\AppData\Roaming\Typora\typora-user-images\image-20240408112948343.png)

时间：40分钟



### 27928: 遍历树

http://cs101.openjudge.cn/practice/27928/



思路：处理输入的时候还好，没有当过儿子的人就是爹

输出的时候，考试的时候没有想到怎么递归，看大佬的代码让我醍醐灌顶。

在按字典序输出的时候，如果检测到的值不是自己的儿子（那么也就代表着是自己）就输出

否则就递归着遍历



代码

```python
class tree_node:
    def __init__(self,value):
        self.value = value
        self.children = []

def tran_tree(root,nodes):
    if not root.children:
        return print(root.value)
    dic = {root.value:root}
    for child in root.children:
        dic[child] = nodes[child]
    for value in sorted(dic.keys()):
        if value in root.children:
            tran_tree(dic[value],nodes)
        else:
            print(root.value) ##判断我要输出的是父节点还是子节点


n = int(input()) ##节点个数
child_node = set()
nodes = {}
for _ in range(n):
    lst = list(map(int,input().split()))
    nodes[lst[0]] = tree_node(lst[0])
    for child in lst[1:]:
        nodes[lst[0]].children.append(child)
        if child not in child_node:
            child_node.add(child)
    for i in sorted(nodes.keys()):
        if i not in child_node:
            root = nodes[i]
            break
tran_tree(root,nodes)

```



![image-20240407171159530](C:\Users\骆春阳\AppData\Roaming\Typora\typora-user-images\image-20240407171159530.png)



时间：90分钟（考试的时候50分钟）

## 2. 学习总结和收获

==如果作业题目简单，有否额外练习题目，比如：OJ“2024spring每日选做”、CF、LeetCode、洛谷等网站题目。==

这次考试AC4用了不到一个小时，但是两道难题没有做出来（时间全花在了最后一道题，但是没递归明白）

不过如果花四十分钟做小组队列的话还是能自己做出来的，遍历树也是看大佬的代码才明白。

总结一下比上次两个小时ac3好了很多，水平大概在AC4.5左右？。

接下来要练练稍微难一点的题，不过首要任务先做每日一题。

每日一题跟的不太好，清明假期荒废了（身体病情比较严重）

之后要抓紧跟上。



