# Assignment #8: 图论：概念、遍历，及 树算

Updated 1919 GMT+8 Apr 8, 2024

2024 spring, Complied by ==骆春阳 工学院==



**说明：**

1）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

2）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

3）如果不能在截止前提交作业，请写明原因。



**编程环境**

操作系统：Windows 11

Python编程环境：Spyder IDE 5.2.2, PyCharm 2023.1.4 (Professional Edition)

C/C++编程环境：Mac terminal vi (version 9.0.1424), g++/gcc (Apple clang version 14.0.3, clang-1403.0.22.14.1)



## 1. 题目

### 19943: 图的拉普拉斯矩阵

matrices, http://cs101.openjudge.cn/practice/19943/

请定义Vertex类，Graph类，然后实现



思路：每个顶点都存储一下邻居和度

虽然这道题邻居用不上，但可以试着先写一下



代码

```python
class vertex:
    def __init__(self,value):
        self.value = value
        self.degree = 0 ##每个节点的度
        self.neighbor = [] ##每个节点的邻居

class graph:
    def __init__(self,n):
        self.n = n ##顶点个数
        self.ver_list = [vertex(i) for i in range(n)]  ##每个顶点存储一下
        self.matrix = [[0 for i in range(n)] for j in range(n)]  ##初始化拉普拉斯矩阵

    def change_vertex(self,a,b): ##修改顶点的值
        self.ver_list[a].degree += 1
        self.ver_list[b].degree += 1
        self.ver_list[a].neighbor.append(b)
        self.ver_list[b].neighbor.append(a)
        self.matrix[a][b] = -1
        self.matrix[b][a] = -1
    def laplace(self):
        for node in self.ver_list:
            value = node.value
            i = node.degree
            self.matrix[value][value] = i
        return self.matrix
n,m = map(int,input().split())
gra = graph(n)
for _ in range(m):
    a,b = map(int,input().split())
    gra.change_vertex(a,b)
result = gra.laplace()
for i in result:
    print(*i)

```



![image-20240413150018905](C:\Users\骆春阳\AppData\Roaming\Typora\typora-user-images\image-20240413150018905.png)



时间：30分钟

复习了一下如何用类来写

### 18160: 最大连通域面积

matrix/dfs similar, http://cs101.openjudge.cn/practice/18160



思路：递归查找



代码

```python
def dfs(i,j,lst,visited):
    if i < 0 or i >= row or j < 0 or j >= line or not lst[i][j] or visited[i][j]:
        return 0
    visited[i][j] = True
    count = 1
    count += dfs(i - 1,j - 1,lst,visited)
    count += dfs(i - 1, j, lst, visited)
    count += dfs(i - 1, j + 1, lst, visited)
    count += dfs(i, j - 1, lst, visited)
    count += dfs(i, j + 1, lst, visited)
    count += dfs(i + 1, j - 1, lst, visited)
    count += dfs(i + 1, j, lst, visited)
    count += dfs(i + 1, j + 1, lst, visited)
    return count


n = int(input()) ##次数
for _ in range(n):
    row,line = map(int,input().split()) ##行数和列数
    lst = []
    for _ in range(row):
        lst.append([])
        for i in input():
            lst[-1].append(0 if i == '.' else 1) ##如果是W就是1，如果是。就是0
    visited = [[False for i in range(line)] for j in range(row)] ##判断循环的是否是否查找过
    M = 0
    for i in range(row):
        for j in range(line):
            if lst[i][j]:
                M = max(M,dfs(i,j,lst,visited))
    print(M)


```



![image-20240413152508397](C:\Users\骆春阳\AppData\Roaming\Typora\typora-user-images\image-20240413152508397.png)



时间：30分钟

### sy383: 最大权值连通块

https://sunnywhy.com/sfbj/10/3/383



思路：将每个顶点存起来，然后遍历即可



代码

```python
class vertex:
    def __init__(self,value):
        self.value = value
        self.degree = 0
        self.neighbor = []


def dfs(node, visited):
    if visited[node.value]:
        return 0
    count = node.degree
    visited[node.value] = True
    if node.neighbor:
        for number in node.neighbor:
            count += dfs(vertex_lst[number],visited)
    return count
n,m = map(int,input().split())
vertex_lst = [vertex(i) for i in range(n)]
visited = [False for i in range(n)]
lst = list(map(int,input().split()))
for i in range(n):
    vertex_lst[i].degree = lst[i] ##修改权值
for _ in range(m):
    a,b = map(int,input().split())
    vertex_lst[a].neighbor.append(b)
    vertex_lst[b].neighbor.append(a) ##互相记录一下
M = 0
for node in vertex_lst:
    M = max(M,dfs(node,visited))
print(M)


```



![image-20240413223152421](C:\Users\骆春阳\AppData\Roaming\Typora\typora-user-images\image-20240413223152421.png)



时间：20分钟

### 03441: 4 Values whose Sum is 0

data structure/binary search, http://cs101.openjudge.cn/practice/03441



思路：将每一列的元素放进一个表里，这样得到四个表，

可以先算前两个表的和，放到字典里，然后查找



代码

```python
def suan(A,B,C,D):
    dic_sum = {}
    count = 0

    ##先存储前两个表的和
    for a in A:
        for b in B:
            if a+b not in dic_sum.keys():
                dic_sum[a+b] = 1
            else:
                dic_sum[a+b] += 1

    for c in C:
        for d in D:
            if - (c + d) in dic_sum.keys():
                count += dic_sum[-(c+d)]
    return count

n = int(input())
A,B,C,D = [],[],[],[]
for _ in range(n):
    a,b,c,d = map(int,input().split())
    A.append(a)
    B.append(b)
    C.append(c)
    D.append(d)
print(suan(A,B,C,D))

```



![image-20240414111353192](C:\Users\骆春阳\AppData\Roaming\Typora\typora-user-images\image-20240414111353192.png)

时间：10分钟（感觉自己算法有点暴力）



### 04089: 电话号码

trie, http://cs101.openjudge.cn/practice/04089/

Trie 数据结构可能需要自学下。



思路：学习了trie之后开始写

刚开始tle

然后我优化了一下输入

同时用了一个新的函数，方便遇到前缀匹配成功后直接退出，省略多余的查找部分

然后出了漏洞，debug之后，发现自己要吞掉一些冗余的输入。

再然后大佬提醒我children的位置用字典查找更快

（字典查找比表的index快）

所以再修改了一下

时间是424ms



代码

```python
class tree_node:
    def __init__(self,value):
        self.value = value
        self.number_end = False ##判断是不是一个号码的结尾
        self.children = {}
def judge(lst,root):
    flag = False ##判断————如果在插入的时候有新的地方变成了True，就不改变flag
    i = 0
    while i < len(lst) and int(lst[i]) in root.children.keys():   ##如果根节点已经存储过这个数字
        root = root.children[int(lst[i])]
        if root.number_end: ##如果在跑的过程中经过了存储的数字
            flag = True
            return flag
        i += 1
    if i == len(lst):
        flag = True
        return flag
    while i < len(lst):
        root.children[int(lst[i])] = tree_node(int(lst[i])) ##把数字放进去
        root = root.children[int(lst[i])]
        i += 1
    root.number_end = True
    return flag
def judge_2():
    judgement = False
    m = int(input())
    j = None
    for i in range(m):
        if judge(input(),root_1): ##如果判断之后是前缀
            judgement = True
            j = i
            break
    if j:
        for j in range(j+1,m):
            s = input()
    return judgement

n = int(input())
for _ in range(n):
    root_1 = tree_node(-1)
    result = judge_2()
    if result:
        print('NO')
    else:
        print('YES')

```

![image-20240414123013635](C:\Users\骆春阳\AppData\Roaming\Typora\typora-user-images\image-20240414123013635.png)

时间：1个小时左右

### 04082: 树的镜面映射

http://cs101.openjudge.cn/practice/04082/



思路：没想明白...感觉自己有一种思路，但是会出问题

想试着用栈来建树，但是没通过兄弟儿子的原理建明白

最后还是决定向大佬学习了一下代码



代码

```python
from collections import deque

class TreeNode:
    def __init__(self, x):
        self.x = x
        self.children = []

def create_node():
    return TreeNode('')

def build_tree(tempList, index):
    node = create_node()
    node.x = tempList[index][0]
    if tempList[index][1] == '0' and node.x != '$':
        index += 1
        child, index = build_tree(tempList, index)
        node.children.append(child)
        index += 1
        child, index = build_tree(tempList, index)
        node.children.append(child)
    return node, index

def print_tree(p):
    Q = deque()
    s = deque()

    # 遍历右子节点并将非虚节点加入栈s
    while p is not None:
        if p.x != '$':
            s.append(p)
        p = p.children[1] if len(p.children) > 1 else None

    # 将栈s中的节点逆序放入队列Q
    while s:
        Q.append(s.pop())

    # 宽度优先遍历队列Q并打印节点值
    while Q:
        p = Q.popleft()
        print(p.x, end=' ')

        # 如果节点有左子节点，将左子节点及其右子节点加入栈s
        if p.children:
            p = p.children[0]
            while p is not None:
                if p.x != '$':
                    s.append(p)
                p = p.children[1] if len(p.children) > 1 else None

            # 将栈s中的节点逆序放入队列Q
            while s:
                Q.append(s.pop())

# 读取输入
n = int(input())
tempList = input().split(' ')

# 构建多叉树
root, _ = build_tree(tempList, 0)

# 执行宽度优先遍历并打印镜像映射序列
print_tree(root)

```



![image-20240416231147042](C:\Users\骆春阳\AppData\Roaming\Typora\typora-user-images\image-20240416231147042.png)



时间：60分钟左右

## 2. 学习总结和收获

期中考试每日选做落下了，要抓紧补上

发现复习的时候做一道数算挺好的（

还要多做题，向AC5，AC6的方向努力





