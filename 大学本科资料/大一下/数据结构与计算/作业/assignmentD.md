# Assignment #D: May月考

Updated 1654 GMT+8 May 8, 2024

2024 spring, Complied by ==骆春阳 工学院==



**说明：**

1）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

2）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

3）如果不能在截止前提交作业，请写明原因。



**编程环境**

操作系统：Windows 11

Python编程环境：PyCharm 2023.1.4



## 1. 题目

### 02808: 校门外的树

http://cs101.openjudge.cn/practice/02808/



思路：遍历特定范围的表，修改特定的值



代码

```python
n,m = map(int,input().split())
lst = [1 for _ in range(n+1)]
for _ in range(m):
    start,ending = map(int,input().split())
    for k in range(start,ending+1):
        if lst[k] != 0:
            lst[k] = 0
print(lst.count(1))

```



![image-20240516154121310](C:\Users\骆春阳\AppData\Roaming\Typora\typora-user-images\image-20240516154121310.png)

时间：2分钟



### 20449: 是否被5整除

http://cs101.openjudge.cn/practice/20449/



思路：队列



代码

```python
s = list(input())
result = ''
k = 0
count = 1
while s:
    k += 2 ** (len(s) - count) * int(s.pop(0))
    if k % 5 == 0:
        result += '1'
    else:
        result += '0'
print(result)

```



![image-20240516154244274](C:\Users\骆春阳\AppData\Roaming\Typora\typora-user-images\image-20240516154244274.png)

时间：5分钟



### 01258: Agri-Net

http://cs101.openjudge.cn/practice/01258/



思路：最小生成树



代码

```python
##权值最小————最小生成树
##用krustal
class DisjointSet: ##先弄一个并查集
    def __init__(self, num_vertices):
        self.parent = list(range(num_vertices))
        self.rank = [0] * num_vertices

    def find(self, x): 
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x != root_y:
            if self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
            elif self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
            else:
                self.parent[root_x] = root_y
                self.rank[root_y] += 1
while True:
    try:
        n = int(input())
        L = []
        for i in range(n):
            lst = list(map(int,input().split()))
            for j in range(len(lst)):
                if j != i:
                    L.append((i,j,lst[j]))

        L.sort(key = lambda x:x[2]) ##按照边的权值比大小
        tree = DisjointSet(n)
        count = 0 ##记录边数
        result = 0
        while count < n - 1:

            element = L.pop(0)
            x = element[0]
            y = element[1]
            if tree.find(x) != tree.find(y):
                tree.union(x,y)
                count += 1
                result += element[2]
        print(result)
    except EOFError:
        break




```

![image-20240516160451530](C:\Users\骆春阳\AppData\Roaming\Typora\typora-user-images\image-20240516160451530.png)

复习完图之后写的，15分钟搞定（并查集复制粘贴的）





### 27635: 判断无向图是否连通有无回路(同23163)

http://cs101.openjudge.cn/practice/27635/

用拓扑排序进行判断是否有回路

用bfs判断是否有连通

思路：

```python
##判断是否连通（是否是断开的图）
#判断是否有回路可以看是否有节点属于两个
from collections import defaultdict
from queue import Queue

##用拓扑排序判断是否有回路
def topological_sort(graph):
    indegree = defaultdict(int)
    result = []
    queue = Queue()

    # 计算每个顶点的入度
    for u in graph:
        for v in graph[u]:
            indegree[v] += 1

    # 将入度为 0 的顶点加入队列
    for u in graph:
        if indegree[u] <= 1:
            queue.put(u)

    # 执行拓扑排序
    while not queue.empty():
        u = queue.get()
        result.append(u)

        for v in graph[u]:
            indegree[v] -= 1
            if indegree[v] == 1:
                queue.put(v)

    # 检查是否存在环
    if len(result) == len(graph):
        return result
    else:
        return None
n,m = map(int,input().split())
dic = {}
for i in range(m):
    x,y = map(int,input().split())
    dic.setdefault(x,[]).append(y)
    dic.setdefault(y,[]).append(x)

if topological_sort(dic):
    loop = 'no'
else:
    loop = 'yes'

def judge_connect(graph):
    lst = []
    visited = set()
    lst.append(graph[0]) ##从0开始查找
    count = 0
    while lst:
        root = lst.pop()
        for i in root:
            if i in graph.keys():
                if i not in visited:
                    count += 1
                    visited.add(i)
                    lst.append(graph[i])
    if count == n:
        return True
    else:
        return False
if judge_connect(dic):
    connected = 'yes'
else:
    connected = 'no'
print(f'connected:{connected}')
print(f'loop:{loop}')
```





![image-20240516182550707](C:\Users\骆春阳\AppData\Roaming\Typora\typora-user-images\image-20240516182550707.png)

代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

时间：40分钟左右







### 27947: 动态中位数

http://cs101.openjudge.cn/practice/27947/

思路：刚开始没太懂思路，自知普通算法肯定会超时

后来查了一下，发现用大根堆和小根堆来求中位数，提示的话是用大根堆和小根堆

自己又试着写了一段时间，花了二十多分钟才明白

其实本质上就是将一个数组以中位数为pivot分开，分成一堆较小的和一堆较大的

数据正常比较，往堆里放。但为了保证能够时刻快速得到中位数，保证两个堆的元素数量之差不大于1即可

（题外话，用heapq来构建最大堆的时候可以把原来的数变成相反数即可，但是这样似乎只限于全正或全负，如果想完全实现的话，可能要用二叉树建两个堆）

代码

```python
import heapq

n = int(input())

def insert(num):
    if len(Min) == 0 or num > Min[0]:
        heapq.heappush(Min, num)
    else:
        heapq.heappush(Max,-num)
    if len(Min) - len(Max) > 1:
        heapq.heappush(Max,-heapq.heappop(Min))
    elif len(Max) - len(Min) > 1:
        heapq.heappush(Min, -heapq.heappop(Max))
for _ in range(n):
    result = []
    count = 0
    Min = []
    Max = []
    lst = list(map(int,input().split()))
    for num in range(len(lst)):
        count += 1
        insert(lst[num])
        if count % 2 == 1:
            if len(Min) > len(Max):
                result.append(Min[0])
            else:
                result.append(-Max[0])
    print(len(result))
    print(*result)


```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240516193022254](C:\Users\骆春阳\AppData\Roaming\Typora\typora-user-images\image-20240516193022254.png)

时间：1个小时多

### 28190: 奶牛排队

http://cs101.openjudge.cn/practice/28190/



思路：这题让我痛并快乐着（

首先，去学习了一下单调栈，然后看代码有点懵，看了很久才知道构建单调递减栈和单调递增栈确定边界值。

然后最后循环的时候对每个右边界找一下左边界就可以

（对了这个break还改变了我很久的一个错误的想法， 我又去自己在pycharm上试了一下确认。break只能退出一层循环！！！

之前一直以为break很强大，能直接退出所有循环）

代码

```python
N = int(input())
heights = [int(input()) for _ in range(N)]

left_bound = [-1] * N
right_bound = [N] * N

stack = []  # 单调栈，存储索引

# 求左侧第一个≥h[i]的奶牛位置
for i in range(N): ##单调递减栈
    while stack and heights[stack[-1]] < heights[i]:
        stack.pop()

    if stack:
        left_bound[i] = stack[-1]

    stack.append(i)

stack = []  # 清空栈以供寻找右边界使用

# 求右侧第一个≤h[i]的奶牛位
for i in range(N-1, -1, -1):
    while stack and heights[stack[-1]] > heights[i]:
        stack.pop()

    if stack:
        right_bound[i] = stack[-1]

    stack.append(i)

ans = 0

for i in range(N):  # 枚举右端点 B寻找 A，更新 ans
    for j in range(left_bound[i] + 1, i):
        if right_bound[j] > i:
            ans = max(ans, i - j + 1)
            break
print(ans)

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240517163246251](C:\Users\骆春阳\AppData\Roaming\Typora\typora-user-images\image-20240517163246251.png)

两个点

## 2. 学习总结和收获

==如果作业题目简单，有否额外练习题目，比如：OJ“2024spring每日选做”、CF、LeetCode、洛谷等网站题目。==

1.本周主要把重心放回了数算（后悔期中没有退高等代数，不如学数算）

2。把图整体重新复习了一下，并补了一些习题。

----图以前从来没有接触过，要对那几个算法都熟练，跟着课件额外做了最短路径的一些题和最小生成树的。

3.另外笔试的复习也开始了，跟着老师以前的课件在有规划地看课后笔试题目（上次模考出现了很多原题）

感觉每次考试重新跟着复习一下整理知识点，并把以前课件出的题过了，再抽时间看看往年考试题，笔试应该问题不大。

4.另外，重新做了一份cheetsheet——graph和tree的（但是github不知道怎么创建文件夹，周末要好好开发一下github的功能，然后把这学期的东西都放上去==当然上学期的也抽空整理一下放上面，真的好用）

5.这次模考勉强ac4，因为图练得少，这周正在狂补。

第五题第六题全新的思路，没见过，也拓展了自己的想法，非常开心。



#### sy386: 最短距离 简单

https://sunnywhy.com/sfbj/10/4/386

这题练习Dijkstra 算法。

先用模版写一遍再自己写一遍

注意到老师给的模版代码适用于有向图

无向图只需要修改某一行代码，把两个分别设置成对方的neighbor即可

```python
import heapq
def djs(graph,vertex):
    stack = []
    stack.append((0,vertex))
    visited = set()
    while stack:
        dis,vertex = heapq.heappop(stack)
        if vertex == ending:
            return dis
        if vertex in visited:
            continue
        visited.add(vertex)
        if vertex not in graph:
            continue
        path = graph.get(vertex)
        for x in path:
            target = x[0]
            dist = x[1]
            if target not in visited:
                if dis + dist < distance[target]:
                    distance[target] = dis + dist
                    heapq.heappush(stack, (dis + dist,target))

dic = {}
n_vertex,n_edge,start,ending = map(int,input().split())
distance = [float('inf') for i in range(n_vertex)]
for i in range(n_edge): ##加入进去
    x,y,cost = map(int,input().split())
    dic.setdefault(x,[]).append((y,cost))
    dic.setdefault(y,[]).append((x,cost))
distance[start] = 0
djs(dic,start)
if distance[ending] != float('inf'):
    print(distance[ending])
else:
    print(-1)
```

自己手搓了一下迪杰斯特拉，虽然搓了一点多点，但收获还是很多

刚开始比较倔，试着不用堆写，但发现还是不可避免地会出现问题（数据较大的时候），数据少的时候没有问题，但是跑的时间更长。

第二是要给没有边连的点存有余地，比如老师给的答案代码里用的是二维数组，保证检索的时候一定能找到。

（我刚开始没有想到，于是总出现NoneType的情况，后来加入了==if vertex not in graph==，来判断某一点是否有边）

#### 19930: 寻宝

bfs, http://cs101.openjudge.cn/practice/19930

关键在于队列，保证不同方向的进度是相同的，这样的话能保证找到的第一个结果一定是最好的结果

所以广度优先搜索队列重要在于此。

这题写了三十多分钟（主要是之前没太理解队列在bfs中的作用）

```python
from collections import deque

def bfs(m, n, grid):
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    visited = [[False] * n for _ in range(m)]
    start = (0, 0)
    queue = deque([(start, 0)])  # 起点和步数入队
    visited[start[0]][start[1]] = True

    while queue:
        current, steps = queue.popleft()
        x, y = current

        if grid[x][y] == 1:  # 到达藏宝点
            return steps

        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            if 0 <= nx < m and 0 <= ny < n and grid[nx][ny] != 2 and not visited[nx][ny]:
                visited[nx][ny] = True
                queue.append(((nx, ny), steps + 1))

    return -1  # 无法到达藏宝点

m, n = map(int, input().split())
grid = []

for _ in range(m):
    row = list(map(int, input().split()))
    grid.append(row)

result = bfs(m, n, grid)

if result == -1:
    print("NO")
else:
    print(result)
```



周五前就这样了，周末继续复习图，这块还没有那么熟悉。
