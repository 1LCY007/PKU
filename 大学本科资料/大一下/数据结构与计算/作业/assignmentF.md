# Assignment #F: All-Killed 满分

Updated 1844 GMT+8 May 20, 2024

2024 spring, Complied by ==骆春阳 工学院==



**说明：**

1）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

2）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

3）如果不能在截止前提交作业，请写明原因。



**编程环境**

操作系统：Windows 11

Python编程环境：PyCharm 2023.2.6



## 1. 题目

### 22485: 升空的焰火，从侧面看

http://cs101.openjudge.cn/practice/22485/



思路：巧妙用一下bfs(代码中写错了，原先还以为是dfs)

我的处理方式是，给每个节点都标一下高度，这样遍历的时候只要到某一层的结尾就可以输出了。

代码

```python
class treenode:
    def __init__(self,value):
        self.value = value
        self.height = None
        self.right = None
        self.left = None
n = int(input())
lst = [treenode(i) for i in range(1,n+1)]
for j in range(n):
    left_son,right_son = map(int,input().split())
    if left_son != -1:
        lst[j].left = lst[left_son - 1]
    if right_son != -1:
        lst[j].right = lst[right_son - 1]
root = lst[0]

def find_height(root,height):
    root.height = height
    if root.left:
        find_height(root.left,height + 1)
    if root.right:
        find_height(root.right,height + 1)


def dfs(root):
    queue = []
    data = set() ##记录已经用过的高度
    data.add(0)
    stack = [root.value]
    result = []
    queue.append(root)
    while queue:
        node = queue.pop(0)
        if node.height not in data:
            result.append(stack[-1])
            stack[-1] = node.value
            data.add(node.height)
        else:
            stack[-1] = node.value
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
    result.append(stack[-1])
    return result
find_height(root,0)
print(*dfs(root))

```

![image-20240528185356289](C:\Users\骆春阳\AppData\Roaming\Typora\typora-user-images\image-20240528185356289.png)

30分钟

### 28203:【模板】单调栈

http://cs101.openjudge.cn/practice/28203/



思路：模版单调栈，在cheetsheet中记录了

考试的时候可以直接抄

（这一题自己手敲的，对单调栈加深了理解）



代码

```python
n = int(input())
lst = list(int(i) for i in input().split())
pos = [0 for i in range(n)] ##初始化位置
judge = lst[0]
stack = []
for i in range(n-1,-1,-1):
    while stack and lst[stack[-1]] <= lst[i]:
        stack.pop()
    if stack:
        pos[i] = stack[-1] + 1
    stack.append(i)
print(*pos)


```

![image-20240528194733831](C:\Users\骆春阳\AppData\Roaming\Typora\typora-user-images\image-20240528194733831.png)

代码运行截图 ==（至少包含有"Accepted"）==

15分钟



### 09202: 舰队、海域出击！

http://cs101.openjudge.cn/practice/09202/



思路：

拓扑排序

在老师给的原始代码基础上自己修改了一下。一是取消了导入Queue，因为不太习惯，直接抄容易出问题，不好debug

二是在添加前先检查其是否在graph中，否则会报错（如果不在graph，则说明一条路走到了头

三则是开始录入入度为0的额点，循环的数是节点数，不是len(graph)

原本想的是机考中作为cheetsheet抄的，现在看来好像自己手敲也不算问题了

代码

```python
from collections import defaultdict

def topological_sort(graph):
    indegree = defaultdict(int)
    result = []
    queue = []
    # 计算每个顶点的入度
    for u in graph:
        for v in graph[u]:
            indegree[v] += 1
    # 将入度为 0 的顶点加入队列
    for u in range(1,node+1):
        if indegree[u] == 0:
            queue.append(u)
    # 执行拓扑排序
    while queue:
        u = queue.pop(0)
        result.append(u)
        if u in graph: ##如果不在graph里，说明到了一个终点
            for v in graph[u]:
                indegree[v] -= 1
                if indegree[v] == 0:
                    queue.append(v)
    # 检查是否存在环
    if len(result) == node: ##有环时None，表示无法排序
        return True
    else:
        return False
n = int(input())
for _ in range(n):
    node,edge = map(int,input().split())
    dic = {}
    for i in range(edge):
        start,ending = map(int,input().split())
        dic.setdefault(start,[]).append(ending)
    if not topological_sort(dic):
        print('Yes')
    else:
        print('No')


```

![image-20240528202153369](C:\Users\骆春阳\AppData\Roaming\Typora\typora-user-images\image-20240528202153369.png)

代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

时间：40分钟



### 04135: 月度开销

http://cs101.openjudge.cn/practice/04135/



思路：看了一下思路，很快就明白了，手敲出来。（有一说一，GPT有的题写不对，但是他的思路还是很清晰的）

二分查找，不错不错，

因为要找最大值的最小区间，那么第一步，极端想法。

​	1.这个区间只有一个数，也就是整个数组中的max。

​	2.这个区间很大，是整个数组，也就是sum

这一步明白了之后就很清晰了，我们想要的答案肯定在max和sum中间，而答案要求输出结果是区间的最大值，那就是最终的某个区间的sum

如何找这个最终的sum呢？二分查找提供了一个思路，来把这个大区间不断压缩的操作，用夹挤原理得出一个数，这个压缩的过程，嗯。

——切片。

不断将原有区间按照切片数的不同反复切片，直到找到符合目标的结果。

那么根据这些，我们可以按照一下方式进行切片：

​	1.规定初始切片数cut为1（整个区间），之后每次进行切片操作时cut + 1

​	2.判断是否切片的方式为：该分区间内的总和是否超过了mid.

​		==解释：mid是max和sum之和的二分之一（二分法由来）

​	3.check一下切片数和要求的m是否相同，如果cut>m，说明切片数超了（表示分的太细了，max值要更大一些，取mid值）

如果cut<=m,说明当前分法可以，但是可以试着找一下有没有更好的分法（即再减小sum的值）

当max超过sum的时候，终止条件，我们得到了最终结果（即没有更好的分法了）



代码

```python
n,m = map(int,input().split())
lst = []
for i in range(n):
    lst.append(int(input()))

def check(lst,mid):
    add = 0
    cut = 1
    for i in range(n):
        if lst[i] + add > mid:
            add = lst[i]
            cut += 1
        elif lst[i] + add <= mid:
            add += lst[i]
    if cut <= m:
        return True
    else:
        return False

sum_up = sum(lst)
M = max(lst)
result = 0
while M < sum_up:
    mid = (M + sum_up) // 2
    if check(lst,mid):
        sum_up = mid
        result = mid
    else:
        M = mid + 1
print(result)

```

![image-20240531162301565](C:\Users\骆春阳\AppData\Roaming\Typora\typora-user-images\image-20240531162301565.png)

代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

30分钟



### 07735: 道路

http://cs101.openjudge.cn/practice/07735/



思路：习惯用类存储每个城市的性质。

需要注意的是这道题是有向图，（刚开始按照无向图做的）



代码

```python
import heapq
class city_node:
    def __init__(self,value):
        self.value = value
        self.neighbor = {}

max_cost = int(input())
city_sum = int(input())
road_sum = int(input())
lst = [city_node(i) for i in range(1,city_sum + 1)]
for _ in range(road_sum):
    start,ending,length,cost = map(int,input().split())
    lst[start - 1].neighbor.setdefault(ending,[]).append((length,cost))

end = city_sum
def find_road(lst,max_cost):
    queue = []
    heapq.heappush(queue,(0,0,1)) ##前两个分别是，道路长度，和花费的金币数,最后一个是step，记录走过的路数
    while queue:
        l,c,s = heapq.heappop(queue)
        if s == end: ##到终点了就弹出
            return l
        node = lst[s - 1]
        for pass_city in node.neighbor.keys():
            for pass_l,pass_cost in node.neighbor[pass_city]:
                if pass_cost + c <= max_cost:
                    heapq.heappush(queue,(l + pass_l,pass_cost + c,pass_city))
    return -1
print(find_road(lst,max_cost))


```

![image-20240531175356862](C:\Users\骆春阳\AppData\Roaming\Typora\typora-user-images\image-20240531175356862.png)

代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

时间：50分钟左右



### 01182: 食物链

http://cs101.openjudge.cn/practice/01182/



思路：备考机考，所以这题不死琢磨了（脑袋疼）

初步看了一下，是稍微复杂一点的并查集，因为内涵三个维度来维护并查集，同类，吃别人的，还有被吃的



代码

```python
class DisjointSet:
    def __init__(self, n):
        #设[1,n] 区间表示同类，[n+1,2*n]表示x吃的动物，[2*n+1,3*n]表示吃x的动物。
        self.parent = [i for i in range(3 * n + 1)] # 每个动物有三种可能的类型，用 3 * n 来表示每种类型的并查集
        self.rank = [0] * (3 * n + 1)

    def find(self, u):
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])
        return self.parent[u]

    def union(self, u, v):
        pu, pv = self.find(u), self.find(v)
        if pu == pv:
            return False
        if self.rank[pu] > self.rank[pv]:
            self.parent[pv] = pu
        elif self.rank[pu] < self.rank[pv]:
            self.parent[pu] = pv
        else:
            self.parent[pv] = pu
            self.rank[pu] += 1
        return True


def is_valid(n, k, statements):
    dsu = DisjointSet(n)

    def find_disjoint_set(x):
        if x > n:
            return False
        return True

    false_count = 0
    for d, x, y in statements:
        if not find_disjoint_set(x) or not find_disjoint_set(y):
            false_count += 1
            continue
        if d == 1:  # X and Y are of the same type
            if dsu.find(x) == dsu.find(y + n) or dsu.find(x) == dsu.find(y + 2 * n):
                false_count += 1
            else:
                dsu.union(x, y)
                dsu.union(x + n, y + n)
                dsu.union(x + 2 * n, y + 2 * n)
        else:  # X eats Y
            if dsu.find(x) == dsu.find(y) or dsu.find(x + 2*n) == dsu.find(y):
                false_count += 1
            else: #[1,n] 区间表示同类，[n+1,2*n]表示x吃的动物，[2*n+1,3*n]表示吃x的动物
                dsu.union(x + n, y)
                dsu.union(x, y + 2 * n)
                dsu.union(x + 2 * n, y + n)

    return false_count


if __name__ == "__main__":
    N, K = map(int, input().split())
    statements = []
    for _ in range(K):
        D, X, Y = map(int, input().split())
        statements.append((D, X, Y))
    result = is_valid(N, K, statements)
    print(result)


```

![image-20240531200430235](C:\Users\骆春阳\AppData\Roaming\Typora\typora-user-images\image-20240531200430235.png)

代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==





## 2. 学习总结和收获

==如果作业题目简单，有否额外练习题目，比如：OJ“2024spring每日选做”、CF、LeetCode、洛谷等网站题目。==

= =这次作业总结：

​	图又精进了一些，还学会了单调栈的技巧。月度开销通透了，但是遗憾考试不考

= =备战机考！要主要复习一下图，然后树的递归再看一看。看一看其他老师的题目。

​	（不知道为什么自己对图有一种天生的抵触，不太愿意写图的题orz，咬紧牙关再写几道)

​	（而且总想把图写成树，可能我太喜欢树了）



最后一次总结了，这半年过的真快。

虽然数算结束了，但是还想继续深入学习计算机语言，在这里感受到了快乐。（虽然偶尔挺头疼）

永无止境！

（上学期报提高班就好了）

或许假期可以看一看c++?（有时候为python的速度而忧患orz）

感谢老师，助教和同学的长久陪伴！



