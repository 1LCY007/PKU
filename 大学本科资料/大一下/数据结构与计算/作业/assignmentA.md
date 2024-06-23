# Assignment #A: 图论：遍历，树算及栈

Updated 2018 GMT+8 Apr 21, 2024

2024 spring, Complied by ==骆春阳 工学院==



**说明：**

1）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

2）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

3）如果不能在截止前提交作业，请写明原因。



**编程环境**

操作系统：Windows 11

Python编程环境：PyCharm 2023.1.4 (Professional Edition)



## 1. 题目

### 20743: 整人的提词本

http://cs101.openjudge.cn/practice/20743/



思路：用栈解决，找到右括号再找左括号



代码

```python
s = input()
stack = []
for i in s:
    if i == ')':
        temp = []
        while stack and stack[-1] != '(':
            temp.append(stack.pop())
        if stack and stack[-1] == '(':
            stack.pop()
        stack.extend(temp)
    else:
        stack.append(i)
print(''.join(stack))


```

![image-20240428235952598](C:\Users\骆春阳\AppData\Roaming\Typora\typora-user-images\image-20240428235952598.png)



时间：20分钟

### 02255: 重建二叉树

http://cs101.openjudge.cn/practice/02255/



思路：挺简单的根据前中序表达式建树转后序表达



代码

```python
##给定一棵二叉树的前序遍历和中序遍历的结果，求其后序遍历。
class treenode:
    def __init__(self,value):
        self.value = value
        self.right = None
        self.left = None
def build_tree(pre_s,mid_s):
    if not pre_s:
        return None
    root = pre_s[0]
    root_node = treenode(root)
    root_index_mid = mid_s.index(root)
    root_node.left = build_tree(pre_s[1:root_index_mid+1],mid_s[:root_index_mid])
    root_node.right = build_tree(pre_s[1+root_index_mid:],mid_s[1+root_index_mid:])
    return root_node

def behind(root):
    if root is None:
        return []
    result = []
    result.extend(behind(root.left))
    result.extend(behind(root.right))
    result.append(root.value)
    return result

while True:
    try:
        pre_s,mid_s = input().split()
        root = build_tree(pre_s,mid_s)
        print(''.join(behind(root)))
    except EOFError:
        break

```

![image-20240429001809685](C:\Users\骆春阳\AppData\Roaming\Typora\typora-user-images\image-20240429001809685.png)

代码运行截图 ==（至少包含有"Accepted"）==

时间：20分钟



### 01426: Find The Multiple

http://cs101.openjudge.cn/practice/01426/

要求用bfs实现



思路：有点类似树的层次遍历（刚开始想建一个01二叉树来着）

后来发现用队列（然后超时）

看完答案知道，可以通过用余数的方式来减少多余的计算



代码

```python
def judge(n):
    while not result:
        x = lst.pop(0)
        if x % n == 0:
            return x
        if x % n not in data:
            data.add(x % n)
            lst.append(x * 10)
            lst.append(x * 10 + 1)
while True:
    n = int(input())
    if n == 0:
        break
    data = set()
    lst = [1]
    result = 0
    print(judge(n))


```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240430215905575](C:\Users\骆春阳\AppData\Roaming\Typora\typora-user-images\image-20240430215905575.png)

时间：20分钟

### 04115: 鸣人和佐助

bfs, http://cs101.openjudge.cn/practice/04115/





代码

```python
from collections import deque
class Node:
    def __init__(self, x, y, tools, steps):
        self.x = x
        self.y = y
        self.tools = tools
        self.steps = steps


M, N, T = map(int, input().split())
maze = [list(input()) for _ in range(M)]
visit = [[[0] * (T + 1) for _ in range(N)] for _ in range(M)]
directions = [[-1, 0], [1, 0], [0, -1], [0, 1]]
start = end = None
flag = 0
for i in range(M):
    for j in range(N):
        if maze[i][j] == '@':
            start = Node(i, j, T, 0)
            visit[i][j][T] = 1
        if maze[i][j] == '+':
            end = (i, j)
            maze[i][j] = '*'

queue = deque([start])
while queue:
    node = queue.popleft()
    if (node.x, node.y) == end:
        print(node.steps)
        flag = 1
        break
    for direction in directions:
        nx, ny = node.x + direction[0], node.y + direction[1]
        if 0 <= nx < M and 0 <= ny < N:
            if maze[nx][ny] == '*':
                if not visit[nx][ny][node.tools]:
                    queue.append(Node(nx, ny, node.tools, node.steps + 1))
                    visit[nx][ny][node.tools] = 1
            elif maze[nx][ny] == '#':
                if node.tools > 0 and not visit[nx][ny][node.tools - 1]:
                    queue.append(Node(nx, ny, node.tools - 1, node.steps + 1))
                    visit[nx][ny][node.tools - 1] = 1

if not flag:
    print("-1")

```

![image-20240430220741111](C:\Users\骆春阳\AppData\Roaming\Typora\typora-user-images\image-20240430220741111.png)

代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==





### 20106: 走山路

Dijkstra, http://cs101.openjudge.cn/practice/20106/





代码

```python
# 

```

![image-20240430221014269](C:\Users\骆春阳\AppData\Roaming\Typora\typora-user-images\image-20240430221014269.png)

代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==





### 05442: 兔子与星空

Prim, http://cs101.openjudge.cn/practice/05442/





代码

```python
import heapq

def prim(graph, start):
    mst = []
    used = set([start])
    edges = [
        (cost, start, to)
        for to, cost in graph[start].items()
    ]
    heapq.heapify(edges)

    while edges:
        cost, frm, to = heapq.heappop(edges)
        if to not in used:
            used.add(to)
            mst.append((frm, to, cost))
            for to_next, cost2 in graph[to].items():
                if to_next not in used:
                    heapq.heappush(edges, (cost2, to, to_next))

    return mst

def solve():
    n = int(input())
    graph = {chr(i+65): {} for i in range(n)}
    for i in range(n-1):
        data = input().split()
        star = data[0]
        m = int(data[1])
        for j in range(m):
            to_star = data[2+j*2]
            cost = int(data[3+j*2])
            graph[star][to_star] = cost
            graph[to_star][star] = cost
    mst = prim(graph, 'A')
    print(sum(x[2] for x in mst))

solve()

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240430221213302](C:\Users\骆春阳\AppData\Roaming\Typora\typora-user-images\image-20240430221213302.png)



## 2. 学习总结和收获

因为出去旅游了有些耽误，所以这次抄了两道题，打算明后天回学校之后复习一下。

逐渐开始复习了，应该排好复习计划。这学期有很多知识都是新学习的，树，图，堆等等。

尤其要磨练自己写冗长的代码的耐心。





