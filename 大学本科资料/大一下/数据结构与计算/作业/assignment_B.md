# Assignment #B: 图论和树算

Updated 1709 GMT+8 Apr 28, 2024

2024 spring, Complied by ==骆春阳 工学院==



**说明：**

1）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

2）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

3）如果不能在截止前提交作业，请写明原因。



**编程环境**

操作系统：Windows 11

Python编程环境：PyCharm 2023.1.4



## 1. 题目

### 28170: 算鹰

dfs, http://cs101.openjudge.cn/practice/28170/



思路：刚开始没读懂题，后来发现是联通块数量



代码

```python

lst = []
for _ in range(10):
    lst.append(list(input()))
data = set() ##储存已访问的节点
count = 0
def bfs(lst,i,j):
    if 0 <= i <= 9 and 0 <= j <= 9 and (i,j) not in data:
        if (i,j) not in data:
            data.add((i,j))
        if lst[i][j] == '.':
            bfs(lst,i-1,j)
            bfs(lst,i+1,j)
            bfs(lst,i,j-1)
            bfs(lst,i,j+1)
            return True
        else:
            return False
    return False

for i in range(10):
    for j in range(10):
        if bfs(lst,i,j):
            count += 1
print(count)

```

![image-20240507215447611](C:\Users\骆春阳\AppData\Roaming\Typora\typora-user-images\image-20240507215447611.png)

时间：20分钟





### 02754: 八皇后

dfs, http://cs101.openjudge.cn/practice/02754/



思路：一个比较正常的dfs

答案有个地方比较巧妙，直接将放置皇后的地放填入它的列数，方便后来的整理结果的过程



代码

```python
def is_safe(board, row, col): ##每次放置都要检查一下
    for i in range(row):
        if board[i] == col:
            return False
    i = row - 1
    j = col - 1
    while i >= 0 and j >= 0:
        if board[i] == j:
            return False
        i -= 1
        j -= 1
    i = row - 1
    j = col + 1
    while i >= 0 and j < 8:
        if board[i] == j:
            return False
        i -= 1
        j += 1
    return True

def queen_dfs(board, row):
    if row == 8: ##放置完毕
        # 找到第b个解，将解存储到result列表中
        ans.append(''.join([str(x+1) for x in board]))
        return
    for col in range(8):
        if is_safe(board, row, col):
            board[row] = col
            queen_dfs(board, row + 1)
            board[row] = 0

ans = []
queen_dfs([None]*8, 0)
for _ in range(int(input())):
    print(ans[int(input()) - 1])

```

![image-20240507220129544](C:\Users\骆春阳\AppData\Roaming\Typora\typora-user-images\image-20240507220129544.png)

代码运行截图 ==（至少包含有"Accepted"）==

时间：25分钟

### 03151: Pots

bfs, http://cs101.openjudge.cn/practice/03151/



思路：比较硬核，用bfs的方式找到可行的方法

记得用一个visited储存访问过的状态

（感觉这个visited是核心，是bfs必不可少的一部分）



代码

```python
from collections import deque

def bfs(A, B, C):
    visited = set()  # 用于记录已经访问过的状态
    queue = deque([(0, 0, [])])  # 使用队列进行广度优先搜索，元素为 (current_A, current_B, operations)
    
    while queue:
        current_A, current_B, operations = queue.popleft()
        
        # 判断是否已经达到目标状态
        if current_A == C or current_B == C:
            return operations
        
        # 尝试执行所有可能的操作
        # FILL(i) 操作
        if current_A != A:
            new_A = A
            new_B = current_B
            if (new_A, new_B) not in visited:
                visited.add((new_A, new_B))
                queue.append((new_A, new_B, operations + ["FILL(1)"]))
        
        if current_B != B:
            new_A = current_A
            new_B = B
            if (new_A, new_B) not in visited:
                visited.add((new_A, new_B))
                queue.append((new_A, new_B, operations + ["FILL(2)"]))
        
        # DROP(i) 操作
        if current_A != 0:
            new_A = 0
            new_B = current_B
            if (new_A, new_B) not in visited:
                visited.add((new_A, new_B))
                queue.append((new_A, new_B, operations + ["DROP(1)"]))
        
        if current_B != 0:
            new_A = current_A
            new_B = 0
            if (new_A, new_B) not in visited:
                visited.add((new_A, new_B))
                queue.append((new_A, new_B, operations + ["DROP(2)"]))
        
        # POUR(i, j) 操作
        if current_A != 0 and current_B != B:
            new_A = max(0, current_A - (B - current_B))
            new_B = min(B, current_B + current_A)
            if (new_A, new_B) not in visited:
                visited.add((new_A, new_B))
                queue.append((new_A, new_B, operations + ["POUR(1,2)"]))
        
        if current_B != 0 and current_A != A:
            new_B = max(0, current_B - (A - current_A))
            new_A = min(A, current_A + current_B)
            if (new_A, new_B) not in visited:
                visited.add((new_A, new_B))
                queue.append((new_A, new_B, operations + ["POUR(2,1)"]))
    
    return None  # 如果无法达到目标状态，则返回None

# 读取输入
A, B, C = map(int, input().split())

# 进行广度优先搜索
result = bfs(A, B, C)

# 输出结果
if result is None:
    print("impossible")
else:
    print(len(result))
    for operation in result:
        print(operation)


```

![image-20240507232036164](C:\Users\骆春阳\AppData\Roaming\Typora\typora-user-images\image-20240507232036164.png)

时间：45分钟





### 05907: 二叉树的操作

http://cs101.openjudge.cn/practice/05907/



思路：思路比较清晰，先建树，然后每个操作都定义一个函数

刚开始wa，对了一下答案，发现自己输入的部分和change节点的代码比较复杂（屎山）

于是修改了一下，简洁了很多



代码

```python
class tree_node:
    def __init__(self,value):
        self.value = value
        self.left = None
        self.right = None

def left_action(x):
    for element in lst:
        if element.value == x:
            root = element
            break
    while root.left:
        root = root.left
    return root.value


def change_action(x,y):
    for node in lst:
        if node.left and node.left.value in [x,y]:
            if node.left.value == x:
                node.left = lst[y]
            else:
                node.left = lst[x]
        if node.right and node.right.value in [x,y]:
            if node.right.value == x:
                node.right = lst[y]
            else:
                node.right = lst[x]

t = int(input())
for _ in range(t):
    n,m = map(int,input().split())
    lst = [tree_node(_) for _ in range(n)]
    for i in range(n):
        root,left_root,right_root = map(int,input().split())
        if left_root != -1:
            lst[root].left = lst[left_root]
        if right_root != -1:
            lst[root].right = lst[right_root]

    root = lst[0]

    for j in range(m):
        action = list(map(int,input().split()))
        if action[0] == 2:
            x = action[-1]
            print(left_action(x))
        if action[0] == 1:
            x,y = action[1],action[2]
            change_action(x,y)

```

![image-20240507230709346](C:\Users\骆春阳\AppData\Roaming\Typora\typora-user-images\image-20240507230709346.png)



45分钟





### 18250: 冰阔落 I

Disjoint set, http://cs101.openjudge.cn/practice/18250/



思路：用了并查集就好很多

看gpt写的

代码

```python
def findIceKawlo(n, m, operations):
    cup_to_kawlo = list(range(n + 1))  # 初始化每个杯子的阔落编号为自身
    ice_cups = set(range(1, n + 1))  # 初始化所有杯子都有冰阔落

    def find(cup):
        if cup_to_kawlo[cup] != cup:
            cup_to_kawlo[cup] = find(cup_to_kawlo[cup])
        return cup_to_kawlo[cup]

    def union(cup1, cup2):
        root1 = find(cup1)
        root2 = find(cup2)
        if root1 != root2:
            cup_to_kawlo[root2] = root1
            ice_cups.discard(root2)

    results = []
    for x, y in operations:
        if find(x) == find(y):
            results.append("Yes")
        else:
            results.append("No")
            union(x, y)

    num_ice_cups = len(ice_cups)
    sorted_ice_cups = sorted(ice_cups)

    return results, num_ice_cups, sorted_ice_cups





while True:
    try:
        all_results = []
        n, m = map(int, input().split())
        operations = []
        for _ in range(m):
            x, y = map(int, input().split())
            operations.append((x, y))

        results, num_ice_cups, sorted_ice_cups = findIceKawlo(n, m, operations)
        all_results.append((results, num_ice_cups, sorted_ice_cups))

        # 输出结果
        for results, num_ice_cups, sorted_ice_cups in all_results:
            for result in results:
                print(result)
            print(num_ice_cups)
            print(*sorted_ice_cups)
    except EOFError:
        break

```

![image-20240507234308616](C:\Users\骆春阳\AppData\Roaming\Typora\typora-user-images\image-20240507234308616.png)

时间：60分钟





### 05443: 兔子与樱花

http://cs101.openjudge.cn/practice/05443/



思路：不太会用dijkstra算法，照着答案学了一下

之后还要反复自己写一下试试



代码

```python
import heapq

def dijkstra(adjacency, start):
    distances = {vertex: float('infinity') for vertex in adjacency}
    previous = {vertex: None for vertex in adjacency}
    distances[start] = 0
    pq = [(0, start)]

    while pq:
        current_distance, current_vertex = heapq.heappop(pq)
        if current_distance > distances[current_vertex]:
            continue

        for neighbor, weight in adjacency[current_vertex].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous[neighbor] = current_vertex
                heapq.heappush(pq, (distance, neighbor))

    return distances, previous

def shortest_path_to(adjacency, start, end):
    distances, previous = dijkstra(adjacency, start)
    path = []
    current = end
    while previous[current] is not None:
        path.insert(0, current)
        current = previous[current]
    path.insert(0, start)
    return path, distances[end]

# Read the input data
P = int(input())
places = {input().strip() for _ in range(P)}

Q = int(input())
graph = {place: {} for place in places}
for _ in range(Q):
    src, dest, dist = input().split()
    dist = int(dist)
    graph[src][dest] = dist
    graph[dest][src] = dist  # Assuming the graph is bidirectional

R = int(input())
requests = [input().split() for _ in range(R)]

# Process each request
for start, end in requests:
    if start == end:
        print(start)
        continue

    path, total_dist = shortest_path_to(graph, start, end)
    output = ""
    for i in range(len(path) - 1):
        output += f"{path[i]}->({graph[path[i]][path[i+1]]})->"
    output += f"{end}"
    print(output)

```

![image-20240507232352726](C:\Users\骆春阳\AppData\Roaming\Typora\typora-user-images\image-20240507232352726.png)

时间：50分钟





## 2. 学习总结和收获

昨天几乎通宵学了高代，结果状态很差，今天上课的内容得及时补一下（再也不通宵学习了）（要通宵也要学数算×）

自己写迪杰斯特拉算法还是有些费劲，需要看答案才能完整。

之后这几道相关的题重复做一下，争取能自己手打出来

