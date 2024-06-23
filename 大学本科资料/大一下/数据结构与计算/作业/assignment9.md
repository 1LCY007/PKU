# Assignment #9: 图论：遍历，及 树算

Updated 1739 GMT+8 Apr 14, 2024

2024 spring, Complied by ==骆春阳==



**说明：**

1）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

2）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

3）如果不能在截止前提交作业，请写明原因。



**编程环境**

操作系统：Windows 11

Python编程环境：PyCharm 2023.1.4 



## 1. 题目

### 04081: 树的转换

http://cs101.openjudge.cn/dsapre/04081/



思路：用栈处理



代码

```python
# def tree_heights(s):
    old_height = 0
    max_old = 0
    new_height = 0
    max_new = 0
    stack = []
    for c in s:
        if c == 'd':
            old_height += 1
            max_old = max(max_old, old_height)

            new_height += 1
            stack.append(new_height)
            max_new = max(max_new, new_height)
        else:
            old_height -= 1

            new_height = stack[-1]
            stack.pop()
    return f"{max_old} => {max_new}"

s = input().strip()
print(tree_heights(s))

```

![image-20240423170842133](C:\Users\骆春阳\AppData\Roaming\Typora\typora-user-images\image-20240423170842133.png)

代码运行截图 ==（至少包含有"Accepted"）==

时间：40分钟

### 08581: 扩展二叉树

http://cs101.openjudge.cn/dsapre/08581/



思路：处理输入和之前数的镜面映射中是一样的处理方式，把树建好其他都OK



代码

```python
def build_tree(preorder):
    if not preorder or preorder[0] == '.':
        return None, preorder[1:]
    root = preorder[0]
    left, preorder = build_tree(preorder[1:])
    right, preorder = build_tree(preorder)
    return (root, left, right), preorder

def inorder(tree):
    if tree is None:
        return ''
    root, left, right = tree
    return inorder(left) + root + inorder(right)

def postorder(tree):
    if tree is None:
        return ''
    root, left, right = tree
    return postorder(left) + postorder(right) + root

# 输入处理
preorder = input().strip()

# 构建扩展二叉树
tree, _ = build_tree(preorder)

# 输出结果
print(inorder(tree))
print(postorder(tree))

```



代码运行截图 ==（至少包含有"Accepted"）==

![image-20240423171054355](C:\Users\骆春阳\AppData\Roaming\Typora\typora-user-images\image-20240423171054355.png)

时间：30分钟

### 22067: 快速堆猪

http://cs101.openjudge.cn/practice/22067/



思路：堆，懒删除



代码

```python
import heapq
stack = []
Min = []
min_pop = set()
while True:
    try:
        s = input()
        if s == 'pop':
            if stack:
                min_pop.add(stack.pop())
        elif s == 'min':
            while Min and Min[0] in min_pop:
                min_pop.remove(heapq.heappop(Min))
            if Min:
                print(Min[0])
        elif 'push' in s:
            stack.append(int(list(s.split())[-1]))
            heapq.heappush(Min,stack[-1])
    except EOFError:
        break
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240423171239603](C:\Users\骆春阳\AppData\Roaming\Typora\typora-user-images\image-20240423171239603.png)

时间：20分钟

### 04123: 马走日

dfs, http://cs101.openjudge.cn/practice/04123



思路：



代码

```python
maxn = 10;
sx = [-2, -1, 1, 2, 2, 1, -1, -2]
sy = [1, 2, 2, 1, -1, -2, -2, -1]

ans = 0;


def Dfs(dep: int, x: int, y: int):
    # 是否已经全部走完
    if n * m == dep:
        global ans
        ans += 1
        return

    # 对于每个可以走的点
    for r in range(8):
        s = x + sx[r]
        t = y + sy[r]
        if chess[s][t] == False and 0 <= s < n and 0 <= t < m:
            chess[s][t] = True
            Dfs(dep + 1, s, t)
            chess[s][t] = False;  # 回溯


for _ in range(int(input())):
    n, m, x, y = map(int, input().split())
    chess = [[False] * maxn for _ in range(maxn)]  # False表示没有走过
    ans = 0
    chess[x][y] = True
    Dfs(1, x, y)
    print(ans)

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240423160830537](C:\Users\骆春阳\AppData\Roaming\Typora\typora-user-images\image-20240423160830537.png)



### 28046: 词梯

bfs, http://cs101.openjudge.cn/practice/28046/



思路：利用桶建图再bfs

（大佬的代码好厉害）



代码

```python
from collections import deque
class Vertex:
    def __init__(self,id):
        self.id=id
        self.neighbors={}
        # 当找到路径后，通过Previous用于显式地把路径体现出来
        self.previous=None
        self.color='white'
    def __str__(self):
        return '*'+self.id
class Graph:
    def __init__(self):
        self.vertices={}
        self.num_vertices=0

    def add_vertex(self,id):
        self.vertices[id]=Vertex(id)
        self.num_vertices+=1

    def add_edge(self,v1_id,v2_id):
        # v1_start,v2_end
        if v1_id not in self.vertices:
            self.vertices[v1_id]=Vertex(v1_id)
        if v2_id not in self.vertices:
            self.vertices[v2_id]=Vertex(v2_id)
        v1,v2=self.vertices[v1_id],self.vertices[v2_id]
        v1.neighbors[v2_id]=v2
        self.num_vertices+=1

n,graph,buckets=int(input()),Graph(),{}
words=[input() for _ in range(n)]
for word in words:
    for bit in range(1,len(word)+1):
        tag=word[:bit-1]+'_'+word[bit:]
        bucket=buckets.setdefault(tag,set())
        bucket.add(word)
# for i,j in buckets.items():
#     print(i,j)
for bucket in buckets.values():
    for i in bucket:
        tmp=bucket-{i}
        for j in tmp:
            graph.add_edge(i,j)


start,goal=map(str,input().split())
# BFS,这里不用函数实现
q=deque()
q.append(graph.vertices[start])
current=graph.vertices[start]
# 注：标黑色用于把回头路堵死；标灰色用于把更长的可行路径堵死。
# 由于更长的可行路径被堵死且最短路径唯一，所以每个点的前驱若有有则仅有一个。
while q and current.id!=goal:
    current=q.popleft()
    for vert in current.neighbors.values():
        if vert.color=='white':
            vert.color='grey'
            vert.previous=current
            q.append(vert)
    current.color='black'

def traverse(start:Vertex):
    output=[start.id]
    current=start
    while current.previous:
        output.append(current.previous.id)
        current=current.previous
    return " ".join(output[::-1])

if current.id==goal:
    print(traverse(graph.vertices[goal]))
else:
    print("NO")

```

![image-20240423201708321](C:\Users\骆春阳\AppData\Roaming\Typora\typora-user-images\image-20240423201708321.png)

代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

时间：>2hour



### 28050: 骑士周游

dfs, http://cs101.openjudge.cn/practice/28050/



思路：没有思路，堪堪把代码看懂一些，但还要复习，有点乱



代码

```python
import sys

class Graph:
    def __init__(self):
        self.vertices = {}
        self.num_vertices = 0

    def add_vertex(self, key):
        self.num_vertices = self.num_vertices + 1
        new_ertex = Vertex(key)
        self.vertices[key] = new_ertex
        return new_ertex

    def get_vertex(self, n):
        if n in self.vertices:
            return self.vertices[n]
        else:
            return None

    def __len__(self):
        return self.num_vertices

    def __contains__(self, n):
        return n in self.vertices

    def add_edge(self, f, t, cost=0):
        if f not in self.vertices:
            nv = self.add_vertex(f)
        if t not in self.vertices:
            nv = self.add_vertex(t)
        self.vertices[f].add_neighbor(self.vertices[t], cost)
        #self.vertices[t].add_neighbor(self.vertices[f], cost)

    def getVertices(self):
        return list(self.vertices.keys())

    def __iter__(self):
        return iter(self.vertices.values())


class Vertex:
    def __init__(self, num):
        self.key = num
        self.connectedTo = {}
        self.color = 'white'
        self.distance = sys.maxsize
        self.previous = None
        self.disc = 0
        self.fin = 0

    def __lt__(self,o):
        return self.key < o.key

    def add_neighbor(self, nbr, weight=0):
        self.connectedTo[nbr] = weight


    # def setDiscovery(self, dtime):
    #     self.disc = dtime
    #
    # def setFinish(self, ftime):
    #     self.fin = ftime
    #
    # def getFinish(self):
    #     return self.fin
    #
    # def getDiscovery(self):
    #     return self.disc

    def get_neighbors(self):
        return self.connectedTo.keys()

    # def getWeight(self, nbr):
    #     return self.connectedTo[nbr]

    def __str__(self):
        return str(self.key) + ":color " + self.color + ":disc " + str(self.disc) + ":fin " + str(
            self.fin) + ":dist " + str(self.distance) + ":pred \n\t[" + str(self.previous) + "]\n"



def knight_graph(board_size):
    kt_graph = Graph()
    for row in range(board_size):           #遍历每一行
        for col in range(board_size):       #遍历行上的每一个格子
            node_id = pos_to_node_id(row, col, board_size) #把行、列号转为格子ID
            new_positions = gen_legal_moves(row, col, board_size) #按照 马走日，返回下一步可能位置
            for row2, col2 in new_positions:
                other_node_id = pos_to_node_id(row2, col2, board_size) #下一步的格子ID
                kt_graph.add_edge(node_id, other_node_id) #在骑士周游图中为两个格子加一条边
    return kt_graph

def pos_to_node_id(x, y, bdSize):
    return x * bdSize + y

def gen_legal_moves(row, col, board_size):
    new_moves = []
    move_offsets = [                        # 马走日的8种走法
        (-1, -2),  # left-down-down
        (-1, 2),  # left-up-up
        (-2, -1),  # left-left-down
        (-2, 1),  # left-left-up
        (1, -2),  # right-down-down
        (1, 2),  # right-up-up
        (2, -1),  # right-right-down
        (2, 1),  # right-right-up
    ]
    for r_off, c_off in move_offsets:
        if (                                # #检查，不能走出棋盘
            0 <= row + r_off < board_size
            and 0 <= col + c_off < board_size
        ):
            new_moves.append((row + r_off, col + c_off))
    return new_moves

# def legal_coord(row, col, board_size):
#     return 0 <= row < board_size and 0 <= col < board_size


def knight_tour(n, path, u, limit):
    u.color = "gray"
    path.append(u)              #当前顶点涂色并加入路径
    if n < limit:
        neighbors = ordered_by_avail(u) #对所有的合法移动依次深入
        #neighbors = sorted(list(u.get_neighbors()))
        i = 0

        for nbr in neighbors:
            if nbr.color == "white" and \
                knight_tour(n + 1, path, nbr, limit):   #选择“白色”未经深入的点，层次加一，递归深入
                return True
        else:                       #所有的“下一步”都试了走不通
            path.pop()              #回溯，从路径中删除当前顶点
            u.color = "white"       #当前顶点改回白色
            return False
    else:
        return True

def ordered_by_avail(n):
    res_list = []
    for v in n.get_neighbors():
        if v.color == "white":
            c = 0
            for w in v.get_neighbors():
                if w.color == "white":
                    c += 1
            res_list.append((c,v))
    res_list.sort(key = lambda x: x[0])
    return [y[1] for y in res_list]

def main():
    def NodeToPos(id):
       return ((id//8, id%8))

    bdSize = int(input())  # 棋盘大小
    *start_pos, = map(int, input().split())  # 起始位置
    g = knight_graph(bdSize)
    start_vertex = g.get_vertex(pos_to_node_id(start_pos[0], start_pos[1], bdSize))
    if start_vertex is None:
        print("fail")
        exit(0)

    tour_path = []
    done = knight_tour(0, tour_path, start_vertex, bdSize * bdSize-1)
    if done:
        print("success")
    else:
        print("fail")

    exit(0)

    # 打印路径
    cnt = 0
    for vertex in tour_path:
        cnt += 1
        if cnt % bdSize == 0:
            print()
        else:
            print(vertex.key, end=" ")
            #print(NodeToPos(vertex.key), end=" ")   # 打印坐标

if __name__ == '__main__':
    main()


```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

时间：2hour



![image-20240423170554696](C:\Users\骆春阳\AppData\Roaming\Typora\typora-user-images\image-20240423170554696.png)

## 2. 学习总结和收获

这次作业总体来说，对我还是有些难度的。骑士周游，词梯两道题就给我CPU干烧了。

自己看课件、听课的时候也比较蒙，感觉图的算法目前没有很好地掌握，似乎好像都理解了，但是自己做不出来题。

明白类的巧妙，但是代码一长，自己有时候就写蒙了

还需要多联系一下。

另外，树的部分也要复习一下。

最近各科都飞快地上强度，要排好时间逐个击破了。







