# Assignment #5: "树"算：概念、表示、解析、遍历

Updated 2124 GMT+8 March 17, 2024

2024 spring, Complied by ==骆春阳 工学院==



**说明：**

1）The complete process to learn DSA from scratch can be broken into 4 parts:

Learn about Time complexities, learn the basics of individual Data Structures, learn the basics of Algorithms, and practice Problems.

2）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

3）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

4）如果不能在截止前提交作业，请写明原因。



**编程环境**

操作系统：Windows 111

Python编程环境：PyCharm 2023.1.4



## 1. 题目

### 27638: 求二叉树的高度和叶子数目

http://cs101.openjudge.cn/practice/27638/



思路：

鉴于树的特征，所以叶子可以直接看输入有几个双-1

高度的话则需要创建完树之后递归一下，如果没有儿子那么就是节点的结束，否则加1

代码

```python
#n = int(input())
class tree_node:
    def __init__(self):
        self.left = None
        self.right = None

tree = [tree_node() for _ in range(n)]
parent = [False for _ in range(n)] ##用于判断是否有父节点
def count_height(i):
    if i is None: ##如果没有都没有儿子，那么就是一个节点的结束
        return -1
    return max(count_height(tree[i].left),count_height(tree[i].right)) + 1
result = 0
for i in range(n):
    left_index,right_index = map(int,input().split())
    if left_index != -1:
        tree[i].left = left_index
        parent[left_index] = True
    if right_index != -1:
        tree[i].right = right_index
        parent[right_index] = True
    if left_index == -1 and right_index == -1:
        result += 1
root = parent.index(False) ##找到根节点
print(count_height(root),result)


```



代码运行截图 ==（至少包含有"Accepted"）==

![image-20240319194046599](C:\Users\骆春阳\AppData\Roaming\Typora\typora-user-images\image-20240319194046599.png)

时间：25分钟

### 24729: 括号嵌套树

http://cs101.openjudge.cn/practice/24729/



思路：这题的两个难点在于处理数据输入和递归部分。

处理数据的时候，可以用栈来存放每一个临时的节点，之后对栈顶的节点进行相应的操作

递归的时候要理清extend，append，以及创建临时表的顺序

要反复练习这个题，做到熟练

代码

```python
# 输入
# A(B(E),C(F,G),D(H(I)))
# 输出
# ABECFGDHI 前序遍历
# EBFGCIHDA
s = list(input())
l = len(s)
class tree_node:
    def __init__(self,value):
        self.value = value ##存放字母
        self.children = []
def action_tree(s,l):
    stack = []
    node = None
    for i in s:
        if i.isalpha(): ##如果是字母
            node = tree_node(i)
            if stack: ##如果有父亲
                stack[-1].children.append(node) ##当前节点成为上一个节点的儿子
                # ##每一个字母都创建一个节点
        elif i == '(':
            if node: ##把当前的节点推入栈中
                stack.append(node)
            node = None
        elif i == ',':
            pass
        elif i == ')':
            if stack:
                node = stack.pop()
    return node ##最后弹出的是根节点

root = action_tree(s,l)

def front_order(node):
    result = [node.value]
    for i in node.children:
        result.extend(front_order(i))
    return ''.join(result)
def behind_order(node):
    R = []
    for i in node.children:
        R.extend(behind_order(i))
    R.append(node.value)
    return ''.join(R)
print(front_order(root))
print(behind_order(root))
 

```



代码运行截图 ==（至少包含有"Accepted"）==

![image-20240320172632261](C:\Users\骆春阳\AppData\Roaming\Typora\typora-user-images\image-20240320172632261.png)

时间：一个小时

### 02775: 文件结构“图”

http://cs101.openjudge.cn/practice/02775/



思路：树的排序过程其实不难，但是按顺序输出让我有点难办。

还有就是怎么退出输入。

最终看了大佬的代码，又学了一个exit包



代码

```python
from sys import exit

class dir:
    def __init__(self, dname):
        self.name = dname
        self.dirs = []
        self.files = []
    
    def getGraph(self):
        g = [self.name]
        for d in self.dirs:
            subg = d.getGraph()
            g.extend(["|     " + s for s in subg])
        for f in sorted(self.files):
            g.append(f)
        return g

n = 0
while True:
    n += 1
    stack = [dir("ROOT")]
    while (s := input()) != "*":
        if s == "#": exit(0)
        if s[0] == 'f':
            stack[-1].files.append(s)
        elif s[0] == 'd':
            stack.append(dir(s))
            stack[-2].dirs.append(stack[-1])
        else:
            stack.pop()
    print(f"DATA SET {n}:")
    print(*stack[0].getGraph(), sep='\n')
    print()

```

![image-20240325174019046](C:\Users\骆春阳\AppData\Roaming\Typora\typora-user-images\image-20240325174019046.png)



时间：60分钟（后来看大佬的代码）

### 25140: 根据后序表达式建立队列表达式

http://cs101.openjudge.cn/practice/25140/



思路：理解建立表达式树用了很长时间，然后宽度优先搜索在大佬的提醒下用了队列。



代码

```python
class tree_node:
    def __init__(self,value):
        self.value = value
        self.left = None
        self.right = None
def action(s):
    stack = []
    node = None ##创建一个临时节点
    for i in s:
        if i.upper() != i: ##小写字母
            node = tree_node(i)
            stack.append(node) ##放一个节点
        elif i.upper() == i: ##大写字母
            node = tree_node(i)
            node.right = stack.pop()
            node.left = stack.pop()
            stack.append(node)
    return stack.pop()
def bfs(root):
    deque = [root]
    result = []
    while deque:
        node = deque.pop(0)
        result.append(node.value)
        if node.left != None:
            deque.append(node.left)
        if node.right != None:
            deque.append(node.right)
    return result
n = int(input())
for _ in range(n):
    s = list(input())
    root = action(s)
    lst = (bfs(root))[::-1]
    print(''.join(lst))
```

![image-20240320205728861](C:\Users\骆春阳\AppData\Roaming\Typora\typora-user-images\image-20240320205728861.png)

时间：一个半小时





### 24750: 根据二叉树中后序序列建树

http://cs101.openjudge.cn/practice/24750/



思路： 开始没理解中序是怎么遍历的

后来用递归的想法发现还是很快就写出来了

主要就是根据中序遍历的特点，先从后序找到根节点，带入到中序遍历，找到左儿子和右儿子，反复如此

但是建树的思路还得多练，不够熟练



代码

```python
class tree_node:
    def __init__(self,value):
        self.value = value
        self.left = None
        self.right = None
def build(mid_order,behind_order):
    if not mid_order or not behind_order:
        return None
    root = behind_order.pop()
    root_node = tree_node(root)
    root_index = mid_order.index(root) #找到在中序表达式里root的位置
    root_node.right = build(mid_order[root_index+1:],behind_order)
    root_node.left = build(mid_order[:root_index],behind_order)
    return root_node
mid_order = list(input())
behind_order = list(input())
root = build(mid_order,behind_order)
def preorder(root):
    if root is None:
        return []
    result = [root.value]
    result.extend(preorder(root.left))
    result.extend(preorder(root.right))
    return result
print(''.join(preorder(root)))


```

![image-20240321115703569](C:\Users\骆春阳\AppData\Roaming\Typora\typora-user-images\image-20240321115703569.png)

时间：一个小时





### 22158: 根据二叉树前中序序列建树

http://cs101.openjudge.cn/practice/22158/



思路：和前面的中后序序列建树思路差不多



代码

```python
class tree_node:
    def __init__(self,value):
        self.value = value
        self.left = None
        self.right = None
def build(mid_order,pre_order):
    if not mid_order or not pre_order:
        return None
    root = pre_order[0]

    root_node = tree_node(root)
    root_index = mid_order.index(root) #找到在中序表达式里root的位置
    root_node.left = build(mid_order[:root_index],pre_order[1:root_index+1])
    root_node.right = build(mid_order[root_index+1:],pre_order[root_index+1:])
    return root_node

def preorder(root):
    if root is None:
        return []
    result = []
    result.extend(preorder(root.left))
    result.extend(preorder(root.right))
    result.append(root.value)
    return result
while True:
    try:
        pre_order = list(input())
        mid_order = list(input())
        root = build(mid_order,pre_order)
        print(''.join(preorder(root)))
    except EOFError:
        break



```



![image-20240321165949528](C:\Users\骆春阳\AppData\Roaming\Typora\typora-user-images\image-20240321165949528.png)

时间：40分钟左右



## 2. 学习总结和收获

==如果作业题目简单，有否额外练习题目，比如：OJ“2024spring每日选做”、CF、LeetCode、洛谷等网站题目。==

这一周开始对作业题的思路还不够清楚，后来慢慢自己做出来。

之后连着几天，都是把做过的树的题反复地重新做，建树遍历树越来越熟练了。

尤其对前序后序遍历的过程，递归的思路非常清晰了。

但是文件结构图还是没有自己弄出来，借鉴了大佬的

加油，这一周继续刷树。

（又看大佬代码学习了exit包）

ps：看大佬代码真的能让人思路清晰，以后做新题不能全靠自己想，应该想一会就看一看大佬的思路，试着临摹一下，然后再做同类型的题看能不能自己想出来
