# Assignment #4: 排序、栈、队列和树

Updated 0005 GMT+8 March 11, 2024

2024 spring, Complied by ==骆春阳 工学院==



**说明：**

1）The complete process to learn DSA from scratch can be broken into 4 parts:

Learn about Time complexities, learn the basics of individual Data Structures, learn the basics of Algorithms, and practice Problems.

2）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

3）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

4）如果不能在截止前提交作业，请写明原因。



**编程环境**

操作系统：Windows 11

Python编程环境：PyCharm 2023.1.4 (Professional Edition)



## 1. 题目

### 05902: 双端队列

http://cs101.openjudge.cn/practice/05902/



思路：用Python的deque包实现



代码

```python
from collections import deque
n = int(input())
def action(judge,number,data):
    if judge == 1:
        data.append(number)
        return data
    if judge == 2 and data:
        if number == 0:
            data.popleft()
            return data
        if number == 1:
            data.pop()
            return data
    if not data:
        return
for _ in range(n):
    t = int(input())
    data = deque()
    for i in range(t):
        judge,number = map(int,input().split())
        action(judge,number,data)
    if data:
        print(*data)
    else:
        print('NULL')

```



![image-20240315194616510](C:\Users\骆春阳\AppData\Roaming\Typora\typora-user-images\image-20240315194616510.png)

时间：10分钟



### 02694: 波兰表达式

http://cs101.openjudge.cn/practice/02694/



思路：递归，栈



代码

```python
import math
lst = list(input().split())
stack = [] ##储存数字
dic = '+-*/'
n = len(lst) - 1
def count(n,lst,stack):
    if n < 0:
        return stack[0]
    if lst[n] not in dic:
        stack.append(lst[n])
        return count(n-1,lst,stack)
    if lst[n] in dic:
        right = stack.pop()
        left = stack.pop()
        stack.append(eval(f'{right} {lst[n]} {left}' ))
        return count(n-1,lst,stack)
print(f'{count(n,lst,stack):.6f}')

```

![image-20240315200816705](C:\Users\骆春阳\AppData\Roaming\Typora\typora-user-images\image-20240315200816705.png)

时间： 15分钟



### 24591: 中序表达式转后序表达式

http://cs101.openjudge.cn/practice/24591/



思路：栈，这题是自己写的，自己想明白了原理很开心。

（详情见代码）



代码

```python
## 第一个问题是如何处理运算符，
## 比如 1 * 2 + 3 那么就应该输出 1 2 * 3 +
## 如果是 1 + 2 * 3 那么就应该输出 1 2 3 * +
## 所以检测到运算符后，如果上一个运算符的优先级高于或等于它，就先把上一个运算法输出
## 第二个问题是如何处理括号
## 括号的优先级高于所有，所以遇到括号，一定要先输出括号内的所有元素
n = int(input())
dic = {'+': 1, '-': 1, '*': 2, '/': 2, '(': 0, ')': 0}
def recursion(lst,s,l,result,stack_action):
    ##注意if的顺序！！！
    if l == len(lst):
        if s != '':
            result .append(s)
        while stack_action:
            result.append(stack_action.pop())
        return result
    if lst[l] not in dic: ##如果是数字
        s += lst[l]
        return recursion(lst,s,l+1,result,stack_action)
    if lst[l] in dic :
        ##这里一定要先把数字放进去
        if s != '':
            result.append(s)
        s = ''  ##把数字放进去
        if lst[l] == '(':
            stack_action.append('(')
            return recursion(lst, s, l + 1, result, stack_action)
        if lst[l] == ')':
            while stack_action and stack_action[-1] != '(':
                result.append(stack_action.pop())

            stack_action.pop()
            return recursion(lst, s, l + 1, result, stack_action)
        #先看括号
        if stack_action:
            while stack_action and dic.get(stack_action[-1]) >= dic.get(lst[l]) :
                result.append(stack_action.pop())
            stack_action.append(lst[l])
            return recursion(lst, s, l + 1, result, stack_action)
        else:
            stack_action.append(lst[l])
            return recursion(lst,s,l+1,result,stack_action)

for _ in range(n):
    lst = list(input())
    l = 0
    stack_action = []  # 存放符号
    result = []  # 最终结果
    s = ''
    print(*recursion(lst,s,l,result,stack_action))
```

![image-20240317153803707](C:\Users\骆春阳\AppData\Roaming\Typora\typora-user-images\image-20240317153803707.png)

时间：五十分钟左右





### 22068: 合法出栈序列

http://cs101.openjudge.cn/practice/22068/



思路：（经同学指点）

一个数只有在栈顶的时候才能弹出来，当后面的数进入的时候就没法先弹出它



代码

```python
##模拟：
## 一个数只有在栈顶的时候才能弹出来
## 后面的数来的时候，它就弹不出去了
answer = input()
def judge(answer,s):
    if len(answer) != len(s): ##先检查长度
        print('NO')
        return
    stack = []
    lst = list(answer)
    for i in s:
        while (not stack or stack[-1] != i) and lst : ##如果栈空了或者不是栈顶元素
            stack.append(lst.pop(0))
        if (not stack or stack[-1] != i):
            print('NO')
            return
        stack.pop() ##匹配成功就可以删了
    print('YES')
    return

while True:
    try:
        s = input()
        judge(answer,s)
    except EOFError:
        break

```

![image-20240317185419811](C:\Users\骆春阳\AppData\Roaming\Typora\typora-user-images\image-20240317185419811.png)

时间：40分钟（苦思冥想）





### 06646: 二叉树的深度

http://cs101.openjudge.cn/practice/06646/



思路：用树实现，递归一下，每个节点都看一下子节点的情况。



代码

```python
class tree_node:
    def __init__(self):
        self.left = None
        self.right = None
def tree_depth(tree):
    if tree is None:
        return 0
    left_depth = tree_depth(tree.left)
    right_depth = tree_depth(tree.right)
    return max(left_depth, right_depth) + 1

n = int(input()) ##节点的数量
tree = [tree_node() for _ in range(n)]
for i in range(n):
    L,R = map(int,input().split())
    if L != -1:
        tree[i].left = tree[L-1]
    if R != -1:
        tree[i].right = tree[R-1]
root = tree[0]
depth = tree_depth(root)
print(depth)

```



![image-20240318185701512](C:\Users\骆春阳\AppData\Roaming\Typora\typora-user-images\image-20240318185701512.png)

20分钟

### 02299: Ultra-QuickSort

http://cs101.openjudge.cn/practice/02299/



思路：



代码

```python
# 

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==





## 2. 学习总结和收获

==如果作业题目简单，有否额外练习题目，比如：OJ“2024spring每日选做”、CF、LeetCode、洛谷等网站题目。==

自己学习了一下链表，链表学了挺久，主要对各个元素值间用链的连接不太理解。后来想清楚。

然后在一个学过计算机竞赛的大佬的教学下终于理解了树，包括树的遍历、二叉树的特殊性、以及课件上的题。

（当然自己写还有些费事，不过可以开始刷每日选做的树了）

对类的理解不够深刻，但是发现Python中的类真的太神了，真的好用。

本周最开心的一件事，自己把中序转后序表达式写了出来，然后大佬又教了我合法出栈序列的思路。

每个算法的独特的特点就是题目的突破口，比如合法出栈序列，要知道一个数字弹出只能在栈顶弹出，其实抓好这一个特点开始想问题就会轻松了。再比如二叉树的特点让它方便递归等等。



ps:本周收获最大的一点，找到了一个院内的大佬学习，虽然他学的是c，但是算法上真的给我提供了很多帮助。

有大腿可以抱了嘿嘿嘿



