# Assignment #6: "树"算：Huffman,BinHeap,BST,AVL,DisjointSet

Updated 2214 GMT+8 March 24, 2024

2024 spring, Complied by ==骆春阳 工学院==



**说明：**

1）这次作业内容不简单，耗时长的话直接参考题解。

2）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

3）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

4）如果不能在截止前提交作业，请写明原因。



**编程环境**

操作系统：Windows 11

Python编程环境：PyCharm 2023.1.4 





## 1. 题目

### 22275: 二叉搜索树的遍历

http://cs101.openjudge.cn/practice/22275/



思路：刚开始思路很古板，还想用类建树解决

后来发现其实递归一下就可以



代码

```python
def recursion(lst):
    if not lst:
        return []
    root = lst[0]
    root_left = [x for x in lst if x < root]
    root_right = [x for x in lst if x > root]
    return recursion(root_left) + recursion(root_right) + [str(root)]

n = int(input())
lst = list(map(int,input().split()))
result = recursion(lst)
print(*result)

```

![image-20240326232252075](C:\Users\骆春阳\AppData\Roaming\Typora\typora-user-images\image-20240326232252075.png)

时间：20分钟





### 05455: 二叉搜索树的层次遍历

http://cs101.openjudge.cn/practice/05455/



思路：每一次插入的时候，从根节点开始比较（递归）



代码

```python
class tree_node:
    def __init__(self,value):
        self.value = value
        self.left = None
        self.right = None
def insert_tree(node,value):
     if node is None:
         return tree_node(value)
     if node.value > value:  ##如果比根节点小
         node.left = insert_tree(node.left,value) ##和根节点的左节点继续比较
     elif node.value < value:
         node.right = insert_tree(node.right,value)
     return node
def recursion(node):
    deque = [node]
    result = []
    while deque:
        x = deque.pop(0)
        if x.left is not None:
            deque.append(x.left)
        if x.right is not None:
            deque.append(x.right)
        result.append(x.value)
    return result
lst = list(map(int,input().split()))
node = None
data = set()
for i in lst:
    if i not in data:
        data.add(i)
        node = insert_tree(node,i)
print(*recursion(node))

```

![image-20240328112729206](C:\Users\骆春阳\AppData\Roaming\Typora\typora-user-images\image-20240328112729206.png)

时间：40分钟





### 04078: 实现堆结构

http://cs101.openjudge.cn/practice/04078/

练习自己写个BinHeap。当然机考时候，如果遇到这样题目，直接import heapq。手搓栈、队列、堆、AVL等，考试前需要搓个遍。



思路：

第一关键点在于，完全二叉树的规律。子节点和父节点间存在着稳定的代数关系。

第二关键在于，每次插入的时候，应该插入到堆底，之后向上。

​	那么这里就需要一个up的函数，反复递归地和父节点相比较。

第三关键在于，每次删除的时候，删除堆顶元素。

​	但是这样的话，堆顶就空了，如果要保持堆的结构稳定。

​	就要先把最后的一个元素放到堆顶，然后堆他排序。

​	那么这里就需要一个down的函数，反复递归地和子节点相比较。



代码

```python
#class tree_node:
    def __init__(self):
        self.heap = []

    def parent(self, i):
        return (i - 1) // 2

    def left_child(self, i):
        return 2 * i + 1

    def right_child(self, i):
        return 2 * i + 2

    def swap(self, i, j):
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i] ##一次交换操作

    def insert(self, item):  ##添加，没加入一次就比较一下
        self.heap.append(item)
        self.heapify_up(len(self.heap) - 1) ##插入的元素在最后面，让它挪上去

    def delete(self):
        if len(self.heap) == 0:
            raise IndexError("Heap is empty")

        self.swap(0, len(self.heap) - 1)
        min_value = self.heap.pop()
        self.heapify_down(0)
        return min_value

    def heapify_up(self, i):
        while i > 0 and self.heap[i] < self.heap[self.parent(i)]: ##当前节点比父节点小
            self.swap(i, self.parent(i))
            i = self.parent(i)

    def heapify_down(self, i):
        min_index = i
        left = self.left_child(i)
        right = self.right_child(i)
			#完全二叉树，所以先左后右的比较
        if left < len(self.heap) and self.heap[left] < self.heap[min_index]:
            min_index = left

        if right < len(self.heap) and self.heap[right] < self.heap[min_index]:
            min_index = right

        if i != min_index:  ##判断是否交换过
            self.swap(i, min_index)
            self.heapify_down(min_index)
n = int(input())
lst = tree_node()
for _ in range(n):
    s = input()
    if s[0] == '1':

        lst.insert(int(s[2:]))
    if s[0] == '2':
        print(lst.delete())
```



![image-20240328173403905](C:\Users\骆春阳\AppData\Roaming\Typora\typora-user-images\image-20240328173403905.png)

时间：很长

（代码没有思路，然后找gpt学习，之后两三天反复自己试着搓堆，搓出来了）

### 22161: 哈夫曼编码树

http://cs101.openjudge.cn/practice/22161/



思路：又听大佬讲了一遍，但是自己实现的时候出了很多问题

后来理解了哈夫曼树的精妙之处。

1. 自定义lt方法来比较
2. 递归合并权值，构建树
3. 将树的叶子结点储存，变为解码器

代码

```python
import heapq
class tree_node:
    def __init__(self,weight,char = None):
        self.weight = weight
        self.char = char
        self.left = None
        self.right = None

    def __lt__(self,other):
        if self.weight == other.weight:
            return self.char < other.char
        return self.weight < other.weight

def build_tree(dic):
    heap = []
    for key,value in dic.items():
        heapq.heappush(heap,tree_node(value,key))
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merge = tree_node(left.weight + right.weight) ##char是None
        ##合并两个节点
        merge.left = left
        merge.right = right
        ##将合并的节点与左儿子和右儿子连接起来
        heapq.heappush(heap,merge)
    return heap[0]  ##得到根节点

def encode_tree(root): ##解码
    codes = {}
    def trans(node,string):
        if node.char:
            codes[node.char] = string ##记录一下每个字符的编码
        else:
            trans(node.left,string + '0')
            trans(node.right,string + '1')

    trans(root,'')
    return codes

def encode_action(root,string):
    decode = ''
    node = root
    for i in string:
        if i == '0':
            node = node.left
        else:
            node = node.right
        if node.char:
            decode += node.char ##找到对应字符就输出
            node = root
    return decode

def code_action(string,codes):
    decode = ''
    for i in string:
        decode += codes[i]
    return decode
##处理输入
n = int(input())
dic = {}
for _ in range(n):
    char,weight = input().split()
    dic[char] = int(weight)
root = build_tree(dic)
codes = encode_tree(root)
s = []
while True:
    try:
        s.append(input())
    except EOFError:
        break
result = []
for string in s:
    if string[0] in '01':
        result.append(encode_action(root,string))
    else:
        result.append(code_action(string,codes))
for i in result:
    print(i)
```



![image-20240328231948999](C:\Users\骆春阳\AppData\Roaming\Typora\typora-user-images\image-20240328231948999.png)



时间：60分钟

### 晴问9.5: 平衡二叉树的建立

https://sunnywhy.com/sfbj/9/5/359



思路：找gpt学习了，然后同时左旋右旋的部分没太理解，在b站上看了一会，看懂了

感慨平衡二叉树的美妙，太美丽了，递归得漂亮。

同时gpt还为我补上了搜索最小节点值和删除。搜索最小节点很简单，但是删除那里我想了很久，最后看明白很精妙。

这里补上删除的代码。

```python
    def delete(self, root, val):
        if root is None:
            return root
        elif val < root.val:
            root.left = self.delete(root.left, val)
        elif val > root.val:
            root.right = self.delete(root.right, val)
        else:
            # 找到要删除的节点
            if root.left is None:
                temp = root.right
                root = None
                return temp
            elif root.right is None:
                temp = root.left
                root = None
                return temp
            else:
                # 有两个子节点的情况，找到右子树中的最小节点
                temp = self.get_min_value_node(root.right)
                root.val = temp.val
                root.right = self.delete(root.right, temp.val)

        # 更新节点高度
        self.update_height(root)

        # 平衡二叉树调整
        balance = self.get_balance(root)

        # 左左情况，执行右旋转
        if balance > 1 and self.get_balance(root.left) >= 0:
            return self.rotate_right(root)

        # 左右情况，先左旋转再右旋转
        if balance > 1 and self.get_balance(root.left) < 0:
            root.left = self.rotate_left(root.left)
            return self.rotate_right(root)

        # 右右情况，执行左旋转
        if balance < -1 and self.get_balance(root.right) <= 0:
            return self.rotate_left(root)

        # 右左情况，先右旋转再左旋转
        if balance < -1 and self.get_balance(root.right) > 0:
            root.right = self.rotate_right(root.right)
            return self.rotate_left(root)

        return root

```

太美妙了

通过反复的比较来找到要删除的结点

并从这个结点的右边开始找（因为右边的值肯定比左边大）

那么我只需要右子树找到一个最小值替换，就不会改变当前树的整体结构





代码

```python
class tree_node:
    def __init__(self,value):
        self.value = value
        self.left = None
        self.right = None
        self.height = 1  ## 每个节点储存一下高度，方便比较
class AVL_tree:
    def __init__(self):
        self.root = None

    def _get_height(self,root):
        if root is None:
            return 0
        return root.height

    def _get_balance(self,root):
       if root is None:
           return 0
       return self._get_height(root.left) - self._get_height(root.right)

    def _update_height(self,node):
        if node is None:
            return 0
        node.height = max(self._get_height(node.left),self._get_height(node.right)) + 1

    def _rotate_left(self,root): ##左旋
        root_2 = root.right ##右节点要替换原来的根节点
        root_2_left = root_2.left ##新根的临时左节点

        root_2.left = root
        root.right = root_2_left
        self._update_height(root)
        self._update_height(root_2)
        return root_2 ##输出新的根节点

    def _rotate_right(self, root):  ##右旋
        root_2 = root.left  ##左节点要替换原来的根节点
        root_2_right = root_2.right  ##新根的临时右节点

        root_2.right = root
        root.left= root_2_right
        self._update_height(root)
        self._update_height(root_2)
        return root_2  ##输出新的根节点

    def __insert__(self, root, value):
        if root is None:
            return tree_node(value)
        elif root.value < value: ##比当前节点大，往右走
            root.right = self.__insert__(root.right,value) ##递归的插入，当前节点右节点
        elif root.value > value:
            root.left = self.__insert__(root.left,value)

        self._update_height(root)
        balance = self._get_balance(root)

        ##判断是否进行左旋右旋的操作
        if balance > 1 and value < root.left.value: ##LL
            return self._rotate_right(root)

        if balance < -1 and value > root.right.value: ##RR
            return self._rotate_left(root)

        if balance > 1 and value > root.left.value: ##LR 先左旋再右旋
            root.left = self._rotate_left(root.left)
            return self._rotate_right(root)

        if balance < -1 and value < root.right.value: ##RL 先右旋再左旋
            root.right = self._rotate_right(root.right)
            return self._rotate_left(root)

        return root

    def __pre_order__(self,root):

        if root is None:
            return []
        if root is not None:
            result = [root.value]
            result.extend(self.__pre_order__(root.left))
            result.extend(self.__pre_order__(root.right))
            return result

n = int(input())
root = None
tree = AVL_tree()
for i in map(int,input().split()):
    root = tree.__insert__(root,i)
result = tree.__pre_order__(root)
print(*result)

```

==

![image-20240330153352987](C:\Users\骆春阳\AppData\Roaming\Typora\typora-user-images\image-20240330153352987.png)

时间：3个小时？反正很长



### 02524: 宗教信仰

http://cs101.openjudge.cn/practice/02524/



思路：并查集的思想



代码

```python
class disj_set:
    def __init__(self,n):
        self.rank = [1 for i in range(n)]
        self.parent = [i for i in range(n)]

    def find(self,x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]


    def union(self,x,y):
        x_root = self.find(x)
        y_root = self.find(y)

        if x_root == y_root:
            return

        if self.rank[x_root] > self.rank[y_root]:
            self.parent[y_root] = x_root
        elif self.rank[y_root] < self.rank[x_root]:
            self.parent[x_root] = y_root
        else:
            self.parent[y_root] = x_root
            self.rank[x_root] += 1
case = 1
while True:
    n,m = map(int,input().split())
    d_set = disj_set(n)
    if n == 0 and m == 0:
        break
    for _ in range(m):
        s1,s2 = map(int,input().split())
        d_set.union(s1-1,s2-1)
    result = d_set.parent
    R = 0
    for i in range(len(result)):
        if result[i] == i:
            R += 1
    print(f'Case {case}: {R}')
    case += 1

```

![image-20240331164945894](C:\Users\骆春阳\AppData\Roaming\Typora\typora-user-images\image-20240331164945894.png)

时间：50分钟（理解并查集理解了一会）



## 2. 学习总结和收获

==如果作业题目简单，有否额外练习题目，比如：OJ“2024spring每日选做”、CF、LeetCode、洛谷等网站题目。==

二叉搜索树，快捷方便酷。

哈夫曼之树，让人头疼哭。

再说平衡树，电脑前呜呜。

并查集更畜，大脑将要卒。

反复手搓树，绩点要落幕。

递归树算图，期中劝退噜。

---------------------------------

虽说心中苦，但是快乐足。

区区编码树，思想要清楚。

导入堆的库，很快递归出。

平衡二叉树，左右旋转酷。

递归复递归，美丽多妙处。

只需耐心足，多练便明悟。

