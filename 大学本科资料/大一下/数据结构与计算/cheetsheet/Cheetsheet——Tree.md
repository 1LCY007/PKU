# Cheetsheet——Tree

Updated 2024.5.23	==工学院 骆春阳==

## 一、心得

本学期第一次接触树，但是不得不说是我最喜欢的一类算法。主要在于其让我第一次体会到python中定义类的魅力。

### 关键词

1.树的基本概念。（深度和高度不要弄混，笔试部分还是很常见的）

**层级 Level**：
从根节点开始到达一个节点的路径，所包含的==边的数量==，称为这个节点的层级。
如图 D 的层级为 2，根节点的层级为 0。

有时候，题目中会给出概念定义，如：

**高度 Height**：树中所有节点的==最大层级==称为树的高度，如图1所示树的高度为 2。

**二叉树深度**：从根结点到叶结点依次经过的结点（含根、叶结点）形成树的一条路径，==最长路径的节点个数==为树的深度

2.按形态分类（主要是二叉树）

（1）完全二叉树——第n-1层全满，最后一层按顺序排列

（2）满二叉树——二叉树的最下面一层元素全部满就是满二叉树

（3）avl树——平衡因子，左右子树高度差不超过1

​	= =这块一定要弄懂左右旋的概念，理解什么时候左旋什么时候右旋，以及操作的具体过程！

（4）二叉查找树(二叉排序\搜索树)
	= =特点：没有相同键值的节点。

​	= =若左子树不空，那么其所有子孙都比根节点小。

​	= =若右子树不空，那么其所有子孙都比根节点大。

​	= =左右子树也分别为二叉排序树。

（5）哈夫曼树——哈夫曼树是一种针对权值的二叉树。一般为了减少计算机运算速度，将权重大的放在最前面

### 技巧：

​	结合一些队列，栈，堆等思想，运用树会有奇效！

## 二、例题

### 1.哈弗曼编码

这段代码首先定义了一个 `Node` 类来表示哈夫曼树的节点。然后，使用最小堆来构建哈夫曼树，每次==从堆中取出两个频率最小的节点==进行合并，直到堆中只剩下一个节点，即哈夫曼树的根节点。接着，使用递归方法计算哈夫曼树的带权外部路径长度（weighted external path length）。最后，输出计算得到的带权外部路径长度。

你可以运行这段代码来得到该最优二叉编码树的带权外部路径长度。

```python
import heapq

class Node:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

def huffman_encoding(char_freq):
    heap = [Node(char, freq) for char, freq in char_freq.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = Node(None, left.freq + right.freq) # note: 合并之后 char 字典是空
        merged.left = left
        merged.right = right
        heapq.heappush(heap, merged)

    return heap[0]

def external_path_length(node, depth=0):
    if node is None:
        return 0
    if node.left is None and node.right is None:
        return depth * node.freq
    return (external_path_length(node.left, depth + 1) +
            external_path_length(node.right, depth + 1))

def main():
    char_freq = {'a': 3, 'b': 4, 'c': 5, 'd': 6, 'e': 8, 'f': 9, 'g': 11, 'h': 12}
    huffman_tree = huffman_encoding(char_freq)
    external_length = external_path_length(huffman_tree)
    print("The weighted external path length of the Huffman tree is:", external_length)

if __name__ == "__main__":
    main()

# Output:
# The weighted external path length of the Huffman tree is: 169 
```

#### 哈夫曼编码树

http://cs101.openjudge.cn/practice/22161/

```python
import heapq

class Node:
    def __init__(self, weight, char=None):
        self.weight = weight
        self.char = char
        self.left = None
        self.right = None

    def __lt__(self, other):
        if self.weight == other.weight:
            return self.char < other.char
        return self.weight < other.weight

def build_huffman_tree(characters):
    heap = []
    for char, weight in characters.items():
        heapq.heappush(heap, Node(weight, char))

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = Node(left.weight + right.weight) #note: 合并后，char 字段默认值是空
        merged.left = left
        merged.right = right
        heapq.heappush(heap, merged)

    return heap[0]

def encode_huffman_tree(root):
    codes = {}

    def traverse(node, code):
        if node.char:
            codes[node.char] = code
        else:
            traverse(node.left, code + '0')
            traverse(node.right, code + '1')

    traverse(root, '')
    return codes

def huffman_encoding(codes, string):
    encoded = ''
    for char in string:
        encoded += codes[char]
    return encoded

def huffman_decoding(root, encoded_string):
    decoded = ''
    node = root
    for bit in encoded_string:
        if bit == '0':
            node = node.left
        else:
            node = node.right

        if node.char:
            decoded += node.char
            node = root
    return decoded

# 读取输入
n = int(input())
characters = {}
for _ in range(n):
    char, weight = input().split()
    characters[char] = int(weight)

#string = input().strip()
#encoded_string = input().strip()

# 构建哈夫曼编码树
huffman_tree = build_huffman_tree(characters)

# 编码和解码
codes = encode_huffman_tree(huffman_tree)

strings = []
while True:
    try:
        line = input()
        if line:
            strings.append(line)
        else:
            break
    except EOFError:
        break

results = []
#print(strings)
for string in strings:
    if string[0] in ('0','1'):
        results.append(huffman_decoding(huffman_tree, string))
    else:
        results.append(huffman_encoding(codes, string))

for result in results:
    print(result)
```



### 2.二叉堆实现（最小堆）

1.每次插入元素从==最后插入==，然后进行调整堆的操作

2.每次删除元素从堆顶找元素，找到元素后将该元素和最后一个元素换位，然后进行==重排操作==

（这里要求的是删除最小元素，即堆顶元素）

http://cs101.openjudge.cn/practice/04078/

```python
class tree_node:
    def __init__(self):
        self.heap = []

    def parent(self, i):
        return (i - 1) // 2

    def left_child(self, i):
        return 2 * i + 1

    def right_child(self, i):
        return 2 * i + 2

    def swap(self, i, j):
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]

    def insert(self, item):
        self.heap.append(item)
        self.heapify_up(len(self.heap) - 1)

    def delete(self):
        if len(self.heap) == 0:
            raise IndexError("Heap is empty")

        self.swap(0, len(self.heap) - 1)
        min_value = self.heap.pop()
        self.heapify_down(0)
        return min_value

    def heapify_up(self, i):
        while i > 0 and self.heap[i] < self.heap[self.parent(i)]:
            self.swap(i, self.parent(i))
            i = self.parent(i)

    def heapify_down(self, i):
        min_index = i
        left = self.left_child(i)
        right = self.right_child(i)

        if left < len(self.heap) and self.heap[left] < self.heap[min_index]:
            min_index = left

        if right < len(self.heap) and self.heap[right] < self.heap[min_index]:
            min_index = right

        if i != min_index:
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

### 3.二叉搜索树的层次遍历



http://cs101.openjudge.cn/practice/05455/

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



### 4.avl树！！！==（重难点）==

https://sunnywhy.com/sfbj/9/5/359

#### 3.1、插入—— 左左型的右旋：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191122102755404.png)

由上图可知：在插入之前树是一颗AVL树，而插入结点之后，T的左右子树高度差的绝对值不再 <= 1，此时AVL树的平衡性被破坏，我们要对其进行旋转。

在结点T的 **左结点（L）** 的 **左子树（L）** 上做了插入元素的操作，我们称这种情况为 **左左型** ，我们应该进行右旋转。

**注：** T 表示 平衡因子(bf)大于1的节点。

##### 3.1.1、右旋的具体步骤：

- T向右旋转成为L的右结点
- L的右节点Y 放到 T的左孩子上

旋转中心是根节点T的左节点（L）。

这样即可得到一颗新的AVL树，旋转过程图如下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191122102924684.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3hpYW9qaW4yMWNlbg==,size_16,color_FFFFFF,t_70)

##### 3.1.2、右旋的动画演示：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190728064308414.gif)

##### 3.1.3、右旋示例：

**示例1：**

左左情况下，插入新数据1 时，进行右旋操作：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190728064331164.png)

**示例2：**

插入 节点2后，进行右旋转：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190728064349606.jpeg)

#### 3.2、插入——左右型的左右旋：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190823111508904.png)

由上图可知，我们在T结点的左结点的右子树上插入一个元素时，会使得根为T的树的左右子树高度差的绝对值不再 <= 1，如果只是进行简单的右旋，得到的树仍然是不平衡的。

在结点T的 **左结点（L）** 的 **右子树（R）** 上做了插入元素的操作，我们称这种情况为 **左右型** ，我们应该进行左右旋。

**注：** T 表示 平衡因子(bf)大于1的节点。

##### 3.2.1、左右旋的两次旋转步骤：

第1次是左旋转：

- L节点 左旋转，成为R的左节点
- R的左节点（Y1） 左旋转，成为 L的右节点（即左子节点左转）

第2次是右旋转：

- T节点 右旋转，成为R的右节点
- R的右节点（Y2）右旋转，成为T的左节点（即右子节点右转）

旋转中心是根节点 T 的左节点（R）。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191122104309314.png)

##### 3.2.2、左右旋转的示例：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191122100417493.png)

一定要先把上面的左左型、左右型的旋转搞明白了， 下面的右右型、右左型的旋转就容易理解了。

#### 3.3、插入——右右型的左旋：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190823111022851.png)

由上图可知：在插入之前树是一颗AVL树，而插入新结点之后，T的左右子树高度差的绝对值不再 <+ 1，此时AVL树的平衡性被破坏，我们要对其进行旋转。

在结点T的 **右结点（R）** 的 **右子树（R）** 上做了插入元素的操作，我们称这种情况为 **右右型** ，我们应该进行左旋转。

**注：** T 表示 平衡因子(bf)大于1的节点。

##### 3.3.1、左旋的具体步骤：

- T向左旋转成为R的左结点
- R的左节点Y放到T的右孩子上

旋转中心是根节点T的右节点（R）。

这样即可得到一颗新的AVL树，旋转过程图如下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190823111011943.png)

##### 3.3.2、动画演示：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190728064514413.gif)

##### 3.3.3、左旋举例：

**示例1：**

右右情况下，插入新数据时，左旋操作：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190728064535581.png)

**示例2：**
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190728064549810.jpeg)

#### 3.4、插入——右左型的右左旋：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190823114424449.png)

由上图可知，我们在T结点的右结点的左子树上插入一个元素时，会使得根为T的树的左右子树高度差的绝对值不再 < 1，如果只是进行简单的左旋，得到的树仍然是不平衡的。

在结点T的 **右结点（R）** 的 **左子树（L）** 上做了插入元素的操作，我们称这种情况为 **右左型** ，我们应该进行右左旋。

**注：** T 表示 平衡因子(bf)大于1的节点。

##### 3.4.1、右左旋的两次旋转步骤：

第1次是右旋转：

- R 节点 右旋转，成为L的右节点
- L的右节点（Y2） 右旋转，成为R的左节点（即右子节点右转）

第2次是左旋转：

- T 节点 左旋转，成为L的左节点
- L的左节点（Y1）左旋转，成为T的右节点 （即左子节点左转）

旋转中心是根节点 T 的右节点（L）。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190823115620684.png)

##### 3.4.2、右左旋的的示例

![在这里插入图片描述](https://img-blog.csdnimg.cn/2019112210022926.png)

#### 4、总结：

在插入的过程中，会出现一下四种情况破坏AVL树的特性，我们可以采取如下相应的旋转。

| 插入位置                                              | 状态   | 操作   |
| ----------------------------------------------------- | ------ | ------ |
| 在结点T的左结点（L）的 **左子树（L）** 上做了插入元素 | 左左型 | 右旋   |
| 在结点T的左结点（L）的 **右子树（R）** 上做了插入元素 | 左右型 | 左右旋 |
| 在结点T的右结点（R）的 **右子树（R）** 上做了插入元素 | 右右型 | 左旋   |
| 在结点T的右结点（R）的 **左子树（L）** 上做了插入元素 | 右左型 | 右左旋 |

```python
# 定义二叉树节点类
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
        self.height = 1  # 节点高度，默认为1

# 定义平衡二叉树类
class AVLTree:
    def __init__(self):
        self.root = None

    # 获取节点的高度
    def get_height(self, node):
        if node is None:
            return 0
        return node.height

    # 获取节点的平衡因子
    def get_balance(self, node):
        if node is None:
            return 0
        return self.get_height(node.left) - self.get_height(node.right)

    # 更新节点的高度
    def update_height(self, node):
        if node is None:
            return
        node.height = 1 + max(self.get_height(node.left), self.get_height(node.right))

    # 执行右旋转操作
    def rotate_right(self, z):
        y = z.left
        T3 = y.right

        # 执行旋转
        y.right = z
        z.left = T3

        # 更新节点高度
        self.update_height(z)
        self.update_height(y)

        return y

    # 执行左旋转操作
    def rotate_left(self, z):
        y = z.right
        T2 = y.left

        # 执行旋转
        y.left = z
        z.right = T2

        # 更新节点高度
        self.update_height(z)
        self.update_height(y)

        return y

    # 插入节点
    def insert(self, root, val):
        if root is None:
            return TreeNode(val)
        elif val < root.val:
            root.left = self.insert(root.left, val)
        else:
            root.right = self.insert(root.right, val)

        # 更新节点高度
        self.update_height(root)

        # 平衡二叉树调整
        balance = self.get_balance(root)

        # 左左情况，执行右旋转
        if balance > 1 and val < root.left.val:
            return self.rotate_right(root)

        # 右右情况，执行左旋转
        if balance < -1 and val > root.right.val:
            return self.rotate_left(root)

        # 左右情况，先左旋转再右旋转
        if balance > 1 and val > root.left.val:
            root.left = self.rotate_left(root.left)
            return self.rotate_right(root)

        # 右左情况，先右旋转再左旋转
        if balance < -1 and val < root.right.val:
            root.right = self.rotate_right(root.right)
            return self.rotate_left(root)

        return root

    # 删除节点
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

    # 获取最小值节点
    def get_min_value_node(self, root):
        if root is None or root.left is None:
            return root
        return self.get_min_value_node(root.left)

    # 先序遍历二叉树
    def pre_order_traversal(self, root):
        if root is None:
            return []
        if root is not None:
            result = [root.val]
            result.extend(self.pre_order_traversal(root.left))
            result.extend(self.pre_order_traversal(root.right))
            return result

# 示例用法
tree = AVLTree()
root = None
n = int(input())
for i in map(int,input().split()):
    root = tree.insert(root,i)
result = tree.pre_order_traversal(root)
print(*result)
```

在上述代码中，平衡二叉树的删除操作通过递归实现。下面是删除操作的具体实现步骤：

1. 首先，在 `delete()` 方法中，我们使用递归来找到要删除的节点。
2. 如果要删除的节点值小于当前节点的值，则递归调用 `delete()` 方法并将当前节点的左子树更新为删除后的结果。
3. 如果要删除的节点值大于当前节点的值，则递归调用 `delete()` 方法并将当前节点的右子树更新为删除后的结果。
4. 如果要删除的节点值等于当前节点的值，则执行以下步骤：
   - 如果要删除的节点没有左子树或右子树，那么我们可以直接删除该节点，然后返回它的非空子树（如果存在）。
   - 如果要删除的节点有两个子节点，我们需要找到它的右子树中的最小节点（后继节点），将该最小节点的值赋给当前节点，并递归地删除该最小节点。
5. 在删除节点后，我们需要更新节点的高度。
6. 最后，我们执行平衡二叉树的调整以保持平衡性。根据删除操作导致的不平衡情况，可以进行以下旋转操作：
   - 左左情况：执行右旋转。
   - 左右情况：先对当前节点的左子节点进行左旋转，然后再对当前节点进行右旋转。
   - 右右情况：执行左旋转。
   - 右左情况：先对当前节点的右子节点进行右旋转，然后再对当前节点进行左旋转。

上述步骤确保了在删除节点后保持平衡二叉树的性质。

#### 关键点：

1. 更新平衡因子

2. 插入元素后，即使调整为平衡位置

3. 如何调整：左旋，右旋

4. 删除的操作

   ···





### 4.并查集

并查集是一种用于维护集合（组）的数据结构，它通常用于解决一些离线查询、动态连通性和图论等相关问题。

其中最常见的应用场景是解决图论中的连通性问题，例如判断图中两个节点是否连通、查找图的连通分量、判断图是否为一棵树等等。并查集可以快速地合并两个节点所在的集合，以及查询两个节点是否属于同一个集合，从而有效地判断图的连通性。

并查集还可以用于解决一些离线查询问题，例如静态连通性查询和==最小生成树问题==，以及一些==动态连通性问题==，例如支持动态加边和删边的连通性问题。


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
```



### 5.树的遍历

**前序遍历**
在前序遍历中，先访问根节点，然后递归地前序遍历左子树，最后递归地前序遍历右子树。

**中序遍历**
在中序遍历中，先递归地中序遍历左子树，然后访问根节点，最后递归地中序遍历右子树。

**后序遍历**
在后序遍历中，先递归地后序遍历左子树，然后递归地后序遍历右子树，最后访问根节点。

#### ==代码

```python
def preorder(root): ##前序遍历
    if root is None:
        return []
    result = [root.value]
    result.extend(preorder(root.left))
    result.extend(preorder(root.right))
    return result

##改变result的位置便可以把前序变成中序，变成后序
    
def behindorder(root): ##后序遍历
    if root is None:
        return []
    result = []
    result.extend(behindorder(root.left))
    result.extend(behindorder(root.right))
    result.append(root.value)
    return result
```





