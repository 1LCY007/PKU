# Cheetsheet——Tree

## 1.哈夫曼编码

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

### 哈夫曼编码树

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



## 2.二叉堆实现（最小堆）

1.每次插入元素从最后插入，然后进行调整堆的操作

2.每次删除元素从堆顶找元素，找到元素后将该元素和最后一个元素换位，然后进行重排操作

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

## 3.二叉搜索树的层次遍历



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



## avl树

https://sunnywhy.com/sfbj/9/5/359

AVL树实现映射抽象数据类型的方式与普通的二叉搜索树一样，唯一的差别就是性能。实现AVL树时，要记录每个节点的平衡因子。我们通过查看每个节点左右子树的高度来实现这一点。更正式地说，我们将平衡因子定义为左右子树的高度之差。

$balance Factor = height (left SubTree) - height(right SubTree)$

根据上述定义，如果平衡因子大于零，我们称之为左倾；如果平衡因子小于零，就是右倾；如果平衡因子等于零，那么树就是完全平衡的。为了实现AVL树并利用平衡树的优势，我们将平衡因子为-1、0和1的树都定义为平衡树。一旦某个节点的平衡因子超出这个范围，我们就需要通过一个过程让树恢复平衡。图6-26展示了一棵右倾树及其中每个节点的平衡因子。

我们已经证明，保持AVL树的平衡会带来很大的性能优势，现在看看如何往树中插入一个键。所有新键都是以叶子节点插入的，因为新叶子节点的平衡因子是零，所以新插节点没有什么限制条件。但插入新节点后，必须更新父节点的平衡因子。新的叶子节点对其父节点平衡因子的影响取决于它是左子节点还是右子节点。如果是右子节点，父节点的平衡因子减一。如果是左子节点，则父节点的平衡因子加一。这个关系可以递归地应用到每个祖先，直到根节点。既然更新平衡因子是递归过程，就来检查以下两种基本情况：

❏ 递归调用抵达根节点；
❏ 父节点的平衡因子调整为零；可以确信，如果子树的平衡因子为零，那么祖先节点的平衡因子将不会有变化。

我们将AVL树实现为BinarySearchTree的子类。首先重载\_put方法，然后新写updateBalance辅助方法，如代码清单6-37所示。可以看到，除了在第10行和第16行调用updateBalance以外，\_put方法的定义和代码清单6-25中的几乎一模一样。

要理解什么是旋转，来看一个简单的例子。考虑图6-28中左边的树。这棵树失衡了，平衡因子是-2。要让它恢复平衡，我们围绕以节点A为根节点的子树做一次左旋。

![../_images/simpleunbalanced.png](https://raw.githubusercontent.com/GMyhf/img/main/img/simpleunbalanced.png)

图6-28 通过左旋让失衡的树恢复平衡

本质上，左旋包括以下步骤。

❏ 将右子节点（节点B）提升为子树的根节点。
❏ 将旧根节点（节点A）作为新根节点的左子节点。
❏ 如果新根节点（节点B）已经有一个左子节点，将其作为新左子节点（节点A）的右子节点。注意，因为节点B之前是节点A的右子节点，所以此时节点A必然没有右子节点。因此，可以为它添加新的右子节点，而无须过多考虑。

左旋过程在概念上很简单，但代码细节有点复杂，因为需要将节点挪来挪去，以保证二叉搜索树的性质。另外，还要保证正确地更新父指针。

左旋后得到另一棵失衡的树，如图6-32所示。如果在此基础上做一次右旋，就回到了图6-31的状态。

![../_images/badrotate.png](https://raw.githubusercontent.com/GMyhf/img/main/img/badrotate.png)

图6-32 左旋后，树朝另一个方向失衡

要解决这种问题，必须遵循以下规则。
❏ 如果子树需要左旋，首先检查右子树的平衡因子。如果右子树左倾，就对右子树做一次右旋，再围绕原节点做一次左旋。
❏ 如果子树需要右旋，首先检查左子树的平衡因子。如果左子树右倾，就对左子树做一次左旋，再围绕原节点做一次右旋。
图6-33展示了如何通过以上规则解决图6-31和图6-32中的困境。围绕节点C做一次右旋，再围绕节点A做一次左旋，就能让子树恢复平衡。





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

### 关键点：

1. 更新平衡因子

2. 插入元素后，即使调整为平衡位置

3. 如何调整：左旋，右旋

4. 删除的操作

   ···





## 4.并查集

并查集是一种用于维护集合（组）的数据结构，它通常用于解决一些离线查询、动态连通性和图论等相关问题。

其中最常见的应用场景是解决图论中的连通性问题，例如判断图中两个节点是否连通、查找图的连通分量、判断图是否为一棵树等等。并查集可以快速地合并两个节点所在的集合，以及查询两个节点是否属于同一个集合，从而有效地判断图的连通性。

并查集还可以用于解决一些离线查询问题，例如静态连通性查询和最小生成树问题，以及一些动态连通性问题，例如支持动态加边和删边的连通性问题。


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



## 5.树的遍历

**前序遍历**
在前序遍历中，先访问根节点，然后递归地前序遍历左子树，最后递归地前序遍历右子树。

**中序遍历**
在中序遍历中，先递归地中序遍历左子树，然后访问根节点，最后递归地中序遍历右子树。

**后序遍历**
在后序遍历中，先递归地后序遍历左子树，然后递归地后序遍历右子树，最后访问根节点。

### ==代码

```python
def preorder(root): ##前序遍历
    if root is None:
        return []
    result = [root.value]
    result.extend(preorder(root.left))
    result.extend(preorder(root.right))
    return result
    
    
def behindorder(root): ##后序遍历
    if root is None:
        return []
    result = []
    result.extend(behindorder(root.left))
    result.extend(behindorder(root.right))
    result.append(root.value)
    return result
```





