# Cheetsheet_2(算法)

## 1.二分查找

### 时间复杂度

对于包含n个元素的列表，二分查找最多需要log~2~n步，简单查找最多需要n步。

### 原理

输入一个==有序的==元素列表。

如果要查找的元素包含在列表中，二分查找返回其位置，否则返回NULL。

### e.g.

要查找某个单词，直到单词开头字母为m，会从字典的中间开始翻找。

### 关于排序

#### 归并排序：

```python
def mergeSort(arr):
	if len(arr) > 1:
		mid = len(arr)//2

		L = arr[:mid]	# Dividing the array elements
		R = arr[mid:] # Into 2 halves

		mergeSort(L) # Sorting the first half
		mergeSort(R) # Sorting the second half

		i = j = k = 0
		# Copy data to temp arrays L[] and R[]
		while i < len(L) and j < len(R):
			if L[i] <= R[j]:
				arr[k] = L[i]
				i += 1
			else:
				arr[k] = R[j]
				j += 1
			k += 1

		# Checking if any element was left
		while i < len(L):
			arr[k] = L[i]
			i += 1
			k += 1

		while j < len(R):
			arr[k] = R[j]
			j += 1
			k += 1


if __name__ == '__main__':
	arr = [12, 11, 13, 5, 6, 7]
	mergeSort(arr)
	print(' '.join(map(str, arr)))
# Output: 5 6 7 11 12 13
```

#### 桶排序：

1. 根据待排序集合中最大元素和最小元素的差值范围和映射规则，确定申请的桶个数；

2. 遍历排序序列，将每个元素放到对应的桶里去；

3. 对不是空的桶进行排序；

4. 按顺序访问桶，将桶中的元素依次放回到原序列中对应的位置，完成排序。

   ```python
   
   from typing import List
   
   def bucket_sort(arr:List[int]):
       """桶排序"""
       min_num = min(arr)
       max_num = max(arr)
       # 桶的大小
       bucket_range = (max_num-min_num) / len(arr)
       # 桶数组
       count_list = [ [] for i in range(len(arr) + 1)]
       # 向桶数组填数
       for i in arr:
           count_list[int((i-min_num)//bucket_range)].append(i)
       arr.clear()
       # 回填，这里桶内部排序直接调用了sorted
       for i in count_list:
           for j in sorted(i):
               arr.append(j)
   ```

   





## 2.栈

### 原理：

有一系列对象组成的集合，这些对象的插入和删除操作遵循后进先出（LIFO）的原则。

### 代码及语法：

```python
Class Stack: ##定义一个栈类
	def __init__(self):  ##初始化栈
		self.items = []
	
	def isEmpty(self):  #判断是否为空
		return self.items == []
    
    def push(self,item): #进栈
    	self.items.append(item)

	def pop(self): #出栈
		return self.items.pop()
	
	def peek(self):  
		return self.items[len(self.items)-1]
	
	def size(slef):  # 获取大小
		return len(self.items)
   
if __name__ == "__main__":
    opstack = Stack()  ## 建立一个空栈
    a = [6,5,4]
    opstack.push(a)
    if not opstack.isEmpty():
       print(opstack.pop())

```



### e.g.:

括号匹配问题

## 3.队列：

### 原理：

由一系列对象组成的集合，这些对象的插入和删除遵循先进先出的原则。

元素可以在任何时刻进行插入，但是只有处在队列最前面的元素才能被删除。

### 代码及语法：

```python
from collections import deque
data = deque() ##创建一个队列
data = deque(maxlen = 3)  ##指定最大容纳数
##当列表中元素个数超过最大值的时候，会自动删除最老的那个元素
data.appendleft(1) ##从左边添加元素

data.clear()
data.copy()  ##深度拷贝
data.count('value')

data2 = deque()
data.extend(data2)
##两个队列合并，extend(value),  value的值可以是deque对象也可以是可迭代的对象，字符串，列表，元组等等

data.extendleft() ##value的值，也是反着来的，注意看下面的打印，从左侧开始往里面加


data.index(value,start=None,end=None)
data.insert(index,value)
data.pop()
data.popleft() ##注意从队列左侧删除或者添加元素的时间复杂度为O(n)
data.remove(value)
data.reverse()


data1 = deque('12345')
data1.rotate(3)  ##队列中的元素向右旋转3个
print(data1)
data1.rotate(-3) ##向左旋转3个
print(data1)

#输出结果
#deque(['3', '4', '5', '1', '2'])
#deque(['1', '2', '3', '4', '5'])
```



## 4.dfs(图) （暂时还没会）

### 原理：

DFS 是一种递归或栈（堆栈）数据结构的算法，用于图的遍历。

从一个起始节点开始，尽可能深入图的分支，直到无法继续深入，然后回溯并探索其他分支。

通过标记已访问的节点来避免重复访问。

### 步骤：

1.创建一个空的栈（Stack）数据结构，用于存储待访问的节点。

2.从起始节点开始，将其标记为已访问并入栈。

3.重复以下步骤，直到栈为空： a. 出栈一个节点，并标记为已访问。 b. 检查该节点的所有未被访问的邻居节点。 c. 对于每个未访问的邻居节点，将其标记为已访问并入栈。

4.如果无法再继续，即没有未访问的邻居节点，返回上一个节点并继续。

5.重复步骤2-4，直到遍历整个图。
————————————————

原文链接：https://blog.csdn.net/qq_35831906/article/details/133829680

### 代码：![img](https://img-blog.csdnimg.cn/482d05b4130c4c8ca1134d4552b7c91e.png)

```python
# 找从A到G的路径
# 定义示例图
GRAPH = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': ['F'],
    'F': ['G'],
    'G': []
}
 
# 定义DFS算法，查找从起始节点到目标节点的路径
def dfs(graph, start, end, path=[]):
    # 将当前节点添加到路径中
    path = path + [start]
 
    # 如果当前节点等于目标节点，返回找到的路径
    if start == end:
        return path
 
    # 如果当前节点不在图中，返回None
    if start not in graph:
        return None
 
    # 遍历当前节点的邻居节点
    for node in graph[start]:
        # 如果邻居节点不在已访问的路径中，继续DFS
        if node not in path:
            new_path = dfs(graph, node, end, path)
            # 如果找到路径，返回该路径
            if new_path:
                return new_path
 
    # 如果无法找到路径，返回None
    return None
 
# 调用DFS算法查找从A到G的路径
path = dfs(GRAPH, 'A', 'G')
if path:
	print("Path from A to G:", path)
else:
    print("No path found.")

```



## 5.基于数组的序列：

列表类（list），元组类（tuple)，字符串类（str)

数组中每个位置称为**单元**，并用整数索引值描述该数组。

### 拷贝

通过复制创建一个新的列表时，

比如backup = list(primes) 或是backup = primes.copy()，这些是**浅拷贝**。

拷贝出来的列表和原列表共享元素，同时改变元素的值。

可以用**copy模块的deepcopy函数**来复制列表的元素，得到一个具有全新元素的新列表，这种方法被称为**深拷贝**

### 字符串

```python
letters = ''
for c in document:
	if c isalpha():
	letters += c
```

这种方法可能会很慢，时间复杂度能达到O（n^2^)

改进一下：

```python
temp = []
for c in document:
	if c.isalpha():
		temp.append(c)
letters = ''.join(temp)
```

创建一个临时空表会更好一些

这种方法能确保运行时间为O（n)

进一步完善，用==列表推导式语法==来创建临时表

```python
letters = ''.join([c for c in document if c.isalpha])
```

更进一步的，我们使用==生成器理解==可以完全避免使用临时表。

```python
letters = ''.join(c for c in document if c.isalpha())
```



## 6.dp

解决方式：二维数组，动态规划



### 一维数组

#### e.g.最长上升子序列

http://cs101.openjudge.cn/practice/02757/

```python
n = int(input())
lst = list(map(int,input().split()))
dp = [1 for _ in range(n)]
for i in range(n):
    for j in range(i): ##每个重新比一下，因为从两个数一直推广到所有，所以不会有误差
        if lst[j] < lst[i]: ##如果小于了，就可以加入进去
            dp[i] = max(dp[j] + 1,dp[i]) ##取最大值，如果把那个数加进去了或者没有加那个数
print(max(dp))
```

#### e.g.拦截导弹

http://cs101.openjudge.cn/practice/02945/

```python
n = int(input())
lst = list(map(int,input().split()))
dp = [1 for _ in range(n)]
for i in range(n):
    for j in range(i):
        if lst[j] >= lst[i]:
            dp[i] = max(dp[j] + 1,dp[i])
print(max(dp))
```



### 二维数组

#### e.g. 采药

http://cs101.openjudge.cn/practice/02773/

```python
```









































## 7.堆

堆是一种特殊的[树形数据结构](https://so.csdn.net/so/search?q=树形数据结构&spm=1001.2101.3001.7020)，其中每个节点的值都小于或等于（最小堆）或大于或等于（最大堆）其子节点的值。堆分为最小堆和最大堆两种类型，其中：

- 最小堆： 父节点的值小于或等于其子节点的值。
- 最大堆： 父节点的值大于或等于其子节点的值。
  堆常用于实现优先队列和堆排序等算法。

== 看到一直要用min，max的基本都要用堆

```python
import heapq
x = [1,2,3,5,7]

heapq.heapify(x)
###将列表转换为堆。

heapq.heappushpop(heap, item)
##将 item 放入堆中，然后弹出并返回 heap 的最小元素。该组合操作比先调用 heappush() 再调用 heappop() 运行起来更有效率

heapq.heapreplace(heap, item)
##弹出并返回最小的元素，并且添加一个新元素item

heapq.heappop(heap,item)
heapq.heappush(heap,item)
```



### 懒删除

懒删除就是，我表面上删除一个元素，实际上没有从堆里拿出来。

而是当我访问堆，要pop堆顶的时候，检查一下这个元素被没被删过

http://cs101.openjudge.cn/2024sp_routine/22067/

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

## 8.递归

必须有一个明确的结束条件。
每次进入更深一层递归时，问题规模（计算量）相比上次递归都应有所减少。
递归效率不高，递归层次过多会导致栈溢出（在计算机中，函数调用是通过栈（stack）这种数据结构实现的，每当进入一个函数调用，栈就会加一层栈帧，每当函数返回，栈就会减一层栈帧。由于栈的大小不是无限的，所以，递归调用的次数过多，会导致栈溢出

== 每个函数的操作，分为平行的操作和深度的操作。

————————————————

原文链接：https://blog.csdn.net/qq_44034384/article/details/107682376

```python
import sys
sys.setrecursionlimit(1000000) ##修改递归深度
```

1. A recursive algorithm must have a **base case**.
2. A recursive algorithm must change its state and move toward the base case.
3. A recursive algorithm must call itself, recursively.

### e.g.汉诺塔问题

*#将n-1个小圆盘从A经过C移动到B* 

*#将n-1个小圆盘从B经过A移动到C*

http://cs101.openjudge.cn/2024sp_routine/solution/44172210/

```python
# 将编号为numdisk的盘子从init杆移至desti杆 
def moveOne(numDisk : int, init : str, desti : str):
    print("{}:{}->{}".format(numDisk, init, desti))

#将numDisks个盘子从init杆借助temp杆移至desti杆
def move(numDisks : int, init : str, temp : str, desti : str):
    if numDisks == 1:
        moveOne(1, init, desti)
    else: 
        # 首先将上面的（numDisk-1）个盘子从init杆借助desti杆移至temp杆
        move(numDisks-1, init, desti, temp) 
        
        # 然后将编号为numDisks的盘子从init杆移至desti杆
        moveOne(numDisks, init, desti)
        
        # 最后将上面的（numDisks-1）个盘子从temp杆借助init杆移至desti杆 
        move(numDisks-1, temp, init, desti)

n, a, b, c = input().split()
move(int(n), a, b, c)
```

## 9.树





