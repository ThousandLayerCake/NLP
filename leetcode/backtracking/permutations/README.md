# 排列问题

排列问题是给定一个列表，让你返回所有可能的全排列。

因此，排列问题有以下特点：

1. 不需要 `startIndex`，因为列表内是有序的，所以就算元素相同，顺序不同的两个列表也是可以并存的。





## 目录

- [46.全排列](./46.md)
- [47.全排列 II](./47.md)

  



## 回溯法做题思路

全排列问题是很适合用回溯法来解决的，因为他和回溯法一样，本质上是要 **穷举所有可能**。



**再次回顾一下，回溯法的思路：**

1. 将题目抽象为树，把树先画出来。
2. 根据模板确定函数参数，横向循环，纵向递归。
3. 确定去重的关键，考虑 **横向去重** 和 **纵向去重**。
   - 横向去重：一般是列表与列表的重复
   - 纵向去重：一般是列表内的元素重复
4. 将结果加入到 **结果列表**。
5. 判断是否结束，如果只有唯一答案可以提前结束。





