# 棋盘问题

这里记录利用回溯法来解决的一类棋盘问题。

棋盘问题可以视为将棋盘所有的点作为一个列表，**目标是在这个列表中找到符合要求的一个子集**。

只要有列表，取子集，那便与组合问题大差不差了。只是限制条件上会多了一些。





## 目录

- [51. N皇后](./51.md)
- [37. 解数独](./37.md)









## 回溯法做题思路

**再次回顾一下，回溯法的思路：**

1. 将题目抽象为树，把树先画出来。
2. 根据模板确定函数参数，横向循环，纵向递归。
3. 确定去重的关键，考虑 **横向去重** 和 **纵向去重**。
   - 横向去重：一般是列表与列表的重复
   - 纵向去重：一般是列表内的元素重复
4. 将结果加入到 **结果列表**。
5. 判断是否结束，如果只有唯一答案可以提前结束。

