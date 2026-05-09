from __future__ import annotations

from typing import Optional


class TreeNode:
    """二叉树节点。"""

    def __init__(
        self,
        val: int = 0,
        left: Optional[TreeNode] = None,
        right: Optional[TreeNode] = None,
    ) -> None:
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        """返回二叉树中的最大路径和。

        路径可以从任意节点开始到任意节点结束，每次只能经过一条边。
        """

        ans: int = -(10**9)  # 全局最大值

        def dfs(node: Optional[TreeNode]) -> int:
            """返回以 node 为起点的单边最大路径和。"""
            nonlocal ans
            if node is None:
                return 0
            left_gain = max(dfs(node.left), 0)
            right_gain = max(dfs(node.right), 0)
            # 经过当前节点的最大路径和
            cur = node.val + left_gain + right_gain
            ans = max(ans, cur)
            # 返回单边最大值给父节点使用
            return node.val + max(left_gain, right_gain)

        dfs(root)
        return ans
