from tree_sitter import Language, Parser
from collections import deque
from queue import LifoQueue
from transformers import Trainer
from transformers import Trainer, TrainingArguments

Trainer().train()
class TreeNode:
    def __init__(self,node_info, parent=None):
        self.node_information = node_info
        self.children = []
        self.visited = False
        self.parent = parent
    def add_child(self,child):
        self.children.append(child)
code_segment = """class Knapsack { 

    // A utility function that returns maximum of two integers 
    static int max(int a, int b) { a = 2 ; b = a * 2 ; return (a > b) ? a : b; }}"""
JAVA_LANGUAGE = Language('build/my-languages.so', 'java')
parser = Parser()
parser.set_language(JAVA_LANGUAGE)


def get_traversable_tree(root_node):
    item_queue = deque()
    item_queue.append(tree.root_node)
    parent_tree_node = TreeNode(root_node)
    object_mapper = {str(root_node): parent_tree_node}
    while len(item_queue) > 0:
        current_node = item_queue.popleft()
        parent_tree_node = object_mapper.get(str(current_node))
        for child in current_node.children:
            child_tree_node = TreeNode(node_info=child, parent=parent_tree_node)
            object_mapper[str(child)] = child_tree_node
            parent_tree_node.add_child(child_tree_node)
            item_queue.append(child)
    return object_mapper[str(root_node)]

def get_bracket_representation(root_node):
    stack = LifoQueue()
    stack.put(root_node)
    text = ""
    while stack.qsize() > 0:
        current_node = stack.get()
        if isinstance(current_node, str):
            text += current_node
        else:
            current_node.visited = True
            text += current_node.node_information.type + "("
            stack.put(")")
            for child in current_node.children:
                if not child.visited:
                    stack.put(child)
    return text


tree = parser.parse(bytes(code_segment,"utf-8"))
cursor = tree.walk()
traversable_tree_root = get_traversable_tree(tree.root_node)
print(get_bracket_representation(traversable_tree_root))
# print(cursor)
# q = deque()
# q.append(tree.root_node)
# parent = None
# parent_tree_node = TreeNode(tree.root_node)
# object_mapper = {str(tree.root_node): parent_tree_node}
# while len(q) > 0:
#     current_node = q.popleft()
#     print(current_node)
#     parent_tree_node = object_mapper.get(str(current_node))
#     for child in current_node.children:
#         child_tree_node = TreeNode(node_info=child,parent=parent_tree_node)
#         object_mapper[str(child)] = child_tree_node
#         parent_tree_node.add_child(child_tree_node)
#         q.append(child)
# print(tree.root_node)