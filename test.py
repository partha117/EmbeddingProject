import javalang
import pandas as pd
import os
from collections import deque

code_segment = open("Data/UUID_Files/ddf096e9-d8f8-40ef-9ded-4d2b44fff4ce.java","r").read()
code_segment = """class Knapsack { 
  
    // A utility function that returns maximum of two integers 
    static int max(int a, int b) { return (a > b) ? a : b; } 
  
    // Returns the maximum value that can  
    // be put in a knapsack of capacity W 
    static int knapSack(int W, int wt[], int val[], int n) 
    { 
        // Base Case 
        if (n == 0 || W == 0) 
            return 0; 
  
        // If weight of the nth item is more  
        // than Knapsack capacity W, then 
        // this item cannot be included in the optimal solution 
        if (wt[n - 1] > W) 
            return knapSack(W, wt, val, n - 1); 
  
        // Return the maximum of two cases: 
        // (1) nth item included 
        // (2) not included 
        else
            return max(val[n - 1] + knapSack(W - wt[n - 1], wt, val, n - 1), 
                       knapSack(W, wt, val, n - 1)); 
    } 
  
    // Driver program to test above function 
    public static void main(String args[]) 
    { 
        int val[] = new int[] { 60, 100, 120 }; 
        int wt[] = new int[] { 10, 20, 30 }; 
        int W = 50; 
        int n = val.length; 
        System.out.println(knapSack(W, wt, val, n)); 
    } 
} """
code_segment = """class Knapsack { 

    // A utility function that returns maximum of two integers 
    static int max(int a, int b) { return (a > b) ? a : b; }}"""
tree = javalang.parse.parse(code_segment)
object_map = dict()
for path, node in tree:
    temp = ""
    for item in node.children:
        if isinstance(item, javalang.ast.Node):
            temp += str(type(item)) + " "
            object_map[id(item)] = str(type(node))
    print("Parent: " + object_map.get(id(node), "None"))
    print("Node: " + str(type(node)))
    print("Children: " + temp)
# q = deque()
# q.append(tree)
# while(len(q) > 0):
#     item = q.popleft()
#     print(item.name if hasattr(item, "name") else item)
#     if hasattr(item,"children"):
#         for children in item.children:
#             if isinstance(children,list):
#                 for element in children:
#                     if element is not None:
#                         q.append(element)
#             # else:
#             #     if children is not None:
#             #         q.append(children)
# print(item)
# for path, node in tree:
#     if len(path) > 0:
#         if isinstance(path[-1], list):
#             print(path[-1][0] == node)
#     print("Path:", path,"Node:", node.name if hasattr(node, "name") else node)
# print('wcwqc')
# df = pd.read_csv("Data/Unified_Data.csv")
# df = df[df['language_name'] == 'Java']
# df.reset_index(drop=True, inplace=True)
# df.to_csv("Data/Java_Unified_Data.csv", index=False)