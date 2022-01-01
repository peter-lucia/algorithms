# Algorithms Review Guide

#### Topics:

1. [Graph](#graph)
2. [Linked List](#linked-list)
3. [Dynamic Programming](#dynamic-programming)
4. [Sorting And Searching](#sorting-and-searching)
5. [Tree / Binary Search Tree](#tree--binary-search-tree)
6. [Number Theory](#number-theory)
7. [BIT Manipulation](#bit-manipulation)
8. [String / Array / Stack](#string--array)
9. [Python](#python)
10. [Preparation Guide](#preparation-guide)
11. [NP-Complete](#np-complete)

##### Graph
1. [Breadth First Search (BFS)](graph.ipynb)
2. [Depth First Search (DFS)](graph.ipynb)
3. [Shortest Path from source to all vertices **Dijkstra**](graph.ipynb)
4. Shortest Path from every vertex to every other vertex **Floyd Warshall**
5. To detect cycle in a Graph **Union Find**
6. [Minimum Spanning tree **Prim**](graph.ipynb)
7. [Minimum Spanning tree **Kruskal**](graph.ipynb)
8. Topological Sort
9. Boggle (Find all possible words in a board of characters)
10. Bridges in a Graph
11. There are 3 basic ways to represent a graph in memory (objects and pointers, matrix, and adjacency list). 
    [Each representation and its pros + cons. ](graph.ipynb)
12. [1293. Shortest Path in a Grid with Obstacles Elimination](graph.ipynb)

##### Trees / Binary Search Tree

1. [Basic tree construction, traversal and manipulation algorithms](tree.ipynb)
2. [Binary Trees](tree.ipynb)
3. [N-ary trees](tree.ipynb)
4. [Trie-trees](tree.ipynb)
5. [BFS Binary Tree](tree.ipynb)
6. Tree traversal algorithms: [BFS Adj list](graph.ipynb) and [DFS Adj list](graph.ipynb)
7. The difference between [inorder, postorder and preorder](tree.ipynb).
8. [Find Minimum Depth of a Binary Tree](tree.ipynb)
9. Maximum Path Sum in a Binary Tree
10. Check if a given array can represent Preorder Traversal of Binary Search Tree
11. Check whether a binary tree is a full binary tree or not
12. [Bottom View Binary Tree](tree.ipynb)
13. [Print Nodes in Top View of Binary Tree](tree.ipynb)
14. Remove nodes on root to leaf paths of length < K
15. Lowest Common Ancestor in a Binary Search Tree
16. Check if a binary tree is subtree of another binary tree
17. Reverse alternate levels of a perfect binary tree
18. [Find and Remove Leaves of Binary Tree](tree.ipynb)
19. [Find and Remove Leaves of Binary Tree (DFS)](graph.ipynb)
20. [Find mode in binary search tree (BFS)](tree.ipynb)
21. [ 116. Populating Next Right Pointers in Each Node in a Perfect Binary Tree](tree.ipynb)
22. [1026. Maximum Difference Between Node and Ancestor](tree.ipynb)

##### HashTables
1. [How to implement one using only arrays](hashmap.ipynb).
2. [432. Reconstruct original digits from english](hashmap.ipynb)
3. [811. Subdomain Visit count](hashmap.ipynb)


##### Linked List

1. [Insertion of a node in Linked List (On the basis of some constraints)](linked_list.ipynb)
2. [Delete a given node in Linked List (under given constraints)](linked_list.ipynb)
3. [Compare two strings represented as linked lists](linked_list.ipynb)
4. [Add Two Numbers Represented By Linked Lists](linked_list.ipynb)
5. Merge A Linked List Into Another Linked List At Alternate Positions
6. Reverse A List In Groups Of Given Size
7. Union And Intersection Of 2 Linked Lists
8. Detect And Remove Loop In A Linked List
9. Merge Sort For Linked Lists
10. Select A Random Node from A Singly Linked List
11. [Reverse a linked list](linked_list.ipynb) 
12. [2095. Delete the Middle Node of a Linked List](linked_list.ipynb)

##### Dynamic Programming

1. [Longest Common Subsequence](dynamic_programming.ipynb)
2. [Longest Increasing Subsequence](dynamic_programming.ipynb)
3. Edit Distance
4. Minimum Partition
5. Ways to Cover a Distance
6. Longest Path In Matrix
7. Subset Sum Problem
8. Optimal Strategy for a Game
9. [0-1 Knapsack Problem](dynamic_programming.ipynb)
10. Boolean Parenthesization Problem
11. [Get nth number in the Fibonacci Sequence](dynamic_programming.ipynb)
12. [Longest string chain](dynamic_programming.ipynb)
13. [792. Number of matching subsequences](dynamic_programming.ipynb)

##### Sorting And Searching

1. [Binary Search](sorting_and_searching.ipynb)
2. Search an element in a sorted and rotated array
3. Bubble Sort
4. Insertion Sort
5. [Merge Sort](sorting_and_searching.ipynb) can be [highly useful in situations where quicksort is impractical](https://www.geeksforgeeks.org/quick-sort-vs-merge-sort/)
6. [Heap Sort (Binary Heap)](sorting_and_searching.ipynb)
7. [Quick Sort](sorting_and_searching.ipynb)
8. Interpolation Search
9. Find Kth Smallest/Largest Element In Unsorted Array
10. Given a sorted array and a number x, find the pair in array whose sum is closest to x
11. [Counting Sort Pseudocode](https://en.wikipedia.org/wiki/Counting_sort) Worst Time: O(n+k), Worst Space: O(k), k = max(nums) 
12. [Stock Buy Sell to Maximize Profit](sorting_and_searching.ipynb)


##### Number Theory

1. Modular Exponentiation
2. Modular multiplicative inverse
3. [Primality Test | Set 2 (Fermat Method)](number_theory.ipynb)
4. Euler’s Totient Function
5. [Sieve of Eratosthenes](number_theory.ipynb)
6. Convex Hull
7. Basic and Extended Euclidean algorithms
8. Segmented Sieve
9. Chinese remainder theorem
10. Lucas Theorem
11. [Check if a number is prime or not](number_theory.ipynb)
12. [Number of primes less than n](number_theory.ipynb)
13. [1015. Smallest Integer Divisible by K](number_theory.ipynb)

##### Math: Combinatorics and Probability
15. [Probability problems](probability.ipynb), and other Discrete Math 101 situations. 
16. The essentials of [combinatorics](number_theory.ipynb) and probability. 
17. [N-choose-K problems](number_theory.ipynb)

##### BIT Manipulation

1. [Maximum Subarray XOR](bit_manipulation.ipynb)
2. Magic Number
3. Sum of bit differences among all pairs
4. Swap All Odds And Even Bits
5. Find the element that appears once
6. [Binary representation of a given number](bit_manipulation.ipynb)
7. Count total set bits in all numbers from 1 to n
8. Rotate bits of a number
9. Count number of bits to be flipped to convert A to B
10. Find Next Sparse Number
11. [476. Number Complement](bit_manipulation.ipynb)


##### String / Array

1. [Reverse an array without affecting special characters](string_array.ipynb)
2. [All Possible Palindromic Partitions](string_array.ipynb)
3. Count triplets with sum smaller than a given value
4. Convert array into Zig-Zag fashion
5. Generate all possible sorted arrays from alternate elements of two given sorted arrays
6. Pythagorean Triplet in an array
7. Length of the largest subarray with contiguous elements
8. Find the smallest positive integer value that cannot be represented as sum of any subset of a given array
9. Smallest subarray with sum greater than a given value
10. [735. Asteroid Collision](string_array.ipynb)
11. [20. Valid Parentheses](string_array.ipynb)
12. [189. Rotate Array](string_array.ipynb)
13. [68. Text Justification](string_array.ipynb)
14. [2007. Find Original Array from doubled array](string_array.ipynb)
15. [822. Find And Replace String](string_array.ipynb)
16. [384. Shuffle an Array](string_array.ipynb)

##### Python Language

1. [Defaultdict](python.ipynb)
2. [Bisect](python.ipynb)

#### NP-Complete

NP-Complete means a problem is both NP-Hard and a solution is verifiable in polynomial time.

Structure of NP-Complete proofs

1. Demonstrate that we can validate a solution for B in polynomial time **(B is in NP)**
2. Show the reduction from a known problem, $A \leq_p B$ (A is no harder than B and B is at least as hard as A). **(B is NP_Hard)**
   1. Instance of A converted to instance of B in polynomial time
   2. Solution of B converted to solution of A in polynomial time
   3. If you have a solution for B you have a solution for A
   4. If no solution for B no solution for A (or contra-positive – if you have a solution for A then you have a solution for B)

##### TODO
- [ ] Implement balanced binary trees: red/black tree, a splay tree, and AVL trees 
- [ ] Dijkstra's algorithm and the A* algorithm.
- [ ] BFS vs DFS tradeoffs
- [ ] The traveling salesman and the knapsack problem


##### References

* [Big-O Cheatsheet](https://www.bigocheatsheet.com)
* [Determine Primes via Square Root](http://mathandmultimedia.com/2012/06/02/determining-primes-through-square-root/)
* [Primes Stackoverflow](https://stackoverflow.com/questions/29595849/explain-a-code-to-check-primality-based-on-fermats-little-theorem)
* [Baillie-PSW Primality Test](https://en.wikipedia.org/wiki/Baillie–PSW_primality_test)
* [Wiki - Fermat Pseudoprime](https://en.wikipedia.org/wiki/Fermat_pseudoprime)
* [Primality Test](https://www.geeksforgeeks.org/primality-test-set-2-fermet-method/)
* [Youtube: Khan Academy Combination formula](https://www.youtube.com/watch?v=p8vIcmr_Pqo)
* [Dijkstra's Algorithm](https://www.analyticssteps.com/blogs/dijkstras-algorithm-shortest-path-algorithm)
* [Graph data structure](https://www.section.io/engineering-education/graph-data-structure-python/)
* [Adjacency List vs. Adjacency Matrix](https://www.geeksforgeeks.org/comparison-between-adjacency-list-and-adjacency-matrix-representation-of-graph/)
* [Common algorithms](https://www.geeksforgeeks.org/top-10-algorithms-in-interview-questions/)
* [Python heapq](https://docs.python.org/3/library/heapq.html)
