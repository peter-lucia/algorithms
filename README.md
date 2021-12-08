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
6. Minimum Spanning tree **Prim**
7. Minimum Spanning tree **Kruskal**
8. Topological Sort
9. Boggle (Find all possible words in a board of characters)
10. Bridges in a Graph



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

##### Sorting And Searching

1. [Binary Search](sorting_and_searching.ipynb)
2. Search an element in a sorted and rotated array
3. Bubble Sort
4. Insertion Sort
5. [Merge Sort](sorting_and_searching.ipynb)
6. [Heap Sort (Binary Heap)](sorting_and_searching.ipynb)
7. [Quick Sort](sorting_and_searching.ipynb)
8. Interpolation Search
9. Find Kth Smallest/Largest Element In Unsorted Array
10. Given a sorted array and a number x, find the pair in array whose sum is closest to x
11. [Counting Sort Pseudocode](https://en.wikipedia.org/wiki/Counting_sort) Worst Time: O(n+k), Worst Space: O(k), k = max(nums) 

##### Tree / Binary Search Tree

1. [Find Minimum Depth of a Binary Tree](tree.ipynb)
2. Maximum Path Sum in a Binary Tree
3. Check if a given array can represent Preorder Traversal of Binary Search Tree
4. Check whether a binary tree is a full binary tree or not
5. [Bottom View Binary Tree](tree.ipynb)
6. [Print Nodes in Top View of Binary Tree](tree.ipynb)
7. Remove nodes on root to leaf paths of length < K
8. Lowest Common Ancestor in a Binary Search Tree
9. Check if a binary tree is subtree of another binary tree
10. Reverse alternate levels of a perfect binary tree
11. [Find and Remove Leaves of Binary Tree](tree.ipynb)
12. [Find and Remove Leaves of Binary Tree (DFS)](graph.ipynb)

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
13. **TODO** Probability problems
14. **TODO** N choose k problems

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



##### String / Array

1. Reverse an array without affecting special characters
2. All Possible Palindromic Partitions
3. Count triplets with sum smaller than a given value
4. Convert array into Zig-Zag fashion
5. Generate all possible sorted arrays from alternate elements of two given sorted arrays
6. Pythagorean Triplet in an array
7. Length of the largest subarray with contiguous elements
8. Find the smallest positive integer value that cannot be represented as sum of any subset of a given array
9. Smallest subarray with sum greater than a given value
10. [Stock Buy Sell to Maximize Profit](sorting_and_searching.ipynb)
11. [735. Asteroid Collision](string_array.ipynb)
12. [20. Valid Parentheses](string_array.ipynb)
13. [189. Rotate Array](string_array.ipynb)

##### Python

1. [Defaultdict](python.ipynb)
2. [Bisect](python.ipynb)


##### Notes

1. Algorithm Complexity: Review the [Big-O Cheat Sheet](https://www.bigocheatsheet.com)
2. [Sorting](sorting_and_searching.ipynb): Know the details of two sorting algorithms such as [quicksort](sorting_and_searching.ipynb) and [merge sort](sorting_and_searching.ipynb)). 
    Merge sort can be
    [highly useful in situations where quicksort is impractical](https://www.geeksforgeeks.org/quick-sort-vs-merge-sort/)
3. [Hashtables](hashmap.ipynb): [How to implement one using only arrays](hashmap.ipynb). 
4. [Trees](tree.ipynb): Know about trees, basic tree construction, traversal and manipulation algorithms. 
    Familiarize yourself with [binary trees](tree.ipynb), [n-ary trees](tree.ipynb), and [trie-trees](tree.ipynb). 
    Be familiar with at least one type of balanced binary tree, whether it's a red/black tree, 
    a splay tree or an AVL tree, and know how it's implemented. 
    Tree traversal algorithms: [BFS Adj list](graph.ipynb), [BFS Binary Tree](tree.ipynb) 
    and [DFS Adj list](graph.ipynb), and the difference between [inorder, postorder and preorder](tree.ipynb).
5. [Graphs](graph.ipynb): There are 3 basic ways to represent a graph in memory (objects and pointers, matrix, and adjacency list). 
    [Each representation and its pros + cons. ](graph.ipynb)
    Know the basic graph traversal algorithms: breadth-first search and depth-first search. 
    Know their computational complexity, their tradeoffs, and how to implement them in real code. 
    Be sure to review Dijkstra's algorithm and the A* algorithm.
6. Other data structures: It is important to review as many other data structures and algorithms as possible. 
    Review the most famous classes of NP-complete problems, 
    such as traveling salesman and the knapsack problem, and be able to recognize them. [Know what NP-complete means.](#np-complete)
  
7. Mathematics: Review [probability problems](probability.ipynb), and other Discrete Math 101 situations. 
    Spend some time refreshing the essentials of [combinatorics](number_theory.ipynb) and probability. 
    Be familiar with [n-choose-k problems](number_theory.ipynb) and their ilk - the more the better.

#### NP-Complete

NP-Complete means a problem is both NP-Hard and a solution is verifiable in polynomial time.

Structure of NP-Complete proofs

1. Demonstrate that we can validate a solution for B in polynomial time **(B is in NP)**
2. Show the reduction from a known problem, $A \leq_p B$ (A is no harder than B and B is at least as hard as A). **(B is NP_Hard)**
   1. Instance of A converted to instance of B in polynomial time
   2. Solution of B converted to solution of A in polynomial time
   3. If you have a solution for B you have a solution for A
   4. If no solution for B no solution for A (or contra-positive – if you have a solution for A then you have a solution for B)

##### References

* [Big-O Cheatsheet](https://www.bigocheatsheet.com)
* [Primes](http://mathandmultimedia.com/2012/06/02/determining-primes-through-square-root/)
* [Primes](https://stackoverflow.com/questions/29595849/explain-a-code-to-check-primality-based-on-fermats-little-theorem)
* [Baillie-PSW Primality Test](https://en.wikipedia.org/wiki/Baillie–PSW_primality_test)
* [Fermat Pseudoprime](https://en.wikipedia.org/wiki/Fermat_pseudoprime)
* [Primality Test](https://www.geeksforgeeks.org/primality-test-set-2-fermet-method/)
* [Youtube: Khan Academy Combination formula](https://www.youtube.com/watch?v=p8vIcmr_Pqo)
* [Dijkstra's Algorithm](https://www.analyticssteps.com/blogs/dijkstras-algorithm-shortest-path-algorithm)
* [Graph data structure](https://www.section.io/engineering-education/graph-data-structure-python/)
* [Adjacency List vs. Adjacency Matrix](https://www.geeksforgeeks.org/comparison-between-adjacency-list-and-adjacency-matrix-representation-of-graph/)
* [Common algorithms](https://www.geeksforgeeks.org/top-10-algorithms-in-interview-questions/)
* [Python heapq](https://docs.python.org/3/library/heapq.html)
