# Algorithms Review Guide

#### Table of Contents:

1. [Graph](#graph)
2. [Tree / Binary Search Tree](#trees--binary-search-tree)
3. [HashTables](#hashtables)
4. [Sorting And Searching](#sorting-and-searching)
5. [Dynamic Programming](#dynamic-programming)
6. [Number Theory](#number-theory)
7. [Math: Combinatorics / Probability)](#math-combinatorics-and-probability)
8. [String / Array / Stack](#string--array)
9. [Linked List](#linked-list)
10. [BIT Manipulation](#bit-manipulation)
11. [Python](#python-language)
12. [NP-Complete](#np-complete)
13. [References](#references)

##### Graph
1. [Breadth First Search (BFS)](graph.html)
2. [Depth First Search (DFS)](graph.html)
3. [Shortest Path from source to all vertices **Dijkstra**](graph.html)
4. [Shortest Path from every vertex to every other vertex **Floyd Warshall**](graph.html)
5. [To detect cycle in a Graph **DFS**](graph.html)
6. [To detect cycle in a Graph **Union Find**](graph.html)
7. [Minimum Spanning tree **Prim**](graph.html)
8. [Minimum Spanning tree **Kruskal**](graph.html)
9. [Topological Sort](graph.html)
10. [Boggle (Find all possible words in a board of characters)](graph.html)
11. [Bridges in a Graph](graph.html)
12. There are 3 basic ways to represent a graph in memory (objects and pointers, matrix, and adjacency list). 
    [Each representation and its pros + cons. ](graph.html)
13. [1293. Shortest Path in a Grid with Obstacles Elimination](graph.html)

##### Trees / Binary Search Tree

1. [Basic tree construction, traversal and manipulation algorithms](tree.html)
2. [Binary Trees](tree.html)
3. [N-ary trees](tree.html)
4. [Trie-trees](tree.html)
5. [BFS Binary Tree](tree.html)
6. [AVL Tree](tree.html)
7. Tree traversal algorithms: [BFS Adj list](graph.html) and [DFS Adj list](graph.html)
8. The difference between [inorder, postorder and preorder](tree.html).
9. [Find Minimum Depth of a Binary Tree](tree.html)
10. [Maximum Path Sum in a Binary Tree](tree.html)
11. [Check if a given array can represent Preorder Traversal of Binary Search Tree](tree.html)
12. [Check whether a binary tree is a full binary tree or not](tree.html)
13. [Bottom View Binary Tree](tree.html)
14. [Print Nodes in Top View of Binary Tree](tree.html)
15. [Remove nodes on root to leaf paths of length < K](tree.html)
16. [Lowest Common Ancestor in a Binary Search Tree](tree.html)
17. [Check if a binary tree is subtree of another binary tree](tree.html)
18. [Reverse alternate levels of a perfect binary tree](tree.html)
19. [Find and Remove Leaves of Binary Tree](tree.html)
20. [Find and Remove Leaves of Binary Tree (DFS)](graph.html)
21. [Find mode in binary search tree (BFS)](tree.html)
22. [ 116. Populating Next Right Pointers in Each Node in a Perfect Binary Tree](tree.html)
23. [1026. Maximum Difference Between Node and Ancestor](tree.html)

##### HashTables
1. [How to implement one using only arrays](hashmap.html).
2. [432. Reconstruct original digits from english](hashmap.html)
3. [811. Subdomain Visit count](hashmap.html)
4. [49. Group Anagrams](hashmap.html)
5. [249. Group Shifted Strings](hashmap.html)
6. [347. Top k frequent elements](hashmap.html)

##### Sorting And Searching

1. [Binary Search](sorting_and_searching.html)
2. [Search an element in a sorted and rotated array](sorting_and_searching.html)
3. [Bubble Sort](sorting_and_searching.html)
4. [Insertion Sort](sorting_and_searching.html)
5. [Merge Sort](sorting_and_searching.html) can be [highly useful in situations where quicksort is impractical](https://www.geeksforgeeks.org/quick-sort-vs-merge-sort/)
6. [Heap Sort (Binary Heap)](sorting_and_searching.html)
7. [Quick Sort](sorting_and_searching.html)
8. Interpolation Search
9. Find Kth Smallest/Largest Element In Unsorted Array
10. Given a sorted array and a number x, find the pair in array whose sum is closest to x
11. [Counting Sort Pseudocode](https://en.wikipedia.org/wiki/Counting_sort) Worst Time: O(n+k), Worst Space: O(k), k = max(nums)
12. [Stock Buy Sell to Maximize Profit](sorting_and_searching.html)
14. [843. Guess the word](string_array.html)

##### Dynamic Programming

1. [Longest Common Subsequence](dynamic_programming.html)
2. [Longest Increasing Subsequence](dynamic_programming.html)
3. [Edit Distance](dynamic_programming.html)
4. [Minimum Partition](dynamic_programming.html)
5. [Ways to Cover a Distance](dynamic_programming.html)
6. Longest Path In Matrix
7. [Subset Sum Problem](dynamic_programming.html)
8. Optimal Strategy for a Game
9. [0-1 Knapsack Problem](dynamic_programming.html)
10. Boolean Parenthesization Problem
11. [Get nth number in the Fibonacci Sequence](dynamic_programming.html)
12. [Longest string chain](dynamic_programming.html)
13. [792. Number of matching subsequences](dynamic_programming.html)
14. [70. Climb Stairs](dynamic_programming.html)
15. [416. Partition Equal Subset Sum](dynamic_programming.html)


##### Number Theory

1. [Modular Exponentiation](number_theory.html)
2. Modular multiplicative inverse
3. [Primality Test | Set 2 (Fermat Method)](number_theory.html)
4. Euler’s Totient Function
5. [Sieve of Eratosthenes](number_theory.html)
6. Convex Hull
7. Basic and Extended Euclidean algorithms
8. Segmented Sieve
9. Chinese remainder theorem
10. Lucas Theorem
11. [Check if a number is prime or not](number_theory.html)
12. [Number of primes less than n](number_theory.html)
13. [1015. Smallest Integer Divisible by K](number_theory.html)

##### Math: Combinatorics and Probability
15. [Probability problems](probability.html), and other Discrete Math 101 situations. 
16. The essentials of [combinatorics](number_theory.html) and probability. 
17. [N-choose-K problems](number_theory.html)


##### String / Array

1. [Reverse an array without affecting special characters](string_array.html)
2. [All Possible Palindromic Partitions](string_array.html)
3. Count triplets with sum smaller than a given value
4. Convert array into Zig-Zag fashion
5. Generate all possible sorted arrays from alternate elements of two given sorted arrays
6. Pythagorean Triplet in an array
7. Length of the largest subarray with contiguous elements
8. Find the smallest positive integer value that cannot be represented as sum of any subset of a given array
9. Smallest subarray with sum greater than a given value
10. [735. Asteroid Collision](string_array.html)
11. [20. Valid Parentheses](string_array.html)
12. [189. Rotate Array](string_array.html)
13. [68. Text Justification](string_array.html)
14. [2007. Find Original Array from doubled array](string_array.html)
15. [822. Find And Replace String](string_array.html)
16. [384. Shuffle an Array](string_array.html)
17. [53. Maximum subarray](string_array.html)
18. [28. Implmeent strStr() / Rabin-Karp find needle in haystack](string_array.html)
19. [2128. Remove All Ones With Row and Column Flips](string_array.html)

##### Linked List

1. [Insertion of a node in Linked List (On the basis of some constraints)](linked_list.html)
2. [Delete a given node in Linked List (under given constraints)](linked_list.html)
3. [Compare two strings represented as linked lists](linked_list.html)
4. [Add Two Numbers Represented By Linked Lists](linked_list.html)
5. [Merge A Linked List Into Another Linked List At Alternate Positions](linked_list.html)
6. Reverse A List In Groups Of Given Size
7. Union And Intersection Of 2 Linked Lists
8. Detect And Remove Loop In A Linked List
9. Merge Sort For Linked Lists
10. [Select A Random Node from A Singly Linked List](linked_list.html)
11. [Reverse a linked list](linked_list.html)
12. [2095. Delete the Middle Node of a Linked List](linked_list.html)
13. [21. Merge Two Sorted Lists](linked_list.html)

##### BIT Manipulation

1. [Maximum Subarray XOR](bit_manipulation.html)
2. Magic Number
3. Sum of bit differences among all pairs
4. Swap All Odds And Even Bits
5. Find the element that appears once
6. [Binary representation of a given number](bit_manipulation.html)
7. Count total set bits in all numbers from 1 to n
8. Rotate bits of a number
9. Count number of bits to be flipped to convert A to B
10. Find Next Sparse Number
11. [476. Number Complement](bit_manipulation.html)

##### Python Language

1. [Defaultdict](python.html)
2. [Bisect](python.html)
3. [Class variables vs. instance variables](python.html)
4. [Static variables in functions](python.html)

#### Geometry
1. [Is Square Valid](geometry.html)

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
* [Heapsort](https://www.geeksforgeeks.org/heap-sort/)
