#!/usr/bin/env python
# coding: utf-8

# # HashMap

# HashTable implementation using only arrays
# 
# https://leetcode.com/problems/design-hashmap/
# 
# Sources:
# * https://algs4.cs.princeton.edu/34hash/
# * https://pagekeysolutions.com/blog/dsa/hash-table-python/

# ### Simple HashMap (no hash function)

# In[1]:


from typing import List, Optional, Any


class MyHashMap:

    def __init__(self):
        self.size = 10**6 + 1
        self.keys = [-1 for _ in range(self.size)]

    def put(self, key: int, value: int) -> None:
        self.keys[key] = value

    def get(self, key: int) -> int:
        return self.keys[key]

    def remove(self, key: int) -> None:
        self.keys[key] = -1


# ### HashMap (hash function)
# * collision resolution
#   * chaining via a list of buckets to handle collision
#   * the original unhashed key is stored in each bucket
#   so lookups can resolve to the correct key if collision had occurred
# 
# As one of the most intuitive implementations, we could adopt the modulo operator as the hash function, since the key value is of integer type. In addition, in order to minimize the potential collisions, it is advisable to use a prime number as the base of modulo, e.g. 2069.
# 
# Modulo as non-prime:
# 
# 1000 % 10 == 1000 % 100, which is 0
# 
# 2069 is a large prime number
# 
# Here, a bucket is a list of tuples.
# 
# #### Collisions
# To avoid collisions, like with keys 2070 and 1,
# we only update an existing tuple
# in the bucket if the original key is a match even though
# the hashed keys may be the same.
# 
# 
# ```
# key: 2070, value: 2
# key: 1, value: 4
# 
# hash(2070) = 1
# hash(1) = 1
# 
# hash_map[1] = Bucket([(2070, 2), (1, 4)])
# 
# ```
# 
# 

# In[2]:


class Bucket:
    def __init__(self):
        self.data = []

    def update(self, key, value):

        found = False
        for i, kv in enumerate(self.data):
            if kv[0] == key:
                found = True
                self.data[i] = (key, value)
        if not found:
            self.data.append((key, value))

    def get(self, key):
        for k, v in self.data:
            if key == k:
                return v
        return -1

    def remove(self, key):
        for i, kv in enumerate(self.data):
            if kv[0] == key:
                self.data.pop(i)

class MyHashMap:

    def __init__(self):
        # the size of the table should be a prime number
        # to reduce the number of collisions
        self.size = 2069
        self.hash_map = [Bucket() for i in range(self.size)]

    def put(self, key: int, value: int) -> None:
        self.hash_map[self.hash(key)].update(key,value)

    def get(self, key: int) -> int:
        return self.hash_map[self.hash(key)].get(key)

    def remove(self, key: int) -> None:
        self.hash_map[self.hash(key)].remove(key)

    def hash(self, key):
        return key % self.size


# #### Test Collision

# In[3]:


hash_map = MyHashMap()
hash_map.put(2070, 2)
hash_map.put(1, 4)  # collision


# In[4]:


hash_map.get(2070)


# In[5]:


hash_map.get(1)


# #### Hashmap (with hash function and a single class)

# In[6]:


class MyHashMap:

    def __init__(self):
        self.size = 2069  # large prime
        self.keys = [[] for _ in range(self.size)]

    def put(self, key: int, value: int) -> None:
        key_hash = self.hash(key)
        found = False
        for i, kv in enumerate(self.keys[key_hash]):
            if kv[0] == key:
                found = True
                self.keys[key_hash][i] = (key, value)
        if not found:
            self.keys[key_hash].append((key, value))

    def get(self, key: int) -> int:
        key_hash = self.hash(key)
        for i, kv in enumerate(self.keys[key_hash]):
            if kv[0] == key:
                return kv[1]
        return -1

    def remove(self, key: int) -> None:
        key_hash = self.hash(key)
        for i, kv in enumerate(self.keys[key_hash]):
            if kv[0] == key:
                self.keys[key_hash].pop(i)

    def hash(self, key):
        return key % self.size


# #### Test Collision

# In[7]:


hash_map = MyHashMap()
hash_map.put(2070, 2)
hash_map.put(1, 4)  # collision


# In[8]:


hash_map.get(2070)


# In[9]:


hash_map.get(1)


# #### Hash functions (string -> int)
# 
# A good hash function should
# 1. Use all the data in the key
# 2. Uniformly distribute data in the table
# 3. Be deterministic. Gives the same output for the same input.

# In[10]:


def hash(key: str, hash_table_size: int) -> int:
    """
    Computes the hash of a string
    :param key: A string to hash
    :param hash_table_size: preferably a large prime number to avoid collisions
    :return: an index between 0 and hash_table_size
    """
    s = 0
    for c in key:
        # ord converts a string to an int
        s += ord(c)
    return s % hash_table_size

hash("abc", 2069)


# #### Implement a hashmap using only arrays

# In[11]:


class Bucket:

    def __init__(self):
        self.key = None
        self.values = []  #  [(unhashed_key, value), (unhashed_key_2, value), ...]

    def get(self, orig_key):
        for kv in self.values:
            if kv[0] == orig_key:
                return kv[1]
        return None

    def put(self, orig_key, value):
        found = False
        for idx, kv in enumerate(self.values):
            if orig_key == kv[0]:
                self.values[idx] = (orig_key, value)
                found = True
                break
        if not found:
            self.values.append((orig_key, value))

    def remove(self, orig_key):
        for idx, kv in enumerate(self.values):
            if orig_key == kv[0]:
                self.values.pop(idx)
                break

class HashTable:

    def __init__(self):
        # we use a prime number to prevent collisions
        # (i.e. n % prime_number incur fewer collisions than n % even_number for example)
        self.size = 2069
        self.table = [Bucket() for _ in range(self.size)]

    def get(self, key: str):
        table_idx = self.hash(key)
        return self.table[table_idx].get(key)

    def put(self, key: str, value):
        table_idx = self.hash(key)
        self.table[table_idx].put(key, value)

    def remove(self, key: str):
        table_idx = self.hash(key)
        self.table[table_idx].remove(key)

    def hash(self, key: str):
        """
        str -> int -> int % max_hash_table_size
        """

        s = 0
        for c in key:
            s += ord(c)
        return s % self.size


ht = HashTable()
ht.put("My Name", "Peter")
ht.put("My Name", "Peter Lucia")
ht.get("My Name")


# In[12]:


ht.get("My Name")


# In[13]:


ht.remove("My Name")
ht.get("My Name")


# ### Reconstruct original digits from english
# 
# [https://leetcode.com/problems/reconstruct-original-digits-from-english/](https://leetcode.com/problems/reconstruct-original-digits-from-english/)

# In[14]:


from collections import Counter
class Solution:
    def originalDigits(self, s: str) -> str:
        # approach
        # O(n) time complexity
        # O(1) space complexity

        # create a list of numbers 0-9 in english

        # zero - number of z's since it's the only one that has a z
        # one - number of o's minus counts for others with an o: zero, two, four
        # two - number of w's
        # three - number of t's minus counts for others with a 't': two and eight
        # four - number of u's
        # five - number of f's minus count for others with f: four
        # six - number of x's
        # seven - number of s's minus count for others with s: six
        # eight - number of g's
        # nine - number of i's minus count for others with i: eight: six, five

        # build {'a': 1, 'b': 2, 'c': 3}
        lookup = Counter(s)

        result = ""
        result += "0"*(lookup['z'])
        result += "1"*(lookup['o'] - lookup['z'] - lookup['w'] - lookup['u'])
        result += "2"*(lookup['w'])
        result += "3"*(lookup['t'] - lookup['w'] - lookup['g'])
        result += "4"*(lookup['u'])
        result += "5"*(lookup['f'] - lookup['u'])
        result += "6"*(lookup['x'])
        result += "7"*(lookup['s'] - lookup['x'])
        result += "8"*(lookup['g'])
        result += "9"*(lookup['i'] - lookup['g'] - lookup['x'] - (lookup['f'] - lookup['u']))

        return result

Solution().originalDigits("onetwothreefourfivesixseveneightnine")


# ### Subdomain Visit Count
# 
# [https://leetcode.com/problems/subdomain-visit-count/](https://leetcode.com/problems/subdomain-visit-count/)

# In[15]:


from collections import defaultdict
from typing import List

class Solution:
    def subdomainVisits(self, cpdomains: List[str]) -> List[str]:
        # 1. build hashmap

        # google.mail.com -> 900
        # mail.com -> 900 + 1
        # com -> 900 + 50 + 1
        # yahoo.com -> 50
        # intel.mail.com -> 1
        # wiki.org -> 5
        # org -> 5

        lookup = defaultdict(int)
        for cpdomain in cpdomains:
            count, url = cpdomain.split(" ")
            count = int(count)
            domains = url.split(".")
            for i in range(len(domains)):
                # Note:
                # >>> ".".join(['a'])
                # 'a'
                # google.mail.com -> 900
                # mail.com -> 900
                # com -> 900
                key = ".".join(domains[i:])
                lookup[key] += count

        result = [f"{v} {k}" for k,v in lookup.items()]
        return result


# In[16]:


Solution().subdomainVisits(["900 google.mail.com", "50 yahoo.com", "1 intel.mail.com", "5 wiki.org"])


# In[17]:


from typing import List
from collections import defaultdict
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        # LIS modeled solution
        # O(n^2) solution:
        # [1,1,1], k = 2
        # [1,1]
        #   [1,1]
        # result: 2

        # [1,2,3], k = 3
        # [1,2]
        #     [3]
        # result: 2

        # for i = 0 -> n - 1
        #    for j = 0 -> i-1
        #        if sum from j to i == k, increment total

        # Hashmap
        # O(n) solution
        # {
        #  sum: occurrences of sum
        #
        # }
        #
        #
        #
        lookup = defaultdict(int)
        lookup[0] = 1  # always a sum of 0
        n = len(nums)
        running_sum = 0
        result = 0
        for num in nums:
            running_sum += num
            if running_sum - k in lookup:
                # defaultdict(<class 'int'>, {0: 1, 1: 1, 2: 1, 3: 1})
                # k = 2
                #    [1,1,1]  lookup[3-k] = lookup[3-2] = lookup[1] = 1
                result += lookup[running_sum - k]
            lookup[running_sum] += 1

        return result
assert Solution().subarraySum([1,1,1], 2) == 2


# In[18]:


import bisect
class Solution:
    def intToRoman(self, num: int) -> str:

        # key points

        # Create hash table of symbols mapping value to the symbol
        # 1: 'I'
        # 5: 'V'
        # ...

        # Convert num to str, handle each digit individually
        # If 4 or 9 * 10^x is found, handle it separately
        # otherwise, use a separate function to determine sum of each digit

        # Procedure: Iterate over each digit from right to left
        # i = 0
        # for each digit (right to left)
        #   if digit is 4 or 9: special handling
        #   otherwise, use function
        #   i += 1

        lookup = {
            1: "I",
            5: "V",
            10: "X",
            50: "L",
            100: "C",
            500: "D",
            1000: "M",
        }

        digits = [c for c in str(num)][::-1]
        i = 0
        result = ""
        while i < len(digits):
            tens_mult = 10**i
            digit = int(digits[i])*tens_mult
            if digits[i] in ['4', '9']:
                # ...4... = "I" + "V" + existing result
                result = (self.get_symbols(tens_mult, lookup)
                        + self.get_symbols(int(digit) + tens_mult, lookup)
                        + result)
            else:
                result = self.get_symbols(int(digit), lookup) + result
            i += 1

        return result

    def get_symbols(self, num: int, lookup: dict) -> str:
        """
        Recursively find the largest symbol where remainder is >= 0 until remainder is 0

        Assumes 4 and 9 are not present

        Example:
            subtract value of symbol, add symbol to result, keep going until
            remainder is 0

            Start with 27
            largest symbol where remainder is >= 0 is X
            27 - lookup[X] = 27-10 = 17

            17
            largest symbol where remainder is >= 0 is X
            17 - 10 = 7

            7
            largest symbol where remainder is >= 0 is V
            7 - 5 = 2

            2
            largest symbol where remainder is >= 0 is I
            2 - 1 = 1

            1
            largest symbol where remainder is >= 0 is I
            1 - 1 = 0 -> we are done
            XXVII = 27
        """
        ks = list(lookup.keys())
        result = ""
        while num > 0:
            # get last key that's just less than the num
            i = bisect.bisect(ks, num) - 1
            result += lookup[ks[i]]
            num -= ks[i]
        return result
assert Solution().intToRoman(49) == "XLIX"
assert Solution().intToRoman(490) == "CDXC"
assert Solution().intToRoman(4) == "IV"


# ### 49. Group Anagrams
# 
# Source: [https://leetcode.com/problems/group-anagrams/](https://leetcode.com/problems/group-anagrams/)

# In[19]:


from typing import List
from collections import defaultdict
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:

        # Approach #1: create lookup for each word
        # Map sorted version of the string to list of matching strings
        # Return a list of list of values at the end grouped by key
        # Time complexity: O(n)
        # Space complexity: O(n)

        result = defaultdict(list)

        for s in strs:
            key = "".join(sorted(s))
            result[key].append(s)

        return list(result.values())

Solution().groupAnagrams(["eat","tea","tan","ate","nat","bat"])


# In[20]:


# Slower solution
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:

        # Approach #1: create lookup for each word
        # 1. For each word, map each letter in the word to the number of occurrences of the letter in that word (Counter(word))
        # 2. Add the word's dictionary to a list of unique letter dictionaries for all words
        # 3. The result List[List:str]] is the same length as the list of unique dictionaries of words
        # 4. Keep a mapping of word->index in list of unique dictionaries in a separate dict
        # 5. Go through the keys in the word->index dict and add them to the result
        # Example:
        # ["eat","tea","tan","ate","nat","bat"]
        # unique word dictionaries [{e:1, a:1, t:1}, {t:1, a:1, n:1}, {t:1, a:1, b:1} (length = 3)
        # {eat: 0, tea: 0, tan:1, ate: 0, nat: 1, bat: 2}
        # result: [[eat, tea, ate], [tan, nat], [bat]] (length = 3)
        # time complexity: O(n^2) worst case for hash table insertion, O(n) average

        # Approach #2:

        result = defaultdict(list)

        for s in strs:
            d = Counter(s)
            key = "".join([f"{k}{v}" for k, v in sorted(d.items())])
            result[key].append(s)

        groups = []
        for k, v in result.items():
            groups.append(v)
        return groups


# ### 249. Group Shifted Strings
# Source: [https://leetcode.com/problems/group-shifted-strings/](https://leetcode.com/problems/group-shifted-strings/)

# In[21]:


class Solution:
    def groupStrings(self, strings: List[str]) -> List[List[str]]:
        # Approach
        # Shift each string until the starting character is a
        # If the shifted string is in the lookup of previously shifted strings
        # then append it to the list of strings for that key
        # Repeat for all strings

        lookup = defaultdict(list)

        def shift(key: str) -> str:
            # shifts a string until starting character is 'a'

            if key[0] != 'a':
                diff = ord(key[0]) - ord('a')
                result = []
                for c in key:
                    c_shifted = ord(c) - diff
                    if c_shifted < ord('a'):
                        c_shifted = ord('z') - (ord('a') - c_shifted) + 1
                    result.append(chr(c_shifted))
                result = "".join(result)
            else:
                result = key
            return result

        for s in strings:
            s_shifted = shift(s)
            lookup[s_shifted].append(s)

        return lookup.values()
Solution().groupStrings(["abc","bcd","acef","xyz","az","ba","a","z"])


# ### Get the top k frequent elements

# In[22]:


class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:

        # Approach: add each number to a hashmap num (int) -> occurrences (int)
        # sorted the hashmap by values
        # return the first k keys in the sorted hashmap
        # Time complexity: O(n)
        # Space complexity: O(n)

        lookup = defaultdict(int)
        for num in nums:
            lookup[num] += 1

        sorted_lookup = sorted(list(lookup.items()), key=lambda val: val[1], reverse=True)
        result = []
        for i, (key,value) in enumerate(sorted_lookup):
            result.append(key)
            if i+1 == k:
                break
        return result

assert Solution().topKFrequent([1,1,1,2,2,3], 2) == [1,2]


# ### Experiments
# 
# Build a hashmap using arrays
# 

# In[23]:


import bisect
class BBucket:
    def __init__(self):
        self.values = []  # list of tuples

    def get(self, key: str):
        # linear search since keys are str
        for original_key, value in self.values:
            if key == original_key:
                return value

    def update(self, key: str, value: int):
        # just compare the key, float("-inf") will always compare less than a value
        # i = bisect.bisect(self.values, (key, float("-inf")))
        i = bisect.bisect(self.values, (key,))  # ignore second element of tuple
        if i == 0 or i == len(self.values):
            self.values.insert(i, (key, value))
        else:
            self.values[i] = (key, value)

    def remove(self, key: str):
        # linear search since keys are str
        for i in range(len(self.values)):
            if self.values[i] == key:
                self.values.pop(i)


class HashTable:

    def __init__(self):
        self.size = 2069  # prime number to avoid collisions
        self.buckets = [BBucket() for _ in range(self.size)]

    def get(self, key: str):
        return self.buckets[self.hash(key)].get(key)

    def put(self, key: str, value):
        self.buckets[self.hash(key)].update(key, value)

    def hash(self, key: str):
        result = 0
        for c in key:
            result += ord(c) % self.size
        return result

lookup = HashTable()
lookup.put("first", 2)
lookup.put("first", 3)
print(lookup.get("first"))

lookup.put("second", 3)
lookup.put("second", 4)
lookup.put("second", 5)
print(lookup.get("second"))


# In[ ]:




