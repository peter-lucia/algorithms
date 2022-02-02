#!/usr/bin/env python
# coding: utf-8

# # Probability

# In[1]:


import random
from typing import List

class Solution:
    def findSecretWord(self, wordlist: List[str], master: 'Master') -> None:
        """
        https://leetcode.com/problems/guess-the-word/discuss/940407/O(n)-with-detailed-explanation-Python-JavaScript-GoLang

        1. Randomly pick a word from wordlist, guess it and get the number of digits, x, that are in the correct position
        2. Find all candidate words that have at least x digits in common with the last guessed word
        3. Shorten the wordlist to only include the candidates found in step 2. Repeat step 1 up to 10 times.
        """
        random.seed(100)

        if len(wordlist) <= 10:
            for word in wordlist:
                master.guess(word)
            return


        def get_num_matches(w1: str, w2: str) -> int:
            """
            Gets the number of digits that are matching in the same position
            Assumes words are of the same size
            """
            result = 0
            for i in range(len(w1)):
                if w1[i] == w2[i]:
                    result += 1
            return result


        for i in range(10):
            random_idx = random.randrange(len(wordlist))
            guess_word = wordlist[random_idx]
            score = master.guess(guess_word)
            if score == 6:
                return guess_word
            candidates = []
            for word in wordlist:
                if score == get_num_matches(word, guess_word):
                    candidates.append(word)
            wordlist = candidates

