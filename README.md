# `dedup`

Pretty much [this Spark pipeline](https://gist.github.com/ncoop57/c2149e8413a0f0c531051154348a9ed3) but implemented in C++.  Deduplicate documents using MinHash.  The comparison of jaccard scores is could be faster but I wanted to make this simple.
