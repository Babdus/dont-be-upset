# dont-be-upset

### links

https://link.springer.com/chapter/10.1007%2F978-3-319-64698-5_18

https://ieeexplore.ieee.org/document/7442162

https://www.worldscientific.com/doi/abs/10.1142/S0218001418590012

http://openaccess.thecvf.com/content_ICCV_2017/papers/Gur_From_Square_Pieces_ICCV_2017_paper.pdf

https://research.wmz.ninja/articles/2018/03/teaching-a-neural-network-to-solve-jigsaw-puzzles.html

http://cs231n.stanford.edu/reports/2017/pdfs/110.pdf

[Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles](https://arxiv.org/pdf/1603.09246.pdf)

[DeepPermNet: Visual Permutation Learning](https://arxiv.org/pdf/1704.02729.pdf)
(ამის მიხედვით ავაწყოთ ქსელი)

[DEEP PERM-SET NET: LEARN TO PREDICT SETS WITH UNKNOWN PERMUTATION AND CARDINALITY USING DEEP NEURAL NETWORKS](https://arxiv.org/pdf/1805.00613.pdf)

[Solving Jigsaw Puzzles with Genetic Algorithms (youtube)](https://www.youtube.com/watch?v=6DohBytdf6I)

### surat.py

იმპორტისთვის:
```python
from surat import Surat
```

ახალი სურათის შექმნა:
```python
s = Surat(path, m, p)
```
სადაც m არის ერთ სვეტში ან სტრიქონში ფრაგმენტების რაოდენობა, და p არის თითოეულ ფრაგმენტში პიქსელების რაოდენობა

ერთი ფრაგმენტის ამოღება:
```python
f = s.get_fragment(row_n, col_n)
```
ან:
```python
f = s.get_fragment(None, None, number_of_fragment)
```

არეული სურათის დასაწყობად:
```python
s2 = s.rearrange(sequence)
```

უკან დასაბრუნებლად:
```python
s1 = s2.rearrange(sequence, reverse=True)
```

სურათის შესანახად:
```python
s.save(path)
```

ნებისმიერი ორი ფრაგმენტის წყვილის ამოსაღებად (ერთ სურათად მიწებებულის):
```python
pair = s.get_pair(frag_1, frag_2) # from left to right
pair = s.get_pair(frag_2, frag_1) # from right to left
pair = s.get_pair(frag_1, frag_2, left=False, transpose=False) # from top to bottom
pair = s.get_pair(frag_2, frag_1, left=False, transpose=False) # from bottom to top
pair = s.get_pair(frag_1, frag_2, left=False) # from top to bottom and then transposed 90 degrees clockwise
pair = s.get_pair(frag_2, frag_1, left=False) # from bottom to top and then transposed 90 degrees clockwise
```
