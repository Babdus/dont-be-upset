# dont-be-upset

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
