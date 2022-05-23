# OPTICS clustering algorithm

### Data Mining and Data Wharehousing

*Ordering points to identify the clustering structure (OPTICS)* is an algorithm for finding density-based clusters in spatial data. It was presented by Mihael Ankerst, Markus M. Breunig, Hans-Peter Kriegel and Jörg Sander. Its basic idea is similar to DBSCAN, but it addresses one of DBSCAN's major weaknesses: the problem of detecting meaningful clusters in data of varying density.[^1]

[^1]: https://en.wikipedia.org/wiki/OPTICS_algorithm

Steps to run:

`
python3 -m pip install -r requirements.txt
python3 main.py
`