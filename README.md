# brr

## overview

This is my Nth attempt at creating a python-based repo for generative art. It's feeling a bit sispyhean at this point, but I have come to accept that there is something very special and gratifying about starting over from scratch.

The repo is named `brr` because of the sound that my pen plotter makes.

The overall structure is loosely a monorepo with multiple python packages managed by poetry. The packages are split into two categories: libraries and packages. Libraries are intended to contain re-usable code, while packages are intended to contain code that is specific to a particular project. The libraries are intended to be installed as dependencies of the packages. I picked this approach for the following reasons:
1. it allows me to consolidate a handful of small personal repos into one place
2. I have found myself using multiple virtual environments for generative art, with the biggest split being pytorch vs. non-pytorch. This approach allows me to handle that more explicitly
3. I'm hoping that this will make it easier to get into a flow where I make more small projects rather than having one giant project with many disparate experiments
