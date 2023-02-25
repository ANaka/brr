# FN—File Names

## note
This is forked from [Anders Hoff's Fn](https://github.com/inconvergent/fn)

## What is it?

This is a tiny library to generate file names.

It will give you unique file names based on current git commit, as well as the
time and date.


## Why?

I have a lot of projects where I make large amounts of files (images, 3D
models, 2D vector files), and I've always wanted a more efficient way of
maintaining unique file names.

I got the idea for this when I saw how Vera Molar names her works in this
Periscope video https://twitter.com/inconvergent/status/700341427344113665


## Dependecies

The code runs on Linux (only, probably) and requires `git` to be installed. It
also uses `docopt`


## Does it guarantee unique file names in any way?

No. It only uses the current time to make a relatively distinct string—don't
use this for anything remotely important.


## On Use and Contributions

This code is a tool that I have written for my own use. I release it publicly
in case people find it useful. It is not however intended as a
collaboration/Open Source project. As such I am unlikely to accept PRs, reply
to issues, or take requests.
