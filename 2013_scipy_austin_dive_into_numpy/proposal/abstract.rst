Title
=====

Diving into NumPy code

Author(s)
=========

David Cournapeau and Stéfan Van der Walt (Stéfan to be confirmed)

Level
=====

Intermediate/advanced

Audience
========

anyone who is interested in contributing to core NumPy but does not know how to
start, wants to extend NumPy, or is simply interested in understanding NumPy
implementation details

Description
===========

Do you want to contribute to NumPy but find the codebase daunting ? Do you want
to extend NumPy (e.g. adding support for decimal, or arbitrary precision) ? Are
you curious to understand how NumPy works at all ? Then this tutorial is for
you.

The goal of this tutorial is do dive into NumPy codebase, in particular the
core C implementation. You will learn how to build NumPy from sources, how some
of the core concepts such as data types and ufuncs are implemented at the C
level and how it is hooked up to the Python runtime. You will also learn how to
add a new ufunc and a new data type.

During the tutorial, we will also have a look at various tools (unix-oriented)
that can help tracking bugs or follow a particular numpy expression from its
python representation to its low-level implementation.

While a working knowledge of C and Python is required, we do not assume a
preliminary knowledge of the NumPy codebase. An understanding of Python C
extensions is a plus, but not required either.

Outline
=======

The tutorial will be divided in 3 main sections:

        0. Introduction:
          
                - Why extending numpy in C ? (and perhaps more importantly,
                  when you should not)
                - being ready to develop on NumPy: building from sources, and
                  building with different flags (optimisation and debug)

        1. Source code organisation: description of the numpy source tree and
           high-level description of what belongs where: core vs the rest,
           core.multiarray, core.ufunc, scalar arrays and support libraries
           (npysort, npymath)

        2. The main data structures around ndarray: 

                - the arrayobject and data type descriptor, and how they relate
                  to each other.
                - exercise to add a simple array method to the array object
                - dealing with arbitrary array memory layout with iterators

        3. Adding a new dtype:
          
                - Anatomy of the dtype: from a + a to a core C loop
                - Simple example to wrap a software implementation of quadruple
                  precision (revised version of IEEE 754 software)


Exercises
=========

The current set of planned hand-on tasks/exercises:

        - building from sources with debug symbols
        - adding an array method to compute a simple statistic (e.g. kurtosis)
        - adding a new type to handle quadruple precision type

Packages/pre-requisites
=======================

        - You will need a working C compiler (gcc on unix/os x, Visual Studio
          2008 on windows), and be familiar how to use it on your platform
        - git
        - if possible, gdb and cgdb on unix

We may provide a ready to use vagrant Ubuntu VM with everything set-up.
