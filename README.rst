.. -*- mode: rst -*-

|Python35|

.. |Python35| image:: https://img.shields.io/badge/python-3.5-blue.svg
.. _Python35: https://badge.fury.io/py/scikit-learn

2bornot2b
=========

Hacker rank challenge codes: https://www.hackerrank.com/challenges/to-be-what

**Goal**: determine the "be" verb conjugation in a text, among 8 possible forms: 'am','are','were','was','is','been','being','be'.

Two models are implemented here:

* A **regularized logistic regression using TF-IDF** representation of left and right words to predict the correct form, in the 2bornot2b.ipynb notebook
* A **Bidirectional-RNN**. The later is home-made and does not rely on TensorFlow or Theano, as those libraires are not available on HackerRank.

Note that the best scoring submissions rely on rule-based algorithm rather that learning ones, the purpose here is mainly exploratory.


Installation
------------

* Create a virtual environment: ``conda create -n 2bornot2b python=3.5``
* Activate virtual environment: ``source activate 2bornot2b``
* Install requirements: ``pip install -r requirements.txt``

Submission on HackerRank
------------------------

In order to make a submission you can use the nb2script.sh script that converts notebooks to script and remove all print statements but the last, in order to be readily compatible with the HackerRank submission interface.

Remarks
-------

* Various difficulties in English conjugation are highlighted `here <http://grammar.ccc.commnet.edu/grammar/to_be.htm>`_.
* The Custom RNN implementation is inspired by the `WildML <http://www.wildml.com>`_ blog.


