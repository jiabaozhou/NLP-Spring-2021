from collections import Counter
from utils import *


class Test(object):
    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer


dog = Test(Indexer())

dog_indexer = dog.get_indexer()

dog_indexer.add_and_get_index('hello')
dog_indexer.add_and_get_index('not a problem')

print(dog_indexer.index_of('hello'))
print(dog_indexer.index_of('ni hao'))
print(dog_indexer.index_of('not a problem'))


a = Counter()

a.update([1])

a.update([1])

a.update([2])

for key in

print(a)