import collections

class Counter(collections.Counter):
    @property
    def __values(self):
        return self.keys()

    @property
    def __total(self):
        return sum(self.values())

    def count(self, value):
        current = self.get(value, 0)
        self[value] = current + 1

    def count_if(self, value, f):
        if f(value):
            self.count(value)

    def get_avg(self):
        return len(self.__values) / max(1, self.__total)

    def get_total(self):
        return self.__total

    def get_count(self):
        return len(self.__values)

class Metrics:
    def __init__(self):
        self.entity_counter = Counter()
        self.relation_counter = Counter()
        self.triple_counter = Counter()

    def count(self, objs, f=lambda t: True):
        counts = 0
        if not isinstance(objs, list):
            objs = [objs]
        for obj in objs:
            matches = 0
            if f('object'):
                self.entity_counter.count(obj['object'])
                matches += 1
            if f('subject'):
                self.entity_counter.count(obj['subject'])
                matches += 1
            if f('relation'):
                self.relation_counter.count(obj['relation'])
                matches += 1
            counts += matches
            if matches == 3:
                self.triple_counter.count(obj['object'] + '::' + obj['relation'] + '::' + obj['subject'])
        return counts

    def count_if(self, objs1, objs2, f):
        matches = []
        for i, obj1 in enumerate(objs1):
            if i in matches:
                continue
            for j, obj2 in enumerate(objs2):
                if j in matches:
                    continue
                if self.count([obj1], f=lambda t: f(obj1, obj2, t)) > 0:
                    matches.append(i)
                    matches.append(j)
                    break

    def get_count(self):
        return {'entity': self.entity_counter.get_count(),
                'relation': self.relation_counter.get_count(),
                'triple': self.triple_counter.get_count()}

    def get_total(self):
        return {'entity': self.entity_counter.get_total(),
                'relation': self.relation_counter.get_total(),
                'triple': self.triple_counter.get_total()}
    def get_avg(self):
        return {'entity': self.entity_counter.get_avg(),
                'relation': self.relation_counter.get_avg(),
                'triple': self.triple_counter.get_avg()}
