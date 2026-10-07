"""Storage-only bounded aggregates and lossless disk spools."""
import json
import math
import random
import sqlite3
import statistics
import threading
from collections import deque


class BoundedStats:
    def __init__(self, capacity=1024):
        self.capacity=capacity
        self.values=[]
        self.calls=0
        self.total=0.
        self.minimum=self.maximum=self.last=None
        self.random=random.Random(0)

    def append(self, value):
        self.calls+=1
        self.total+=value
        self.last=value
        self.minimum=value if self.minimum is None else min(self.minimum,value)
        self.maximum=value if self.maximum is None else max(self.maximum,value)
        if len(self.values)<self.capacity:
            self.values.append(value)
        else:
            i=self.random.randrange(self.calls)
            if i<self.capacity:self.values[i]=value

    def __len__(self):return self.calls
    def __iter__(self):return iter(self.values)
    def __getitem__(self,index):
        if index == -1:return self.last
        return self.values[index]

    def distribution(self):
        values=sorted(self.values)
        return dict(calls=self.calls,total=self.total,mean=self.total/self.calls if self.calls else None,
            min=self.minimum,max=self.maximum,median=statistics.median(values) if values else None,
            p95=values[max(0,math.ceil(len(values)*.95)-1)] if values else None,
            p99=values[max(0,math.ceil(len(values)*.99)-1)] if values else None,
            quantiles_approximate=self.calls>self.capacity,
            quantile_method='deterministic_uniform_reservoir',quantile_sample_size=len(values),
            quantile_capacity=self.capacity)


class DiskRecords:
    """Lossless buffered JSONL; only the last two records are retained in RAM."""
    def __init__(self,path):
        self.path=path
        self.file=path.open('w',encoding='utf-8',buffering=65536)
        self.tail=deque(maxlen=2)
        self.count=0
        self.lock=threading.RLock()

    def append(self,value):
        with self.lock:
            self.file.write(json.dumps(value,ensure_ascii=False)+'\n')
            self.tail.append(value)
            self.count+=1

    def __len__(self):return self.count
    def __getitem__(self,index):return self.tail[index]
    def __iter__(self):
        with self.lock:self.file.flush()
        with self.path.open(encoding='utf-8') as stream:
            for line in stream:yield json.loads(line)
    def close(self):
        with self.lock:
            if not self.file.closed:self.file.close()


class DiskIntervals:
    """Exact intervals, batch-persisted; final unions use disk-backed sorting."""
    def __init__(self,path):
        self.db=sqlite3.connect(path)
        # Diagnostic spool, not canonical evidence: bounded WAL batches avoid
        # a synchronous journal flush per instrumentation batch.
        self.db.execute('pragma journal_mode=WAL')
        self.db.execute('pragma synchronous=NORMAL')
        self.db.execute('pragma temp_store=FILE')
        self.db.execute('create table intervals(a real,b real,name text,kind text)')
        self.pending=[]
        self.count=0
        self.indexed=False
    def append(self,value):
        self.pending.append(value)
        self.count+=1
        if len(self.pending)>=2048:self.flush()
    def flush(self):
        if self.pending:
            self.db.executemany('insert into intervals values(?,?,?,?)',self.pending)
            self.db.commit()
            self.pending.clear()
    def __len__(self):return self.count
    def names(self):
        self.flush()
        return [r[0] for r in self.db.execute("select distinct name from intervals where kind!='container'")]
    def union(self,lo,hi,*,kinds=None,names=None,suffix=None,exclude_container=False):
        self.flush()
        if not self.indexed:
            self.db.execute('create index interval_order on intervals(a)')
            self.db.execute('create index interval_stage_order on intervals(name,a)')
            self.db.commit()
            self.indexed=True
        where=['b>?','a<?'];params=[lo,hi]
        if kinds:
            where.append('kind in ('+','.join('?' for _ in kinds)+')');params.extend(kinds)
        if names:
            where.append('name in ('+','.join('?' for _ in names)+')');params.extend(names)
        if suffix:
            where.append('substr(name,-?)=?');params.extend([len(suffix),suffix])
        if exclude_container:where.append("kind!='container'")
        end=float('-inf');total=0.
        for a,b in self.db.execute('select a,b from intervals where '+' and '.join(where)+' order by a',params):
            a=max(a,lo);b=min(b,hi)
            if b>a and b>end:total+=b-max(a,end);end=b
        return total
    def close(self):self.flush();self.db.close()


class HotHistory:
    """Cumulative identity count plus only the not-yet-committed payloads."""
    def __init__(self):self.items=[];self.count=0;self.base=0
    def append(self,value):self.items.append(value);self.count+=1
    def __len__(self):return self.count
    def __iter__(self):return iter(self.items)
    def __getitem__(self,index):
        if isinstance(index,slice):
            start=0 if index.start is None else index.start
            if start<self.base:raise IndexError('Committed history is in SQLite')
            stop=None if index.stop is None else index.stop-self.base
            return self.items[start-self.base:stop:index.step]
        return self.items[index-self.base]
    def release(self):self.items.clear();self.base=self.count


class StatePool:
    """Exact route-local hash index; payloads fetched by ID after commit."""
    def __init__(self,lookup):self.index={};self.lookup=lookup
    def get(self,key):
        value=self.index.get(key)
        return self.lookup(value) if isinstance(value,str) else value
    def __setitem__(self,key,value):self.index[key]=value
    def release(self,state):
        self.index[((state['height'],state['width']),state['exact_hash'])]=state['state_id']


class RoutedRecords:
    def __init__(self,records,route):self.records=records;self.route=route
    def append(self,value):self.records.append(dict(value,table_id=self.route['table_id'],seat_id=self.route['seat_id']))
