"""Transactional budgets, immutable jobs and revisioned derived records."""
from __future__ import annotations
from contextlib import contextmanager
import collections
import json
import os
from pathlib import Path
import sqlite3
import time
import uuid
from research1.data_pipeline.util import canonical,digest

class BudgetExceeded(RuntimeError): pass
class ControllerBusy(RuntimeError): pass

class Ledger:
    def __init__(self,campaign:Path):
        self.campaign=Path(campaign);self.campaign.mkdir(parents=True,exist_ok=True)
        self.path=self.campaign/'ledger.sqlite3'
        protocol=json.loads((self.campaign/'protocol.json').read_text(encoding='utf-8'))
        self.budget=protocol['budget']
        self.stage_budget=protocol.get('stage_budget',{})
        with self.connection() as conn:
            conn.executescript('''
            CREATE TABLE IF NOT EXISTS jobs (
                key TEXT PRIMARY KEY,spec TEXT NOT NULL,spec_hash TEXT NOT NULL,
                kind TEXT NOT NULL,status TEXT NOT NULL,owner TEXT,result TEXT,
                created REAL NOT NULL,updated REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS events (
                id TEXT PRIMARY KEY,kind TEXT NOT NULL,payload TEXT NOT NULL,created REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS records (
                kind TEXT NOT NULL,key TEXT NOT NULL,revision INTEGER NOT NULL,payload TEXT NOT NULL,
                PRIMARY KEY(kind,key));
            CREATE TABLE IF NOT EXISTS record_history (
                kind TEXT NOT NULL,key TEXT NOT NULL,revision INTEGER NOT NULL,payload TEXT NOT NULL,
                created REAL NOT NULL,PRIMARY KEY(kind,key,revision));
            ''')

    @contextmanager
    def connection(self):
        conn=sqlite3.connect(self.path,timeout=30,isolation_level=None)
        conn.row_factory=sqlite3.Row
        conn.execute('PRAGMA journal_mode=WAL');conn.execute('PRAGMA synchronous=FULL')
        try: yield conn
        finally: conn.close()

    @contextmanager
    def controller(self,name='controller'):
        if name not in {'controller','launcher'}:raise ValueError('Unknown lock name')
        handle=(self.campaign/(name+'.lock')).open('a+b')
        handle.seek(0)
        if os.name=='nt':
            import msvcrt
            handle.write(b'0');handle.flush();handle.seek(0)
            try: msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
            except OSError:
                handle.close();raise ControllerBusy('Another controller owns this campaign')
        else:
            import fcntl
            try: fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                handle.close();raise ControllerBusy('Another controller owns this campaign')
        owner=str(os.getpid())+':'+uuid.uuid4().hex
        try: yield owner
        finally:
            if os.name=='nt':
                handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
            else: fcntl.flock(handle.fileno(),fcntl.LOCK_UN)
            handle.close()

    def get_job(self,key):
        with self.connection() as conn:
            row=conn.execute('SELECT * FROM jobs WHERE key=?',(key,)).fetchone()
        if row is None:return None
        data=dict(row);data['spec']=json.loads(data['spec'])
        data['result']=json.loads(data['result']) if data['result'] is not None else None
        return data

    def reserve_job(self,key,spec,kind='regular'):
        if kind not in {'regular','retry'}:raise ValueError('Unknown job budget kind')
        encoded=canonical(spec)
        with self.connection() as conn:
            conn.execute('BEGIN IMMEDIATE')
            try:
                old=conn.execute('SELECT spec FROM jobs WHERE key=?',(key,)).fetchone()
                if old:
                    if old['spec']!=encoded:raise ValueError('EvaluationSpec changed for immutable job '+key)
                    conn.commit();return False
                if conn.execute("SELECT 1 FROM records WHERE kind='budget_handover' AND key='successor'").fetchone():
                    raise BudgetExceeded('Budget authorization transferred to the registered successor')
                used=dict(conn.execute('SELECT kind,COUNT(*) FROM jobs GROUP BY kind').fetchall())
                limit=self.budget['regular' if kind=='regular' else 'retries']
                if used.get(kind,0)>=limit or sum(used.values())>=self.budget['total']:
                    raise BudgetExceeded('Frozen task budget exhausted')
                role=spec.get('role')
                if kind=='regular' and role in self.stage_budget:
                    stage_used=sum(json.loads(r[0]).get('role')==role for r in conn.execute("SELECT spec FROM jobs WHERE kind='regular'"))
                    if stage_used>=self.stage_budget[role]:raise BudgetExceeded('Frozen stage budget exhausted: '+role)
                now=time.time()
                conn.execute('INSERT INTO jobs VALUES(?,?,?,?,?,?,?,?,?)',
                    (key,encoded,digest(spec),kind,'READY',None,None,now,now))
                conn.commit();return True
            except BaseException:
                conn.rollback();raise

    def start_job(self,key,owner):
        with self.connection() as conn:
            conn.execute('BEGIN IMMEDIATE')
            row=conn.execute('SELECT status,owner FROM jobs WHERE key=?',(key,)).fetchone()
            if row is None:conn.rollback();raise KeyError(key)
            if row['status']=='RUNNING' and row['owner']==owner:
                conn.commit();return
            if row['status']!='READY':conn.rollback();raise ValueError('Cannot relaunch '+row['status']+' job '+key)
            conn.execute('UPDATE jobs SET status=?,owner=?,updated=? WHERE key=?',('RUNNING',owner,time.time(),key))
            conn.commit()

    def _finish(self,key,result,status):
        payload=canonical(result)
        with self.connection() as conn:
            conn.execute('BEGIN IMMEDIATE')
            old=conn.execute('SELECT status,result FROM jobs WHERE key=?',(key,)).fetchone()
            if old is None:conn.rollback();raise KeyError(key)
            if old['status'] in {'COMPLETE','INCOMPLETE'}:
                if old['status']!=status or old['result']!=payload:
                    conn.rollback();raise ValueError('Cannot overwrite immutable completed job '+key)
                conn.commit();return
            conn.execute('UPDATE jobs SET status=?,result=?,updated=? WHERE key=?',(status,payload,time.time(),key));conn.commit()

    def complete_job(self,key,result): self._finish(key,result,'COMPLETE')
    def fail_job(self,key,result): self._finish(key,result,'INCOMPLETE')

    def add_event(self,kind,payload,event_id=None):
        event_id=event_id or digest({'kind':kind,'payload':payload})
        data=canonical(payload)
        with self.connection() as conn:
            old=conn.execute('SELECT kind,payload FROM events WHERE id=?',(event_id,)).fetchone()
            if old and (old['kind']!=kind or old['payload']!=data):raise ValueError('Event identity collision')
            conn.execute('INSERT OR IGNORE INTO events VALUES(?,?,?,?)',(event_id,kind,data,time.time()))
        return event_id

    def put_record(self,kind,key,payload):
        data=canonical(payload)
        with self.connection() as conn:
            conn.execute('BEGIN IMMEDIATE')
            old=conn.execute('SELECT revision,payload FROM records WHERE kind=? AND key=?',(kind,key)).fetchone()
            if old and old['payload']==data:conn.commit();return old['revision']
            revision=old['revision']+1 if old else 1
            conn.execute('INSERT INTO record_history VALUES(?,?,?,?,?)',(kind,key,revision,data,time.time()))
            conn.execute('INSERT OR REPLACE INTO records VALUES(?,?,?,?)',(kind,key,revision,data));conn.commit()
        return revision

    def records(self,kind):
        with self.connection() as conn:
            return [json.loads(r[0]) for r in conn.execute('SELECT payload FROM records WHERE kind=? ORDER BY key',(kind,))]

    def record_items(self,kind):
        with self.connection() as conn:
            return [(r['key'],json.loads(r['payload'])) for r in conn.execute('SELECT key,payload FROM records WHERE kind=? ORDER BY key',(kind,))]

    def revoke(self,reference,reason):
        changed=[];invalid={reference}
        progress=True
        while progress:
            progress=False
            for kind in ('samples','values','comparisons'):
                for key,row in self.record_items(kind):
                    refs=set(row.get('source_refs',[]))|set(row.get('candidate_ids',[]))
                    refs|={str(row.get(field,'')) for field in ('candidate_id','sample_id','plan_id','state_id','root_lineage_id')}
                    if row.get('revoked') or not ({key}|refs)&invalid:continue
                    row.update(revoked=True,revocation_reason=reason)
                    self.put_record(kind,key,row);changed.append(kind+':'+key)
                    invalid.add(key);invalid.update(str(row.get(f,'')) for f in ('candidate_id','sample_id','plan_id') if row.get(f))
                    progress=True
        self.add_event('revocation',{'reference':reference,'reason':reason,'changed':changed})
        return changed

    def budget_remaining(self,kind):
        if kind not in {'regular','retry'}:raise ValueError(kind)
        with self.connection() as conn:
            used=dict(conn.execute('SELECT kind,COUNT(*) FROM jobs GROUP BY kind').fetchall())
        return max(0,min(self.budget['regular' if kind=='regular' else 'retries']-used.get(kind,0),self.budget['total']-sum(used.values())))

    def status(self):
        with self.connection() as conn:
            statuses=dict(conn.execute('SELECT status,COUNT(*) FROM jobs GROUP BY status').fetchall())
            used=dict(conn.execute('SELECT kind,COUNT(*) FROM jobs GROUP BY kind').fetchall())
            records=dict(conn.execute('SELECT kind,COUNT(*) FROM records GROUP BY kind').fetchall())
        return {'jobs':statuses,'budget':{'limits':self.budget,'used':used,'remaining_regular':self.budget_remaining('regular'),
                'remaining_retry':self.budget_remaining('retry')},'records':records}
