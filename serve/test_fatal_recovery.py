"""Real pipe/process protocol regression for a native ERR that leaves its process alive."""
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
import json
import urllib.error
import urllib.request
from serve.server import StrataEngine, Service, ChatTemplate, EngineDied, EngineFault, serve
from serve.test_server import ByteTokenizer

FAKE=r'''
import sys
from pathlib import Path
state=Path(sys.argv[sys.argv.index('--state')+1])
mode=sys.argv[sys.argv.index('--mode')+1]
print('READY 4096 stop',flush=True)
for line in sys.stdin:
 if line.startswith('QUIT'):break
 if line.startswith(('GEN','BGEN')):
  if not state.exists():
   state.write_text('first request was attempted')
   if mode=='invalid':print('ERR prompt exceeds context',flush=True)
   elif mode=='dies':   # todo 7: a prefill error the engine cannot go on from: ERR, then it unwinds and exits
    print('ERR prefill: growing the hand-off buffer to 16384 rows: no error',flush=True)
    import time;time.sleep(0.2);sys.exit(1)
   elif mode=='fault':print('ERR prompt read failed: prefill: the hand-off buffer could not grow to 16384 rows (no RAM)',flush=True)
   else:
    kind='verify batch' if mode=='batch' else 'verify'
    print(f'ERR {kind}: timed out at layer 18; its GPU waits were released and the GPU finished (#267)',flush=True)
    print('DONE 0 1 0 1 error',flush=True)
   continue
  print('T 90\nDONE 1 1 0 1 stop',flush=True)
'''

class FatalRecovery(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);py=self.root/'native.py'
        if os.name=='nt':   # no shebangs on Windows: a .cmd runs the same fake engine
            py.write_text(FAKE);self.script=self.root/'native.cmd'
            self.script.write_text(f'@"{sys.executable}" "{py}" %*\r\n')
        else:
            self.script=py;py.write_text('#!'+sys.executable+'\n'+FAKE);py.chmod(0o700)
    def engine(self,mode='fatal'):
        e=StrataEngine(str(self.script),['--state',str(self.root/'state'),'--mode',mode])
        self.addCleanup(e.close);return e
    def service(self,e):
        return Service(e,ByteTokenizer(),ChatTemplate(Path(__file__).parent/'chat_template.jinja'))
    def wait_dead(self,e):
        end=time.monotonic()+2
        while e.alive() and time.monotonic()<end:time.sleep(.01)
        self.assertFalse(e.alive(),'unrecoverable native ERR must invalidate health before a new request')
    def run_request(self,svc):
        return list(svc.run([1],False,[],8,{},threading.Event()))
    def test_fatal_error_is_not_a_success_and_next_request_reloads(self):
        e=self.engine();svc=self.service(e)
        with self.assertRaises(EngineDied):self.run_request(svc)
        self.wait_dead(e)
        self.assertIs(svc.engine,e,'the failed request must not be retried automatically')
        self.assertEqual(svc.history[-1]['finish'],'error')
        result=self.run_request(svc);self.addCleanup(svc.engine.close)
        self.assertTrue(svc.engine.alive())
        self.assertTrue(svc.loaded());self.assertEqual(result[-1][1]['completion_tokens'],1)
    def test_http_health_and_next_request_recovery(self):
        old=self.engine();svc=self.service(old);http=serve(svc,port=0)
        base=f'http://127.0.0.1:{http.server_address[1]}'
        body={'messages':[{'role':'user','content':'hi'}],'max_tokens':8,
              'chat_template_kwargs':{'enable_thinking':False},'stream':True}
        try:
            def post():
                req=urllib.request.Request(base+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
                with urllib.request.urlopen(req,timeout=5) as r:return r.read().decode()
            text=post()
            self.assertIn('"error"',text);self.assertTrue(text.rstrip().endswith('data: [DONE]'))
            with urllib.request.urlopen(base+'/health',timeout=5) as r:self.assertFalse(json.load(r)['loaded'])
            body['stream']=False
            self.assertEqual(json.loads(post())['usage']['completion_tokens'],1)
            with urllib.request.urlopen(base+'/health',timeout=5) as r:self.assertTrue(json.load(r)['loaded'])
        finally:
            http.shutdown();http.server_close();svc.engine.close()
    def test_batch_timeout_signature_also_invalidates_engine(self):
        e=self.engine('batch')
        with self.assertRaises(EngineDied):list(e.generate([1],8,{},threading.Event()))
        self.wait_dead(e)

    def feed(self,e,lines,slots):
        import queue
        class P:
            stdout=iter(lines)
            terminated=False
            def terminate(self):self.terminated=True
        old=e.proc;e.proc=P();e.lines=queue.Queue();e.slot_q=[queue.Queue() for _ in range(slots)];e.ended=False
        e._pump()
        self.assertTrue(e.ended);self.assertTrue(e.proc.terminated)
        self.assertIsNone(e.lines.get_nowait());self.assertTrue(e.lines.empty())   # EOF, never the trailing DONE
        e.proc=old
    def test_fatal_lines_are_checked_before_batch_routing(self):
        e=self.engine('fatal')
        self.feed(e,['ERR verify batch: timed out at layer 3\n','BDONE 0 1 0 1 error\n'],2)
        self.assertTrue(all(q.get_nowait() is None and q.empty() for q in e.slot_q))   # only the EOF reached a slot, not the BDONE
    def test_an_earlier_window_never_finished_is_fatal(self):
        e=self.engine('fatal')
        self.feed(e,['ERR verify: an earlier window never finished on the GPU (#267); restart the engine\n','DONE 0 1 0 1 error\n'],0)
        self.feed(e,['ERR verify batch: an earlier window never finished on the GPU (#267)\n'],2)
    def test_generate_batched_refuses_a_dead_engine(self):
        e=self.engine('fatal');e.close()
        with self.assertRaises(EngineDied):list(e.generate_batched([1],8,{},threading.Event()))

    def test_validation_error_does_not_kill_engine(self):
        e=self.engine('invalid')
        with self.assertRaises(ValueError):list(e.generate([1],8,{},threading.Event()))
        self.assertTrue(e.alive())
        self.assertEqual(list(e.generate([1],8,{},threading.Event())),[90])
    def test_err_then_exit_is_a_server_error_and_restarts_at_once(self):
        # todo 7 (Fable stall5 section 3): a 503 (EngineDied), not a 400, and the engine starts again without waiting
        # for the next request
        e=self.engine('dies');svc=self.service(e);gen0=e.gen
        with self.assertRaises(EngineDied):self.run_request(svc)
        self.assertEqual(svc.history[-1]['finish'],'error')
        end=time.monotonic()+10
        while not (e.gen>gen0 and e.alive()) and time.monotonic()<end:time.sleep(.05)
        self.assertTrue(e.gen>gen0 and e.alive(),'the engine must be started again at once, not by the next request')
        self.assertEqual(self.run_request(svc)[-1][1]['completion_tokens'],1)
    def test_engine_fault_that_goes_on_is_a_server_error_on_the_same_process(self):
        e=self.engine('fault');svc=self.service(e);pid=e.proc.pid
        with self.assertRaises(EngineFault):self.run_request(svc)
        self.assertTrue(e.alive());self.assertEqual(e.proc.pid,pid)
        self.assertEqual(self.run_request(svc)[-1][1]['completion_tokens'],1)
        self.assertEqual(e.proc.pid,pid,'the next request is served by the same engine process')
    def test_http_status_of_an_engine_error(self):
        for mode,code in (('dies',503),('fault',503),('invalid',400)):
            (self.root/'state').unlink(missing_ok=True)
            e=self.engine(mode);e.ERR_EXIT_WAIT_S=0.3;svc=self.service(e);http=serve(svc,port=0)
            body={'messages':[{'role':'user','content':'hi'}],'max_tokens':8,'chat_template_kwargs':{'enable_thinking':False}}
            req=urllib.request.Request(f'http://127.0.0.1:{http.server_address[1]}/v1/chat/completions',
                                       data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
            try:
                with self.assertRaises(urllib.error.HTTPError) as cm:urllib.request.urlopen(req,timeout=10)
                self.assertEqual(cm.exception.code,code,mode)
                kind=json.loads(cm.exception.read())['error']['type']
                self.assertEqual(kind,'server_error' if code>=500 else 'invalid_request_error',mode)
                end=time.monotonic()+10   # (the restart 'dies' starts: done before the engine is closed)
                while mode=='dies' and not (svc.engine.gen>1 and svc.engine.alive()) and time.monotonic()<end:time.sleep(.05)
            finally:
                http.shutdown();http.server_close();svc.engine.close()
    def test_plain_err_keeps_its_400_when_the_engine_goes_on(self):
        e=self.engine('invalid');e.ERR_EXIT_WAIT_S=0.3;svc=self.service(e);pid=e.proc.pid
        with self.assertRaises(ValueError) as cm:self.run_request(svc)
        self.assertNotIsInstance(cm.exception,(EngineDied,EngineFault))
        self.assertTrue(e.alive());self.assertEqual(e.proc.pid,pid)
    def test_image_request_errs_do_not_wait_for_an_exit(self):
        # Fable item 7 S1: the GENI refusals leave the engine waiting for its next command, so no 10 s exit wait
        e=self.engine('invalid');e.ERR_EXIT_WAIT_S=5.0
        for m in ('cannot open /tmp/x.emb','bad embeddings file (expected strata-vision records of width 4096)',
                  'short embeddings file','the prompt has more images than the embeddings file',
                  'image 2 has 196 rows but the prompt holds 256 image tokens','the embeddings file has more images than the prompt',
                  'the prompt cannot end in an image','the image position upload failed'):
            t0=time.monotonic();self.assertFalse(e.ended_after_err(m),m);self.assertLess(time.monotonic()-t0,0.5,m)
        self.assertFalse(StrataEngine.ERR_GOES_ON_RE.match('imagery is not supported'))
    def test_closed_serial_engine_rejects_without_accessing_closed_pipe(self):
        e=self.engine();e.close()
        with self.assertRaises(EngineDied):list(e.generate([1],8,{},threading.Event()))
if __name__=='__main__':unittest.main()
