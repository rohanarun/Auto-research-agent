"""Mechanical fixtures, not claims about LLM research quality."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import shutil
import unittest
import time
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location('methods', ROOT / 'Resources/ResearchMethods.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class MethodsTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp(prefix='methods-test-', dir=ROOT / 'tmp'))
        self.workspace = self.folder / 'source'
        self.workspace.mkdir()
        (self.workspace / 'value').write_text('1')
        (self.workspace / '.ignored-research').write_text('preserve me')
        self.cli = self.folder / 'cli.py'
        self.cli.write_text('''import sys,json,uuid
from pathlib import Path
prompt=sys.stdin.read()
Path('INPUT.txt').write_text(prompt)
Path('ARGS.json').write_text(json.dumps(sys.argv))
if 'RSI-top dashboard' in prompt:
 Path('PORTFOLIO.json').write_text(json.dumps([dict(bottleneck='fixture',mechanism=str(i),expected_effect='increment',lineage='initial',verbalized_probability=.5) for i in range(2)]))
Path('value').write_text(str(int(Path('value').read_text())+1))
print(json.dumps({'type':'thread.started','thread_id':str(uuid.uuid4())}))
print(json.dumps({'type':'turn.completed'}))
''')
        self.evaluator = self.folder / 'evaluator.py'
        self.evaluator.write_text('''import sys,json
from pathlib import Path
p=Path(sys.argv[-1]);s=int((p/'value').read_text())
e=p/'.method-evaluation';e.mkdir(exist_ok=True)
(e/'receipt.json').write_text(json.dumps({'value':s}))
print(json.dumps({'valid':True,'score':s,'metrics':{'value':s,'novelty':s-1},'evidence':['.method-evaluation/receipt.json']}))
''')
        self.config = dict(provider='codex', executable=sys.executable, arguments=[str(self.cli), '--json', '-'], policy='Policy, not old conversation',
                           options=dict(rsiTop=False, forkAndFlush=True, branches=2, explorationMinutes=.1, exploitationMinutes=.1, rounds=2))
        self.plan = dict(objective='fixture', worker_prompt='Improve value', evaluator=str(self.evaluator), direction='maximize',
                         evaluation_seconds=.5, metrics=[dict(name='value', definition='value'),dict(name='novelty',definition='novelty')],
                         metric_subset=1, portfolio_size=2, stall_seconds=.00001, meaningful_gain=0)

    def run_method(self, suffix='run'):
        runner = m.Runner(self.config, self.plan, self.workspace, self.folder / suffix)
        return runner, runner.run()

    def test_gateway_secret_is_inherited_but_redacted_before_branch_log_write(self):
        runner = m.Runner(self.config, self.plan, self.workspace, self.folder / 'redaction')
        log = self.folder / 'gateway.log'
        with mock.patch.dict(os.environ, {'OPENROUTER_API_KEY': 'fixture-method-secret'}):
            code, timed_out = runner.process([sys.executable, '-c', "import os,sys;print(os.environ['OPENROUTER_API_KEY']);print(os.environ['OPENROUTER_API_KEY'],file=sys.stderr)"], self.workspace, '', log, 10)
        self.assertEqual(code, 0)
        self.assertFalse(timed_out)
        self.assertNotIn('fixture-method-secret', log.read_text())
        self.assertEqual(log.read_text().count('[REDACTED]'), 2)

    def test_completed_parent_does_not_leave_pipe_holding_grandchild(self):
        runner = m.Runner(self.config, self.plan, self.workspace, self.folder / 'drain')
        script = "import subprocess,sys;subprocess.Popen([sys.executable,'-c','import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(60)']);print('parent finished',flush=True)"
        started = time.monotonic()
        code, timed_out = runner.process([sys.executable, '-c', script], self.workspace, '', self.folder / 'drain.log', 10)
        self.assertEqual(code, 0)
        self.assertFalse(timed_out)
        self.assertLess(time.monotonic() - started, 5)
        self.assertIn('parent finished', (self.folder / 'drain.log').read_text())

    def test_forks_are_fresh_leader_resumes_and_source_is_preserved(self):
        runner, result = self.run_method()
        self.assertEqual(result['status'], 'COMPLETED')
        self.assertEqual((self.workspace / 'value').read_text(), '1')
        self.assertEqual(result['leader']['evaluation']['score'], '5')
        self.assertEqual(len(runner.rows), 7)
        for row in runner.rows[1:]:
            p = Path(row['workspace'])
            self.assertEqual((p/'.ignored-research').read_text(), 'preserve me')
            args = json.loads((p/'ARGS.json').read_text())
            self.assertEqual('resume' in args, row['phase']=='continue')
            self.assertNotIn('old conversation transcript', (p/'INPUT.txt').read_text())

    def test_rsi_and_combination_keep_fixed_dashboard_and_partial_views(self):
        self.config['options']['rsiTop'] = True
        for combined in [False, True]:
            self.config['options']['forkAndFlush'] = combined
            runner, result = self.run_method(str(combined))
            self.assertEqual(result['status'], 'COMPLETED')
            for row in runner.rows[1:]:
                if row['phase']=='explore':
                    launch=json.loads((Path(row['workspace']).parent/'LAUNCH.json').read_text())
                    self.assertEqual(len(launch['metric_view']['definitions']),1)
                    self.assertIn('verbalized_probability', (Path(row['workspace'])/'INPUT.txt').read_text())
            self.assertEqual(json.loads((runner.output/'PLAN.json').read_text()),self.plan)

    def test_rsi_alone_actually_intervenes_after_a_flat_completed_turn(self):
        self.config['options'].update(rsiTop=True,forkAndFlush=False)
        self.evaluator.write_text(self.evaluator.read_text().replace("'score':s", "'score':1"))
        runner,result=self.run_method('rsi-stall')
        probes=[r for r in runner.rows[1:] if r['phase']=='explore']
        self.assertEqual(len(probes),2)
        self.assertTrue(all(r['portfolio_sha256'] for r in probes))
        views=[json.loads((runner.output/r['id']/'LAUNCH.json').read_text())['metric_view'] for r in probes]
        self.assertNotEqual(views[0]['definitions'],views[1]['definitions'])
        self.assertTrue(all(any(d['distinct_values']>=2 for d in v['diagnostics']) for v in views))

    def test_provider_failure_is_not_scored_or_promoted(self):
        self.cli.write_text("import json;print(json.dumps({'type':'error','message':'capacity'}));raise SystemExit(1)")
        runner=m.Runner(self.config,self.plan,self.workspace,self.folder/'failed')
        with self.assertRaisesRegex(RuntimeError,'No provider branch completed'):
            runner.run()
        self.assertTrue(all(not r['evaluation']['valid'] for r in runner.rows[1:]))
        self.assertEqual((self.workspace/'value').read_text(),'1')

    def test_recovered_transport_errors_are_scored_but_terminal_failure_is_not(self):
        self.cli.write_text(self.cli.read_text().replace("print(json.dumps({'type':'turn.completed'}))", "print(json.dumps({'type':'error','message':'Reconnecting after high demand'}))\nprint(json.dumps({'type':'turn.completed'}))"))
        runner,result=self.run_method('recovered')
        self.assertGreater(int(result['leader']['evaluation']['score']),1)
        self.assertTrue(all(r['completed'] for r in runner.rows[1:]))
        self.assertTrue(all(r['provider_outcome']['diagnostic_count']==1 for r in runner.rows[1:]))
        events=[{'type':'turn.completed'},{'type':'turn.failed'}]
        self.assertFalse(m.provider_outcome(events,0,False)['completed'])
        self.assertFalse(m.provider_outcome([{'type':'turn.completed'}],1,False)['completed'])
        self.assertFalse(m.provider_outcome([{'type':'turn.completed'}],0,True)['completed'])

    def test_flat_objective_preserves_research_branch_and_resumes_its_session(self):
        self.evaluator.write_text(self.evaluator.read_text().replace("'score':s", "'score':1"))
        runner,result=self.run_method('flat')
        self.assertNotEqual(result['leader']['id'],'initial')
        self.assertEqual(result['leader']['evaluation']['score'],'1')
        continuations=[x for x in runner.rows[1:] if x['phase']=='continue']
        self.assertTrue(all(x['parent']!='initial' for x in continuations))
        self.assertTrue(all('resume' in json.loads((Path(x['workspace'])/'ARGS.json').read_text()) for x in continuations))
        self.assertEqual((self.workspace/'value').read_text(),'1')
        dashboard=json.loads((runner.output/'DASHBOARD.json').read_text())
        self.assertGreater(len(dashboard['metric_diagnostics']),0)

    def test_absent_optional_observations_do_not_invalidate_or_become_zero(self):
        self.plan['metrics'].append(dict(name='trial_gain', definition='Absent before a trial', optional=True))
        runner,result=self.run_method('optional-metrics')
        self.assertTrue(all(r['evaluation']['valid'] for r in runner.rows))
        self.assertTrue(all('trial_gain' not in r['evaluation']['metrics'] for r in runner.rows))
        diagnostic=next(d for d in runner.dashboard()['metric_diagnostics'] if d['name']=='trial_gain')
        self.assertEqual(diagnostic['observations'],0)
        self.assertEqual(diagnostic['missing_observations'],len(runner.rows))
        self.assertFalse(diagnostic['constant_in_observed_checkpoints'])

    def test_missing_required_metric_fails_preflight_before_worker_calls(self):
        self.plan['metrics'].append(dict(name='required_measure', definition='Required observation'))
        with self.assertRaisesRegex(ValueError,'Initial candidate failed frozen evaluation'):
            self.run_method('required-metric')
        self.assertFalse(list((self.folder/'required-metric').glob('round-*')))

    def test_missing_selection_metric_cannot_rank_even_when_optional(self):
        self.plan['metrics'].append(dict(name='trial_gain', definition='Optional diagnostic', optional=True))
        self.plan['selection_metrics']=[dict(name='trial_gain',direction='maximize')]
        with self.assertRaisesRegex(ValueError,'Initial candidate failed frozen evaluation'):
            self.run_method('missing-selection')
        self.assertFalse(list((self.folder/'missing-selection').glob('round-*')))

    def test_verified_progress_breaks_score_ties_without_overriding_primary_score(self):
        runner=m.Runner(self.config,self.plan,self.workspace,self.folder/'selection')
        self.plan['selection_metrics']=[{'name':'novelty','direction':'maximize'}]
        def row(name,primary,progress,valid=True):
            return {'id':name,'evaluation':{'valid':valid,'score':primary,'metrics':{'novelty':progress}},'time':1}
        incumbent=row('initial',1,0)
        self.assertEqual(runner.choose(incumbent,[row('research',1,3)])['id'],'research')
        self.assertEqual(runner.choose(incumbent,[row('worse',0,999)])['id'],'initial')
        self.assertEqual(runner.choose(incumbent,[row('invalid',999,999,False)])['id'],'initial')
        self.plan['direction']='minimize'
        self.assertEqual(runner.choose(incumbent,[row('worse',2,999)])['id'],'initial')

    def test_evaluation_time_is_reserved_before_last_cooperative_launch(self):
        self.collaborative_cli()
        self.config.update(shareContext=True,sharingIntervalMinutes=.01)
        self.config['options'].update(rounds=1,explorationMinutes=.02,exploitationMinutes=.01)
        runner,result=self.run_method('reserved')
        self.assertEqual(result['status'],'COMPLETED')
        budgets=[json.loads(p.read_text()) for p in runner.output.glob('*/BUDGET.json')]
        self.assertTrue(budgets)
        self.assertTrue(all(x['provider_seconds']>0 and x['evaluation_reserve_seconds']==.5 for x in budgets))
        self.assertTrue(all(not x['timed_out'] for x in runner.rows[1:]))

    def test_invalid_higher_score_cannot_replace_incumbent(self):
        self.evaluator.write_text(self.evaluator.read_text().replace("'valid':True", "'valid':s==1"))
        _,result=self.run_method()
        self.assertEqual(result['leader']['id'],'initial')

    def test_checker_mutation_and_nonfinite_scores_rejected(self):
        for bad in ['NaN','Infinity',True]:
            with self.assertRaises(ValueError):m.score(bad)
        self.evaluator.write_text(self.evaluator.read_text().replace("e=p/", "(p/'value').write_text('999')\ne=p/"))
        with self.assertRaisesRegex(ValueError,'workspace changed'):self.run_method()

    def test_symlink_escape_rejected_before_fork(self):
        (self.workspace/'escape').symlink_to(self.evaluator)
        with self.assertRaisesRegex(ValueError,'symlinks'):self.run_method()

    def test_cancellation_and_timeout_preserve_logs(self):
        self.cli.write_text("import time\nfrom pathlib import Path\nPath('value').write_text('999')\ntime.sleep(30)")
        self.config['options']['explorationMinutes']=.005
        _,result=self.run_method()
        self.assertEqual(result['stop_reason'],'phase_budget_reached')
        self.assertEqual(result['leader']['id'],'initial')
        self.assertEqual(result['leader']['evaluation']['score'],'1')
        self.assertEqual(len(result['pending_workspaces']),2)
        self.assertTrue(all((Path(p['workspace'])/'value').read_text()=='999' for p in result['pending_workspaces']))
        self.assertEqual((self.workspace/'value').read_text(),'1')
        logs=list((self.folder/'run').glob('round-*/RESULT.json'))
        self.assertEqual(len(logs),2)
        self.assertTrue(all(json.loads(p.read_text())['timed_out'] for p in logs))

    def test_cloud_bridge_dispatches_owned_method_runner(self):
        bridge_spec=importlib.util.spec_from_file_location('bridge', ROOT/'Resources/DaytonaWorker.py')
        bridge=importlib.util.module_from_spec(bridge_spec);bridge_spec.loader.exec_module(bridge)
        bridge.R=self.folder/'private';bridge.CO=self.workspace
        method_dir=self.workspace/'SHARED_CONTEXT/methods-fixture';method_dir.mkdir(parents=True)
        shutil.copy2(ROOT/'Resources/ResearchMethods.py',method_dir/'ResearchMethods.py')
        (method_dir/'config.json').write_text(json.dumps(self.config))
        (method_dir/'plan.json').write_text(json.dumps(self.plan))
        state=bridge.R/'providers/fixture/state';state.mkdir(parents=True);(state/'auth.json').write_text('{}')
        operation=bridge.R/'operations/method';operation.mkdir(parents=True)
        (operation/'request.json').write_text(json.dumps(dict(state_id='fixture',research_methods=True,arguments=[str(method_dir)],input='')))
        bridge.terminate=lambda p: {'remaining_pids':[]}
        bridge.run('method')
        status=json.loads((operation/'status.json').read_text())
        self.assertEqual(status['exit_code'],0,(operation/'stderr').read_text())
        result=json.loads((operation/'stdout').read_text().splitlines()[-1])
        self.assertEqual(result['status'],'COMPLETED')
        self.assertEqual(result['leader']['evaluation']['score'],'5')


    def collaborative_cli(self, fail_resume=False):
        self.cli.write_text('''import sys,json,uuid,hashlib
from pathlib import Path
prompt=sys.stdin.read();Path('INPUT.txt').write_text(prompt);Path('ARGS.json').write_text(json.dumps(sys.argv))
resumed='resume' in sys.argv
if FAIL and resumed:
 Path('value').write_text('999');print(json.dumps({'type':'error','message':'fixture failed after changing workspace'}));raise SystemExit(1)
session=sys.argv[sys.argv.index('resume')+1] if resumed else str(uuid.uuid4())
peer_paths=list(Path('SHARED_CONTEXT').glob('peer-evidence-*.json')) if Path('SHARED_CONTEXT').exists() else []
if peer_paths:
 peer=max(peer_paths,key=lambda p:p.stat().st_mtime_ns);bundle=json.loads(peer.read_text())
 reviews=[{'peer_id':r['id'],'finding':r['checkpoint']['summary'],'action':'Independently check peer receipt value '+r['checkpoint']['files']['NATIVE-PROVIDER-RECEIPT.json']['text']} for r in bundle['peers']]
 Path('PEER-REVIEW.json').write_text(json.dumps({'manifest_sha256':hashlib.sha256(peer.read_bytes()).hexdigest(),'reviews':reviews}))
if 'RSI-top dashboard' in prompt:
 Path('PORTFOLIO.json').write_text(json.dumps([dict(bottleneck='fixture',mechanism=str(i),expected_effect='increment',lineage='initial',verbalized_probability=.5) for i in range(2)]))
value=int(Path('value').read_text())+1;Path('value').write_text(str(value))
Path('NATIVE-PROVIDER-RECEIPT.json').write_text(json.dumps({'value':value}))
Path('SHARED_CHECKPOINT.json').write_text(json.dumps({'summary':'Measured fixture value '+str(value),'files':['NATIVE-PROVIDER-RECEIPT.json']}))
print(json.dumps({'type':'thread.started','thread_id':session}));print(json.dumps({'type':'turn.completed'}))
'''.replace('FAIL', repr(fail_resume)))

    def test_four_collaborators_read_peer_receipts_and_reuse_own_session_workspace(self):
        self.collaborative_cli()
        self.config.update(shareContext=True,sharingIntervalMinutes=.02)
        self.config['options'].update(branches=4, rounds=1,rsiTop=True,explorationMinutes=.04,exploitationMinutes=.02)
        runner,result=self.run_method('collaborative')
        self.assertEqual(result['status'],'COMPLETED')
        trace=json.loads((runner.output/'round-1-explore-COLLABORATION.json').read_text())
        self.assertEqual(len(trace['epochs']),2)
        one,two=[e['rows'] for e in trace['epochs']]
        self.assertEqual(len(one),4)
        for a,b in zip(one,two):
            self.assertEqual(a['workspace'],b['workspace'])
            self.assertEqual(a['provider_session'],b['provider_session'])
            review=json.loads((runner.output/b['id']/'PEER-REVIEW.json').read_text())
            self.assertEqual({r['peer_id'] for r in review['reviews']},{r['id'] for r in one})
            self.assertTrue(all('Measured fixture' in r['finding'] and 'receipt value' in r['action'] for r in review['reviews']))
        self.assertEqual(len([p for p in runner.output.glob('round-1-continue-epoch-1-branch-*') if p.is_dir()]),1)
        self.assertEqual((self.workspace/'value').read_text(),'1')

    def test_checkpoint_allowlist_rejects_private_traversal_and_symlinks(self):
        (self.workspace/'NATIVE-PROVIDER-RECEIPT.json').write_text('{"valid":true}')
        checkpoint=self.workspace/'SHARED_CHECKPOINT.json'
        checkpoint.write_text(json.dumps({'summary':'Observed fixture','files':['NATIVE-PROVIDER-RECEIPT.json']}))
        self.assertIn('NATIVE-PROVIDER-RECEIPT.json',m.Runner.shared_checkpoint(self.workspace)['files'])
        for name in ['../evaluator.py','/etc/passwd','.aws/credentials','auth.json']:
            checkpoint.write_text(json.dumps({'summary':'Fixture','files':[name]}))
            with self.assertRaises(ValueError):m.Runner.shared_checkpoint(self.workspace)
        (self.workspace/'link.json').symlink_to(self.evaluator)
        checkpoint.write_text(json.dumps({'summary':'Fixture','files':['link.json']}))
        with self.assertRaises(ValueError):m.Runner.shared_checkpoint(self.workspace)

    def test_selected_compact_workspace_cannot_escape_or_follow_symlink(self):
        seed=self.workspace/'compact';seed.mkdir()
        self.assertEqual(m.selected_workspace(self.workspace,{'workspace':'compact'}),seed)
        for value in ['../outside',str(seed),'.','.auto-research-methods']:
            with self.assertRaises((ValueError,FileNotFoundError)):m.selected_workspace(self.workspace,{'workspace':value})
        (self.workspace/'linked').symlink_to(seed)
        with self.assertRaises(ValueError):m.selected_workspace(self.workspace,{'workspace':'linked'})

    def test_failed_later_epoch_restores_exact_last_completed_bytes_and_receipts(self):
        self.collaborative_cli(fail_resume=True)
        self.config.update(shareContext=True,sharingIntervalMinutes=.015)
        self.config['options'].update(explorationMinutes=.04)
        output=self.folder/'fallback';output.mkdir()
        runner=m.Runner(self.config,self.plan,self.workspace,output)
        shutil.copy2(self.evaluator,runner.evaluator);runner.evaluator_hash=m.digest(runner.evaluator)
        initial=output/'initial';initial.mkdir();base=initial/'workspace';m.snapshot(self.workspace,base)
        leader={'id':'initial','workspace':str(base),'provider_session':None,'parent':None,'evaluation':runner.evaluate(base,initial)}
        candidates=runner.cooperative_cohort(leader,2,'explore',1,[{},{}])
        for row in candidates:
            self.assertTrue(row['evaluation']['valid'])
            self.assertIn('retained_completed_id',row)
            self.assertEqual((Path(row['workspace'])/'value').read_text(),'2')
            for name,digest in row['evaluation']['evidence'].items():self.assertEqual(m.digest(Path(row['workspace'])/name),digest)
            failed=json.loads((output/row['incomplete_successor']/'RESULT.json').read_text())
            self.assertFalse(failed['completed'])



if __name__ == '__main__':
    unittest.main()
